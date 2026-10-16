#!/usr/bin/env python3
"""Quality-comparison client for DeepSeek V4 Flash REAP-150B on colibri (:8090).

Runs ON shadowfax while the REAP server is up (the GPU stage starts it after stopping
llama.cpp; this client never starts or stops anything). Requests are sequential, in prompt
priority order. Decoding is greedy (the DSV4 serve engine is greedy; temperature 0 is sent to
say so) and thinking is OFF (enable_thinking=false = the serve default without COLI_THINK).

Protocol v2 order: the Qwen arms run FIRST (results/qwen-hermes.jsonl, results/qwen-greedy.jsonl).
This client reads them to estimate how long each REAP answer will be.

Window guard (v2):
  expected tokens E(p) = min(max_tokens, 1.5 * r * Q(p))
      Q(p) = longest ok head-to-head Qwen completion for p (a length-truncated one counts as
             max_tokens); r = median REAP/Qwen token ratio over the prompts REAP has answered so
             far, clamped to [0.5, 2.0], 1.0 before the first answer. No Qwen data: E = max_tokens.
  expected time  = prompt bytes / 3 / prefill rate + E / decode rate + 20 s
  soft budget    (--budget-min): a prompt STARTS only if elapsed + expected time fits. Skipped
                 prompts get status "skipped_budget". An answer that has started is never cut by
                 the soft budget, whatever its length.
  hard stop      (--hard-stop-min): an answer still running then is aborted (connection closed,
                 the gateway cancels the generation) with status "aborted_window". Both statuses
                 are "no answer": grade.py leaves the prompt out of both models' totals.
  loop guard     a greedy answer whose last >= 3000 characters are one block repeated >= 5 times
                 is stopped with finish_reason "loop_abort" (graded like "length").
  retry pass     prompts that ended in error/incomplete are retried once at the end if they fit.

Gate (--dry-run, run BEFORE llama.cpp is stopped): exits 2 if the Qwen results are missing for a
prompt, or a head-to-head Qwen answer hit max_tokens on a prompt whose budget could still be
raised (--allow-qwen-truncation overrides). Truncation at a prompt's context-ceiling budget
(budget_at_ceiling) and loop-guard stops are warnings: scored as written, for either model.

Exit codes: 0 done (some prompts may be skipped), 2 gate failed, 3 server unreachable mid-run
(the runner may restart colibri and run the client again: it resumes, skipping ok answers).

Example (on shadowfax, from ~/colibri-bench/quality):
  python3 clients/colibri_client.py --dry-run                     # gate + plan, no server needed
  python3 clients/colibri_client.py --wait-ready-s 1200 --budget-min 120 --hard-stop-min 150
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qclient_common as qc  # noqa: E402

RATIO_CLAMP = (0.5, 2.0)
SAFETY = 1.5


def wait_ready(base, max_wait_s):
    t0 = time.monotonic()
    while True:
        models = qc.http_json(base + "/v1/models", timeout=5)
        if isinstance(models, dict) and models.get("data"):
            return models, round(time.monotonic() - t0, 1)
        if time.monotonic() - t0 > max_wait_s:
            raise SystemExit(f"colibri at {base} not ready after {max_wait_s}s")
        time.sleep(5)


def server_up(base):
    models = qc.http_json(base + "/v1/models", timeout=10)
    return isinstance(models, dict) and bool(models.get("data"))


def expected_tokens(p, qlen, ratios):
    q = (qlen.get(p["id"]) or {}).get("max")
    if not q:
        return p["max_tokens"], None
    r = qc.median_or(ratios, 1.0)
    r = min(max(r, RATIO_CLAMP[0]), RATIO_CLAMP[1])
    return min(p["max_tokens"], int(math.ceil(SAFETY * r * q))), r


def expected_seconds(p, tokens, a, rate):
    return len(p["prompt"].encode("utf-8")) / 3.0 / a.prefill_rate + tokens / rate + 20


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default="deepseek-v4-colibri")
    ap.add_argument("--prompts", default=qc.DEFAULT_PROMPTS)
    ap.add_argument("--out", default=None, help="JSONL output (default results/reap.jsonl)")
    ap.add_argument("--only", default=None, help="comma-separated prompt ids")
    ap.add_argument("--qwen-results", default=None,
                    help="comma-separated Qwen JSONL files for length estimates "
                         "(default results/qwen-hermes.jsonl,results/qwen-greedy.jsonl)")
    ap.add_argument("--allow-qwen-truncation", action="store_true",
                    help="do not fail the gate when a head-to-head Qwen answer hit max_tokens")
    ap.add_argument("--think", action="store_true",
                    help="NOT part of the protocol: enable_thinking=true (writes results/reap-think.jsonl)")
    ap.add_argument("--budget-min", type=float, default=120.0,
                    help="soft budget: a prompt starts only if its expected time fits (minutes)")
    ap.add_argument("--hard-stop-min", type=float, default=150.0,
                    help="hard stop: abort the answer in flight at this point (minutes)")
    ap.add_argument("--decode-rate", type=float, default=3.6,
                    help="tok/s assumed until a rate has been observed")
    ap.add_argument("--prefill-rate", type=float, default=18.0, help="prompt tok/s assumed")
    ap.add_argument("--wait-ready-s", type=float, default=0, help="poll /v1/models up to this long first")
    ap.add_argument("--read-timeout-s", type=float, default=900)
    ap.add_argument("--no-loop-guard", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="gate + plan only; contacts no server")
    a = ap.parse_args()
    if a.hard_stop_min < a.budget_min:
        ap.error("--hard-stop-min must be >= --budget-min")

    arm = "reap-think" if a.think else "reap"
    out = a.out or os.path.join(qc.DEFAULT_RESULTS, arm + ".jsonl")
    doc, prompts = qc.load_prompts(a.prompts, a.only)
    qpaths = (a.qwen_results.split(",") if a.qwen_results else
              [os.path.join(qc.DEFAULT_RESULTS, x + ".jsonl") for x in qc.HEAD_TO_HEAD_QWEN_ARMS])
    qlen = qc.qwen_lengths(qpaths)

    # ---------------------------------------------------------------- gate
    problems, warnings = [], []
    for p in prompts:
        d = qlen.get(p["id"])
        if not d:
            problems.append(f"{p['id']}: no ok head-to-head Qwen result")
            continue
        if d["loop_hits"]:   # a larger budget would not help: scored as written, listed
            warnings.append(f"{p['id']}: {d['loop_hits']} Qwen answer(s) stopped by the loop guard (scored as written)")
        if d["length_hits"] and p.get("budget_at_ceiling"):
            # cannot be raised under --ctx 8192: scored as written (same limit for REAP), listed
            warnings.append(f"{p['id']}: {d['length_hits']} Qwen answer(s) hit the ceiling budget "
                            f"max_tokens={p['max_tokens']} (scored as written)")
        elif d["length_hits"]:
            problems.append(f"{p['id']}: {d['length_hits']} Qwen answer(s) hit max_tokens={p['max_tokens']}"
                            " -> raise this prompt's budget and re-run it in every Qwen arm")
    done = qc.done_pairs(out)
    if a.dry_run:
        print(f"Qwen results: {[x for x in qpaths if os.path.exists(x)]}")
        cum = 0.0
        for p in prompts:
            toks, r = expected_tokens(p, qlen, [])
            secs = expected_seconds(p, toks, a, a.decode_rate)
            if (p["id"], 1) not in done:
                cum += secs
            print(f"  {p['priority']:2d} {p['id']:18s} max_tokens={p['max_tokens']:5d} "
                  f"qwen_max={(qlen.get(p['id']) or {}).get('max')} expected_tokens={toks:5d} "
                  f"expected={secs / 60:5.1f} min cumulative={cum / 60:6.1f} min"
                  + ("  (done)" if (p["id"], 1) in done else ""))
        print(f"soft budget {a.budget_min} min, hard stop {a.hard_stop_min} min, "
              f"decode {a.decode_rate} tok/s assumed")
        for x in warnings:
            print("GATE WARNING:", x)
        for x in problems:
            print("GATE:", x)
        if problems and not a.allow_qwen_truncation:
            print("GATE FAILED: run/finish the Qwen arms (and fix budgets) before the GPU window")
            sys.exit(2)
        print("GATE OK")
        return

    models, ready_s = wait_ready(a.base, a.wait_ready_s)
    ids = [m.get("id") for m in models.get("data", [])]
    if a.model not in ids:
        raise SystemExit(f"model {a.model!r} not served (got {ids})")
    served_ctx = next((m.get("context_length") for m in models["data"] if m.get("id") == a.model), None)
    t_start = time.monotonic()
    hard_deadline = t_start + a.hard_stop_min * 60
    qc.append_jsonl(out, {"type": "run", "arm": arm, "backend": "colibri", "base": a.base,
                          "client": qc.CLIENT_VERSION, "started_at": qc.now_iso(),
                          "server_model": a.model, "server": {"models": models},
                          "ready_wait_s": ready_s, "budget_min": a.budget_min,
                          "hard_stop_min": a.hard_stop_min, "qwen_lengths": qlen,
                          "gate_problems": problems, "gate_warnings": warnings,
                          "loop_guard": not a.no_loop_guard,
                          "prompt_ids": [p["id"] for p in prompts],
                          "env_hint": {k: os.environ.get(k) for k in ("COLI_THINK",)}})
    rates, ratios = [], []
    # ratios from answers of an earlier invocation (resume)
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("type") == "result" and rec.get("status") == "ok":
                    q = (qlen.get(rec["prompt_id"]) or {}).get("max")
                    if q and rec.get("completion_tokens") and rec.get("finish_reason") == "stop":
                        ratios.append(rec["completion_tokens"] / q)
                    if rec.get("decode_tok_s") and (rec.get("completion_tokens") or 0) >= 64:
                        rates.append(rec["decode_tok_s"])

    def attempt(p, attempt_no):
        """Run one prompt if it fits. Returns the record's status, or None if it was not run."""
        body = {"model": a.model, "messages": qc.messages_for(p), "max_tokens": p["max_tokens"],
                "stream": True, "stream_options": {"include_usage": True}, "temperature": 0.0,
                "enable_thinking": bool(a.think)}
        if a.think:
            body["max_tokens"] = p["max_tokens"] + 12288
        rate = qc.median_or(rates, a.decode_rate)
        toks, r = expected_tokens(p, qlen, ratios)
        if a.think:
            toks = body["max_tokens"]
        est_s = expected_seconds(p, toks, a, rate)
        elapsed = time.monotonic() - t_start
        if elapsed + est_s > a.budget_min * 60:
            if attempt_no == 1:
                qc.append_jsonl(out, {
                    "type": "result", "arm": arm, "backend": "colibri", "prompt_id": p["id"],
                    "category": p["category"], "priority": p["priority"], "sample": 1,
                    "status": "skipped_budget", "elapsed_s": round(elapsed, 1),
                    "expected_tokens": toks, "expected_s": round(est_s, 1), "assumed_rate": rate,
                    "length_ratio": r, "started_at": qc.now_iso()})
            print(f"SKIP {p['id']}: expected {est_s / 60:.1f} min ({toks} tok at {rate:.2f} tok/s) "
                  f"does not fit ({elapsed / 60:.1f} of {a.budget_min} min used)", flush=True)
            return None
        started = qc.now_iso()
        measured = qc.stream_chat(a.base + "/v1/chat/completions", body, a.read_timeout_s,
                                  deadline=hard_deadline, loop_abort=not a.no_loop_guard)
        rec = qc.result_record(arm, "colibri", p, 1, None, body, measured)
        if served_ctx and rec.get("prompt_tokens"):
            # the engine clamps max_tokens to ctx - prompt: prove the budget was the one sent
            rec["ctx_clamped"] = rec["prompt_tokens"] + body["max_tokens"] > served_ctx
            rec["served_ctx"] = served_ctx
        rec.update({"started_at": started, "attempt": attempt_no,
                    "thinking": "on" if a.think else "off (enable_thinking=false)",
                    "expected_tokens": toks, "expected_s": round(est_s, 1),
                    "assumed_rate": rate, "length_ratio": r})
        qc.append_jsonl(out, rec)
        if rec.get("decode_tok_s") and (rec.get("completion_tokens") or 0) >= 64:
            rates.append(rec["decode_tok_s"])
        q = (qlen.get(p["id"]) or {}).get("max")
        if rec["status"] == "ok" and q and rec.get("finish_reason") == "stop" and rec.get("completion_tokens"):
            ratios.append(rec["completion_tokens"] / q)
        print(qc.summary_line(rec), f"| expected {toks} tok | elapsed "
              f"{(time.monotonic() - t_start) / 60:.1f} min", flush=True)
        return rec["status"]

    failed = []
    exit_code = 0
    for p in prompts:
        if (p["id"], 1) in done:
            print(f"skip (done) {p['id']}", flush=True)
            continue
        if time.monotonic() > hard_deadline:
            break
        status = attempt(p, 1)
        if status in ("error", "incomplete"):
            if not server_up(a.base):
                print("colibri is not answering /v1/models: stopping (exit 3)", flush=True)
                exit_code = 3
                break
            failed.append(p)
    if exit_code == 0:
        for p in failed:                      # one retry each, only if it still fits
            if time.monotonic() > hard_deadline:
                break
            attempt(p, 2)
    qc.append_jsonl(out, {"type": "run_end", "arm": arm, "ended_at": qc.now_iso(),
                          "elapsed_s": round(time.monotonic() - t_start, 1), "exit_code": exit_code})
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
