#!/usr/bin/env python3
"""Quality-comparison client for Qwen3.8-27B on llama.cpp (the Hermes brain, :18080).

Runs ON shadowfax (llama-server-active listens there). Requests are strictly sequential,
one at a time, and before each one the client waits until at least one of the server's two
slots is idle, so Hermes always keeps a slot. It never restarts or reconfigures llama.cpp.

Arms (see prompts.json "protocol"):
  default        Hermes production sampling (= llama-server-active's CLI defaults, sent
                 explicitly): temperature 0.7, top_p 0.95, top_k 20, min_p 0, presence_penalty 0.5,
                 repeat_penalty 1.0; thinking = server default (--reasoning off); seeds 1,2,3.
  --greedy       temperature 0, top_k 1, no penalties, one sample (deterministic comparator for
                 the greedy, penalty-free REAP engine).
  --think        informational only: chat_template_kwargs.enable_thinking=true, budget + 12288.

Protocol v2: run these arms BEFORE the REAP GPU window; the colibri client reads
results/qwen-hermes.jsonl and results/qwen-greedy.jsonl for its length estimates and its gate
(no head-to-head Qwen answer may end with finish_reason=length).

Examples (on shadowfax, from ~/colibri-bench/quality):
  python3 clients/qwen_llama_client.py                      # -> results/qwen-hermes.jsonl
  python3 clients/qwen_llama_client.py --greedy             # -> results/qwen-greedy.jsonl
  python3 clients/qwen_llama_client.py --think              # -> results/qwen-think.jsonl
  python3 clients/qwen_llama_client.py --only M1_strings --samples 1 --dry-run
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qclient_common as qc  # noqa: E402

HERMES_SAMPLING = {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                   "presence_penalty": 0.5, "repeat_penalty": 1.0}
THINK_EXTRA_TOKENS = 12288


def wait_for_free_slot(base, max_wait_s):
    """Block until llama.cpp reports at least one idle slot (or /slots is unavailable)."""
    t0 = time.monotonic()
    while True:
        slots = qc.http_json(base + "/slots", timeout=5)
        if not isinstance(slots, list):
            return "slots-unavailable"
        busy = sum(1 for s in slots if s.get("is_processing"))
        if busy < len(slots):
            return f"{busy}/{len(slots)} busy"
        if time.monotonic() - t0 > max_wait_s:
            return f"gave-up-waiting {busy}/{len(slots)} busy"
        time.sleep(5)


def server_info(base):
    props = qc.http_json(base + "/props") or {}
    models = qc.http_json(base + "/v1/models") or {}
    model_id = None
    try:
        model_id = models["data"][0]["id"]
    except (KeyError, IndexError, TypeError):
        pass
    dgs = (props.get("default_generation_settings") or {})
    params = dgs.get("params", dgs)
    return model_id, {
        "build_info": props.get("build_info"), "model_path": props.get("model_path"),
        "total_slots": props.get("total_slots"), "n_ctx": dgs.get("n_ctx"),
        "server_sampling_defaults": {k: params.get(k) for k in (
            "temperature", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty")},
        "chat_template_sha256": __import__("hashlib").sha256(
            (props.get("chat_template") or "").encode()).hexdigest(),
        "health": qc.http_json(base + "/health"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:18080")
    ap.add_argument("--prompts", default=qc.DEFAULT_PROMPTS)
    ap.add_argument("--out", default=None, help="JSONL output (default results/<arm>.jsonl)")
    ap.add_argument("--only", default=None, help="comma-separated prompt ids")
    ap.add_argument("--samples", type=int, default=None, help="default 3 (1 for --greedy/--think)")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--pause-s", type=float, default=2.0, help="pause between requests")
    ap.add_argument("--slot-wait-max-s", type=float, default=900)
    ap.add_argument("--read-timeout-s", type=float, default=900)
    ap.add_argument("--no-loop-guard", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.greedy and a.think:
        ap.error("--greedy and --think are separate arms")

    arm = "qwen-greedy" if a.greedy else ("qwen-think" if a.think else "qwen-hermes")
    out = a.out or os.path.join(qc.DEFAULT_RESULTS, arm + ".jsonl")
    nsamples = a.samples or (1 if (a.greedy or a.think) else 3)
    seeds = [int(s) for s in a.seeds.split(",")]
    if nsamples > len(seeds):
        ap.error("need at least as many --seeds as --samples")
    doc, prompts = qc.load_prompts(a.prompts, a.only)

    model_id, info = (None, {}) if a.dry_run else server_info(a.base)
    if not a.dry_run and (info.get("health") or {}).get("status") != "ok":
        raise SystemExit(f"llama.cpp at {a.base} is not healthy: {info.get('health')}")
    if not a.dry_run:
        qc.append_jsonl(out, {"type": "run", "arm": arm, "backend": "llama.cpp", "base": a.base,
                              "client": qc.CLIENT_VERSION, "started_at": qc.now_iso(),
                              "server_model": model_id, "server": info, "samples": nsamples,
                              "seeds": seeds[:nsamples], "prompt_ids": [p["id"] for p in prompts]})
    done = set() if a.dry_run else qc.done_pairs(out)

    for sample in range(1, nsamples + 1):          # sample-major: every prompt once, then again
        seed = seeds[sample - 1]
        for p in prompts:
            if (p["id"], sample) in done:
                print(f"skip (done) {p['id']} s{sample}", flush=True)
                continue
            body = {"model": model_id or "qwen", "messages": qc.messages_for(p),
                    "max_tokens": p["max_tokens"] + (THINK_EXTRA_TOKENS if a.think else 0),
                    "stream": True, "stream_options": {"include_usage": True}}
            if a.greedy:
                # pure argmax like colibri (which has no penalties): override the server's
                # presence_penalty 0.5 default too
                body.update({"temperature": 0.0, "top_k": 1, "presence_penalty": 0.0,
                             "repeat_penalty": 1.0, "seed": seed})
            else:
                body.update(HERMES_SAMPLING)
                body["seed"] = seed
            if a.think:
                body["chat_template_kwargs"] = {"enable_thinking": True}
            if a.dry_run:
                print(json.dumps({k: v for k, v in body.items() if k != "messages"}),
                      p["id"], f"s{sample}", flush=True)
                continue
            slot_state = wait_for_free_slot(a.base, a.slot_wait_max_s)
            started = qc.now_iso()
            # loop guard: abort only in the greedy (deterministic) arm, as for the greedy REAP;
            # sampled arms just record a detection (a sampled model can leave a loop)
            measured = qc.stream_chat(a.base + "/v1/chat/completions", body, a.read_timeout_s,
                                      loop_abort=a.greedy and not a.no_loop_guard)
            rec = qc.result_record(arm, "llama.cpp", p, sample, seed, body, measured)
            rec["started_at"] = started
            rec["slot_state_before"] = slot_state
            rec["thinking"] = "on (chat_template_kwargs)" if a.think else "server default (--reasoning off)"
            qc.append_jsonl(out, rec)
            print(qc.summary_line(rec), flush=True)
            time.sleep(a.pause_s)
    if not a.dry_run:
        qc.append_jsonl(out, {"type": "run_end", "arm": arm, "ended_at": qc.now_iso()})


if __name__ == "__main__":
    main()
