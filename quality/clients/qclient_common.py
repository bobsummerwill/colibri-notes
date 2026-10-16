"""Shared code for the quality-comparison clients (stdlib only, Python 3.8+).

Streams one OpenAI-style chat completion and records, per prompt: full content, reasoning
content (separately), token counts, TTFT, total time, finish_reason, server timings.
Output is append-only JSONL: one {"type": "run", ...} header per invocation and one
{"type": "result", ...} line per (prompt, sample). Re-running skips (prompt, sample) pairs that
already have a status "ok" result, so an interrupted run can simply be restarted.
"""
import datetime
import hashlib
import json
import os
import statistics
import time
import urllib.error
import urllib.request

CLIENT_VERSION = "quality-clients/2 (2026-10-15)"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPTS = os.path.normpath(os.path.join(HERE, "..", "prompts.json"))
DEFAULT_RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def load_prompts(path, only=None):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    prompts = sorted(doc["prompts"], key=lambda p: p["priority"])
    if only:
        wanted = [x.strip() for x in only.split(",") if x.strip()]
        unknown = set(wanted) - {p["id"] for p in prompts}
        if unknown:
            raise SystemExit(f"unknown prompt id(s): {sorted(unknown)}")
        prompts = [p for p in prompts if p["id"] in wanted]
    for p in prompts:
        digest = hashlib.sha256(p["prompt"].encode("utf-8")).hexdigest()
        if p.get("prompt_sha256") and p["prompt_sha256"] != digest:
            raise SystemExit(f"{p['id']}: prompt text does not match prompt_sha256")
    return doc, prompts


def messages_for(prompt):
    """The only thing a model ever sees: one user message with the prompt text."""
    return [{"role": "user", "content": prompt["prompt"]}]


def http_json(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 - probe only
        return None


def done_pairs(out_path):
    """(prompt_id, sample) pairs with an ok result already in out_path."""
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("type") == "result" and rec.get("status") == "ok":
                done.add((rec["prompt_id"], rec["sample"]))
    return done


def append_jsonl(path, rec):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ------------------------------------------------------------------ loop guard
LOOP_MIN_SPAN = 3000      # characters at the end of a channel that must be one repeated block
LOOP_MIN_REPEATS = 5      # ... repeated at least this many times
LOOP_PERIODS = (20, 2000)  # block length range (characters)
LOOP_CHECK_EVERY = 1500   # run the check after this many new characters in a channel


def find_loop(text, min_span=LOOP_MIN_SPAN, min_repeats=LOOP_MIN_REPEATS, periods=LOOP_PERIODS):
    """Return (period, span) if the END of text is an exact repetition, else None.

    The last R = max(min_span, min_repeats * p) characters have period p iff
    text[-R:] == text[-R - p:-p] shifted, i.e. text[n-R+p:n] == text[n-R:n-p]. Conservative on
    purpose: >= 3000 characters made of >= 5 identical copies of one 20-2000 character block.
    Under greedy decoding such a cycle does not end before max_tokens in practice, so stopping
    it changes nothing in the grade (the answer is graded exactly like finish_reason=length) and
    only saves GPU-window time.
    """
    n = len(text)
    lo, hi = periods
    for p in range(lo, min(hi, n // min_repeats) + 1):
        if text[n - 1] != text[n - 1 - p]:
            continue
        r = max(min_span, min_repeats * p)
        if r > n:
            continue
        if text[n - r + p:] == text[n - r:n - p]:
            # extend backwards to report the whole repeated span
            span = r
            while span + 1 <= n and text[n - span - 1] == text[n - span - 1 + p]:
                span += 1
            return p, span
    return None


def stream_chat(url, body, read_timeout_s=900, deadline=None, loop_abort=False):
    """POST a streaming chat completion; return a dict of everything measured.

    TTFT = time to the first NON-EMPTY delta in either channel. Empty deltas are not tokens:
    the opening role chunk carries content "", and colibri's keepalive is
    delta {"reasoning_content": ""} every ~10 s (counted separately as keepalives).

    deadline: time.monotonic() value. If it passes mid-stream the connection is closed (colibri's
    gateway cancels the generation on disconnect) and status becomes "aborted_window".
    loop_abort: if find_loop() fires on a channel, close the stream and record
    finish_reason "loop_abort" (status ok). Without it, a detected loop is only recorded.
    """
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json", "Accept": "text/event-stream"})
    content, reasoning = [], []
    out = {"http_status": None, "error": None, "finish_reason": None, "usage": None,
           "timings": None, "saw_done": False, "ttft_s": None, "ttft_content_s": None,
           "ttft_reasoning_s": None, "chunks": {"content": 0, "reasoning": 0, "keepalive": 0,
                                                "other": 0, "bad_json": 0},
           "tool_calls_seen": False, "response_model": None, "loop": None,
           "aborted": None}
    t0 = time.monotonic()
    since_check = {"content": 0, "reasoning": 0}

    def loop_hit(channel, parts):
        since_check[channel] = 0
        hit = find_loop("".join(parts))
        if hit and out["loop"] is None:
            out["loop"] = {"channel": channel, "period_chars": hit[0], "span_chars": hit[1],
                           "at_s": round(time.monotonic() - t0, 1),
                           "at_chars": sum(len(x) for x in parts)}
        return bool(hit)
    try:
        with urllib.request.urlopen(req, timeout=read_timeout_s) as r:
            out["http_status"] = r.status
            for raw in r:
                if deadline is not None and time.monotonic() > deadline:
                    out["aborted"] = "window"
                    break       # leaving the with-block closes the socket -> server cancels
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    out["saw_done"] = True
                    break
                try:
                    ev = json.loads(payload)
                except ValueError:
                    out["chunks"]["bad_json"] += 1
                    continue
                if ev.get("error"):
                    out["error"] = json.dumps(ev["error"])[:2000]
                if ev.get("model") and not out["response_model"]:
                    out["response_model"] = ev["model"]
                if ev.get("usage"):
                    out["usage"] = ev["usage"]
                if ev.get("timings"):
                    out["timings"] = ev["timings"]
                for ch in ev.get("choices") or []:
                    delta = ch.get("delta") or {}
                    t = time.monotonic() - t0
                    rc = delta.get("reasoning_content")
                    c = delta.get("content")
                    if rc:
                        reasoning.append(rc)
                        out["chunks"]["reasoning"] += 1
                        if out["ttft_reasoning_s"] is None:
                            out["ttft_reasoning_s"] = round(t, 3)
                    if c:
                        content.append(c)
                        out["chunks"]["content"] += 1
                        if out["ttft_content_s"] is None:
                            out["ttft_content_s"] = round(t, 3)
                    if (rc or c) and out["ttft_s"] is None:
                        out["ttft_s"] = round(t, 3)
                    if delta == {"reasoning_content": ""}:
                        out["chunks"]["keepalive"] += 1
                    elif not (rc or c) and set(delta) - {"role", "content", "reasoning_content"}:
                        out["chunks"]["other"] += 1   # e.g. tool_calls; role/empty deltas ignored
                    if delta.get("tool_calls"):
                        out["tool_calls_seen"] = True
                    if ch.get("finish_reason"):
                        out["finish_reason"] = ch["finish_reason"]
                    if rc:
                        since_check["reasoning"] += len(rc)
                    if c:
                        since_check["content"] += len(c)
                    for chan, parts in (("content", content), ("reasoning", reasoning)):
                        if since_check[chan] >= LOOP_CHECK_EVERY and loop_hit(chan, parts) \
                                and loop_abort and not out["finish_reason"]:
                            out["aborted"] = "loop"
                    if out["aborted"]:
                        break
                if out["aborted"]:
                    break
    except urllib.error.HTTPError as e:
        out["http_status"] = e.code
        try:
            out["error"] = f"HTTP {e.code}: " + e.read().decode("utf-8", "replace")[:2000]
        except Exception:  # noqa: BLE001
            out["error"] = f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001 - timeouts, resets: recorded, run continues
        out["error"] = f"{type(e).__name__}: {e}"
    total = time.monotonic() - t0
    out["total_s"] = round(total, 3)
    out["content"] = "".join(content)
    out["reasoning_content"] = "".join(reasoning)
    usage = out["usage"] or {}
    out["prompt_tokens"] = usage.get("prompt_tokens")
    out["completion_tokens"] = usage.get("completion_tokens")
    if out["ttft_s"] is not None and out["completion_tokens"] and total > out["ttft_s"]:
        # tokens after the first one, over the time after the first one
        out["decode_tok_s"] = round((out["completion_tokens"] - 1) / (total - out["ttft_s"]), 3)
    else:
        out["decode_tok_s"] = None
    if out["aborted"] == "loop":
        out["finish_reason"] = "loop_abort"   # graded like "length"; the text is kept as written
        out["status"] = "ok"
    elif out["aborted"] == "window":
        out["status"] = "aborted_window"      # no answer: left out of both models' totals
    elif out["error"]:
        out["status"] = "error"
    elif not out["saw_done"] or not out["finish_reason"]:
        out["status"] = "incomplete"
    else:
        out["status"] = "ok"
    return out


def result_record(arm, backend, prompt, sample, seed, body, measured):
    params = {k: v for k, v in body.items() if k != "messages"}
    rec = {"type": "result", "arm": arm, "backend": backend, "prompt_id": prompt["id"],
           "category": prompt["category"], "priority": prompt["priority"], "sample": sample,
           "seed": seed, "max_tokens": prompt_budget(prompt, body), "params": params,
           "prompt_sha256": hashlib.sha256(prompt["prompt"].encode("utf-8")).hexdigest(),
           "started_at": None}
    rec.update(measured)
    return rec


def prompt_budget(prompt, body):
    return body.get("max_tokens") or body.get("max_completion_tokens") or prompt["max_tokens"]


HEAD_TO_HEAD_QWEN_ARMS = ("qwen-hermes", "qwen-greedy")


def qwen_lengths(paths):
    """{prompt_id: {"max": tokens, "n": records, "length_hits": n, "loop_hits": n}} from Qwen
    head-to-head results.

    Uses ok records only. A finish_reason=length or loop_abort record counts as max_tokens (its
    true length is unknown). length_hits must be 0 for the GPU-stage gate on a prompt whose budget
    could still be raised; loop_hits are only reported (scored as written).
    """
    out = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("type") != "result" or rec.get("status") != "ok" \
                        or rec.get("arm") not in HEAD_TO_HEAD_QWEN_ARMS:
                    continue
                d = out.setdefault(rec["prompt_id"], {"max": 0, "n": 0, "length_hits": 0, "loop_hits": 0})
                toks = rec.get("completion_tokens") or 0
                if rec.get("finish_reason") in ("length", "loop_abort"):
                    # a larger budget fixes a "length" truncation but not a loop
                    d["length_hits" if rec["finish_reason"] == "length" else "loop_hits"] += 1
                    toks = max(toks, rec.get("max_tokens") or 0)   # would have run to max_tokens
                d["max"] = max(d["max"], toks)
                d["n"] += 1
    return out


def median_or(values, default):
    vals = [v for v in values if v]
    return statistics.median(vals) if vals else default


def summary_line(rec):
    return (f"{rec['prompt_id']:18s} s{rec['sample']} {rec['status']:10s} "
            f"finish={rec.get('finish_reason')} tok={rec.get('completion_tokens')} "
            f"ttft={rec.get('ttft_s')}s total={rec.get('total_s')}s "
            f"rate={rec.get('decode_tok_s')} reasoning_chars={len(rec.get('reasoning_content') or '')}"
            + (f" LOOP={rec['loop']}" if rec.get("loop") else "")
            + (f" ERROR={rec['error'][:200]}" if rec.get("error") else ""))
