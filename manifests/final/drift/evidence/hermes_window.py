#!/usr/bin/env python3
"""Read-only capture, on shadowfax, of what Hermes did while llama.cpp was stopped for the drift run.
Window: run_drift.sh stopped llama-server-active at 2026-10-15 17:36:40 PDT and restored it at 18:58:44 PDT
(drift/meta/llama_before.txt, llama_after.txt, run.log)."""
import datetime, json, os, sqlite3, subprocess

SINCE, UNTIL = "2026-10-15 17:33:00", "2026-10-15 19:00:00"
print("# Hermes side effects during the drift GPU window (llama.cpp down 2026-10-15 17:36:40 - 18:58:44 PDT)")
print("# captured read-only on shadowfax by", os.environ.get("USER"), "at",
      subprocess.run(["date", "--iso-8601=seconds"], capture_output=True, text=True).stdout.strip())
print()
print("## 1. shadowfax-ensure-active-stack (user timer, every 5 min): one line per run")
print("# source: journalctl --user -u shadowfax-ensure-active-stack.service --since '%s' --until '%s' -o json" % (SINCE, UNTIL))
out = subprocess.run(["journalctl", "--user", "-u", "shadowfax-ensure-active-stack.service", "--since", SINCE,
                      "--until", UNTIL, "--no-pager", "-o", "json"], capture_output=True, text=True).stdout
runs, cur = [], None
for line in out.splitlines():
    e = json.loads(line)
    msg = e.get("MESSAGE", "")
    if isinstance(msg, list):
        msg = bytes(msg).decode(errors="replace")
    ts = datetime.datetime.fromtimestamp(int(e["__REALTIME_TIMESTAMP"]) / 1e6).strftime("%H:%M:%S")
    if msg.startswith("Starting "):
        cur = {"start": ts, "body": []}
        runs.append(cur)
    elif cur is not None and e.get("SYSLOG_IDENTIFIER") == "env":
        cur["body"].append(msg)
    elif cur is not None and (msg.startswith("Finished ") or msg.startswith("Failed to start ")):
        cur["end"] = ts
        cur["result"] = "ok" if msg.startswith("Finished ") else "FAILED"
print("%-8s %-8s %-6s %-9s %-7s %-10s %s" % ("start", "end", "result", "backend", "proxy", "recovered", "restarted_units / probe error"))
for r in runs:
    try:
        d = json.loads("\n".join(r["body"]))
    except ValueError:
        d = {}
    st = d.get("statuses", {})
    print("%-8s %-8s %-6s %-9s %-7s %-10s %s / %s" % (
        r["start"], r.get("end", "?"), r.get("result", "?"), st.get("backend"), st.get("local_proxy"),
        d.get("recovered"), d.get("restarted_units"), d.get("exact_output_probe", {}).get("error", "-")))
print()
print("## 2. Hermes cron executions claimed 17:00-19:05 PDT (job name, status, times, error, delivery outcome)")
print("# source: ~/.hermes/cron/executions.db (opened mode=ro), names from ~/.hermes/cron/jobs.json")
names = {j.get("id"): j.get("name") for j in json.load(open(os.path.expanduser("~/.hermes/cron/jobs.json")))["jobs"]}
db = sqlite3.connect("file:%s?mode=ro" % os.path.expanduser("~/.hermes/cron/executions.db"), uri=True)
for job, status, claimed, started, finished, err, delivery in db.execute(
        "select job_id, status, claimed_at, started_at, finished_at, coalesce(error, ''), coalesce(delivery_outcome, '') "
        "from executions where claimed_at >= ? and claimed_at < ? order by claimed_at",
        ("2026-10-15T17:00", "2026-10-15T19:05")):
    print("%s %-32s %-9s started %s finished %s error=%r delivery=%s" % (
        job, names.get(job, "?"), status, started[11:19], (finished or "")[11:19], err[:80], delivery))
