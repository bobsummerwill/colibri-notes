#!/bin/bash
# Capture the shadowfax host-side records of the pr1 run window (llama.cpp down 14:12:54 -> 15:20:51 PDT,
# 2026-10-15) that are not in pr1.tgz: the serve unit's systemd accounting (memory peak per sample), the
# llama.cpp unit stop/start, the Hermes shadowfax-ensure-active-stack watchdog runs, the Hermes cron worker
# scopes, and the Hermes cron execution records of that window. Read-only on shadowfax (journalctl, a
# read-only sqlite open of ~/.hermes/cron/executions.db, a read of ~/.hermes/cron/jobs.json for job names).
# usage (on computer4):  ./capture_host_journal_pr1.sh > evidence/pr1-host-journal.txt
set -euo pipefail
H=maplemercury@shadowfax
S="2026-10-15 14:12:50"; U="2026-10-15 15:24:00"
echo "# pr1 host journal extract from shadowfax, captured $(date -Iseconds) by capture_host_journal_pr1.sh (read-only)"
echo "## serve unit: journalctl --user -u colibri-final-pr1-serve.service (Started / Consumed lines)"
ssh $H "journalctl --user -u colibri-final-pr1-serve.service --since \"$S\" --until \"$U\" --no-pager -o short-iso | grep -E \"Started colibri-final-pr1-serve|Consumed\""
echo "## llama.cpp unit and Hermes scopes: journalctl --user (llama-server-active, hermes-*)"
ssh $H "journalctl --user --since \"$S\" --until \"$U\" --no-pager -o short-iso | grep -E \"llama-server-active|hermes-\" | grep -v colibri-final"
echo "## watchdog: journalctl --user -u shadowfax-ensure-active-stack.service"
ssh $H "journalctl --user -u shadowfax-ensure-active-stack.service --since \"$S\" --until \"$U\" --no-pager -o short-iso"
echo "## Hermes cron executions claimed between 14:12:50 and 15:24:00 (executions.db, read-only) with their job name/schedule"
# the script goes over stdin, so nothing is quoted inside the ssh command line
ssh $H python3 - <<'PY'
import json, os, sqlite3
home = os.path.expanduser("~")
db = sqlite3.connect("file:%s/.hermes/cron/executions.db?mode=ro" % home, uri=True)
jobs = json.load(open("%s/.hermes/cron/jobs.json" % home))
jobs = jobs.get("jobs", jobs) if isinstance(jobs, dict) else jobs
jobs = {j["id"]: j for j in (jobs.values() if isinstance(jobs, dict) else jobs) if isinstance(j, dict) and "id" in j}
cols = ("id", "job_id", "status", "claimed_at", "started_at", "finished_at", "error")
for r in db.execute("select %s from executions where claimed_at >= ? and claimed_at < ? order by claimed_at" % ",".join(cols),
                    ("2026-10-15T14:12:50", "2026-10-15T15:24:00")):
    rec = dict(zip(cols, r))
    j = jobs.get(rec["job_id"], {})
    rec["job_name"] = j.get("name")
    rec["job_schedule"] = (j.get("schedule") or {}).get("expr")
    print("cron_execution " + json.dumps(rec, sort_keys=True))
PY
