#!/bin/bash
# Capture the shadowfax host-side records of the f42fix run window that are not in f42fix.tgz: the serve
# unit's systemd accounting (memory peak per sample), the runner unit, the llama.cpp unit stop/start and
# the Hermes shadowfax-ensure-active-stack watchdog runs. Read-only (journalctl only).
# usage (on computer4; the remote part is piped to bash -s):
#   ./capture_host_journal_f42fix.sh "2026-10-16 09:44:00" "2026-10-16 10:45:00" > evidence/f42fix-host-journal.txt
set -euo pipefail
S="$1"; U="$2"
echo "# f42fix host journal extract from shadowfax ($S -> $U PDT), captured $(date -Iseconds) by capture_host_journal_f42fix.sh (read-only)"
ssh maplemercury@shadowfax 'bash -s' <<REMOTE
echo "## serve unit: journalctl --user -u colibri-final2-f42fix-serve.service (Started / Consumed lines)"
journalctl --user -u colibri-final2-f42fix-serve.service --since "$S" --until "$U" --no-pager -o short-iso | grep -E "Started colibri-final2-f42fix-serve|Consumed" | cut -c1-400
echo "## runner unit: journalctl --user -u colibri-final2-f42fix.service (Started / Consumed / Deactivated lines)"
journalctl --user -u colibri-final2-f42fix.service --since "$S" --until "$U" --no-pager -o short-iso | grep -E "Started|Consumed|Deactivated|Finished|Failed" | cut -c1-300
echo "## llama.cpp unit: journalctl --user -u llama-server-active.service (state changes)"
journalctl --user -u llama-server-active.service --since "$S" --until "$U" --no-pager -o short-iso | grep -E "Stopping|Stopped|Starting|Started|Consumed" | cut -c1-300
echo "## watchdog: journalctl --user -u shadowfax-ensure-active-stack.service (start/finish/failure lines)"
journalctl --user -u shadowfax-ensure-active-stack.service --since "$S" --until "$U" --no-pager -o short-iso | grep -E "Starting|Finished|Failed|backend|recovered" | cut -c1-300 || true
REMOTE
