#!/bin/bash
# Capture the shadowfax host-side records of the race run window (llama.cpp stopped 2026-10-16 02:38:14 PDT,
# restored at the time in race/meta/llama_after.txt) that are not in race.tgz: the serve unit's systemd accounting
# (memory peak per sample), the runner unit, the llama.cpp unit stop/start and the Hermes
# shadowfax-ensure-active-stack watchdog runs. Read-only (journalctl only; the Hermes cron database is not opened).
# usage (on computer4):  ./capture_host_journal_race.sh "2026-10-16 02:37:00" "<until>" > evidence/race-host-journal.txt
set -euo pipefail
H=maplemercury@shadowfax
S=${1:-"2026-10-16 02:37:00"}; U=${2:?until time}
echo "# race host journal extract from shadowfax, captured $(date -Iseconds) by capture_host_journal_race.sh (read-only), window $S .. $U"
echo "## serve unit: journalctl --user -u colibri-final2-race-serve.service (Started / Consumed lines)"
ssh $H "journalctl --user -u colibri-final2-race-serve.service --since \"$S\" --until \"$U\" --no-pager -o short-iso | grep -E \"Started colibri-final2-race-serve|Consumed\" | cut -c1-400"
echo "## runner unit: journalctl --user -u colibri-final2-race.service (state changes)"
ssh $H "journalctl --user -u colibri-final2-race.service --since \"$S\" --until \"$U\" --no-pager -o short-iso | grep -E \"Started|Finished|Deactivated|Consumed|Failed\" | cut -c1-300"
echo "## llama.cpp unit: journalctl --user -u llama-server-active.service (state changes)"
ssh $H "journalctl --user -u llama-server-active.service --since \"$S\" --until \"$U\" --no-pager -o short-iso | grep -E \"Stopping|Stopped|Starting|Started|Consumed\" | cut -c1-300"
echo "## watchdog: journalctl --user -u shadowfax-ensure-active-stack.service (start/finish/failure lines)"
ssh $H "journalctl --user -u shadowfax-ensure-active-stack.service --since \"$S\" --until \"$U\" --no-pager -o short-iso | grep -E \"Starting|Finished|Failed|backend|recovered|restarted_units\" | cut -c1-300"
