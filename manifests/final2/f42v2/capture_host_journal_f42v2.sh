#!/bin/bash
# Capture the shadowfax host-side records of the f42v2 run window (llama.cpp down 05:43:53 -> 06:18:34 PDT,
# 2026-10-16) that are not in f42v2.tgz: the serve unit's systemd accounting (memory peak per sample, a
# page-cache warmth indicator), the llama.cpp unit stop/start and the Hermes shadowfax-ensure-active-stack
# watchdog runs. Read-only (journalctl only; the Hermes cron database is not opened).
# usage (on computer4):  ./capture_host_journal_f42v2.sh > evidence/f42v2-host-journal.txt
set -euo pipefail
H=maplemercury@shadowfax
S="2026-10-16 05:43:00"; U="2026-10-16 06:25:00"
echo "# f42v2 host journal extract from shadowfax, captured $(date -Iseconds) by capture_host_journal_f42v2.sh (read-only)"
echo "## serve unit: journalctl --user -u colibri-final2-f42v2-serve.service (Started / Consumed lines)"
ssh $H "journalctl --user -u colibri-final2-f42v2-serve.service --since \"$S\" --until \"$U\" --no-pager -o short-iso | grep -E \"Started colibri-final2-f42v2-serve|Consumed\" | cut -c1-400"
echo "## llama.cpp unit: journalctl --user -u llama-server-active.service (state changes)"
ssh $H "journalctl --user -u llama-server-active.service --since \"$S\" --until \"$U\" --no-pager -o short-iso | grep -E \"Stopping|Stopped|Starting|Started|Consumed\" | cut -c1-300"
echo "## watchdog: journalctl --user -u shadowfax-ensure-active-stack.service (start/finish/failure lines)"
ssh $H "journalctl --user -u shadowfax-ensure-active-stack.service --since \"$S\" --until \"$U\" --no-pager -o short-iso | grep -E \"Starting|Finished|Failed|backend|recovered\" | cut -c1-300"
