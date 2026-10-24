#!/bin/bash
# chain19.sh: once final18 has ended (llama restored by its EXIT trap), build f19/all (llama is up; the build only
# competes with an idle llama), then start final19.
B=~/colibri-bench/final19
until [ -f ~/colibri-bench/final18/chain.log ] && grep -q STOP_AT ~/colibri-bench/final18/chain.log && \
      ! systemctl --user is-active -q colibri-final18 && ! systemctl --user is-active -q chain-final18; do sleep 30; done
echo "$(date -Is) final18 ended; building f19-all" >> $B/chain.log
sed 's#final17/builds#final19/builds#' ~/colibri-bench/final17/build_f17.sh > $B/build_f19.sh && chmod +x $B/build_f19.sh
$B/build_f19.sh f19-all=f19/all >> $B/chain.log 2>&1
grep -q '^EXIT=0' $B/builds/f19-all.build.log || { echo "$(date -Is) build failed" >> $B/chain.log; exit 1; }
sleep 30
T=$(date -d "+90 sec" "+%F %T")
echo "$(date -Is) final19 STOP_AT=$T" >> $B/chain.log
systemd-run --user --unit=colibri-final19 --collect -E STOP_AT="$T" $B/run_final19.sh >> $B/chain.log 2>&1
