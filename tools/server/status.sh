#!/usr/bin/env bash
# icutwin on the AutoDL host: one-screen status of the pipeline steps started with tools/run.sh.
cd /root/autodl-tmp/logs || exit 1
for log in *.log; do
  name=${log%.log}
  [ "$name" = setup ] && continue
  if [ -e "$name.exit" ]; then
    read -r code secs < "$name.exit"
    printf "%-16s %s (exit %s, %ss)\n" "$name" "$([ "$code" = 0 ] && echo done || echo FAILED)" "$code" "$secs"
  else
    printf "%-16s running: %s\n" "$name" "$(tail -c 600 "$log" | tr '\r' '\n' | grep -v '^\s*$' | tail -1 | cut -c1-150)"
  fi
done
echo "load: $(cut -d' ' -f1-3 /proc/loadavg)  mem used/avail GB: $(free -g | awk '/Mem/ {print $3"/"$7}')  disk free: $(df -h /root/autodl-tmp | awk 'NR==2 {print $4}')"
