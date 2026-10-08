#!/usr/bin/env bash
# Thoroughly stop a vLLM server started by run_server.sh: kill the recorded pid AND its
# whole descendant tree, then any lingering process still bound to the port (vLLM 0.19.x
# spawns APIServer/EngineCore children that survive a bare `kill <pid>` — this orphaning
# caused the Phase 4.4 port-collision incident). Usage: kill_server.sh <server.pid file> <port>
set -uo pipefail
PIDFILE="$1"; PORT="$2"

_kill_tree() {
  local pid="$1" c
  for c in $(pgrep -P "$pid" 2>/dev/null); do _kill_tree "$c"; done
  kill "$pid" 2>/dev/null
}

[ -f "$PIDFILE" ] && _kill_tree "$(cat "$PIDFILE")"
# Catch children reparented to init (parent already gone): match the exact port in cmdline.
pkill -f "vllm serve .*--port ${PORT}( |$)" 2>/dev/null
# Wait until nothing answers on the port anymore.
for i in $(seq 1 30); do
  if ! curl -s -o /dev/null --max-time 2 "http://127.0.0.1:${PORT}/health"; then break; fi
  sleep 2
done
# Last resort: find whatever still LISTENS on the port (surviving children reparented to
# init don't carry the vllm-serve cmdline) and SIGKILL it.
for lpid in $(ss -ltnp 2>/dev/null | grep ":${PORT} " | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do
  kill -9 "$lpid" 2>/dev/null
done
sleep 1
exit 0
