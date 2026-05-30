#!/usr/bin/env bash
# Free up the GreenVision demo ports if a previous run left zombies.
#
# Useful when a crashed Uvicorn / Next.js / MLflow UI is still holding a port
# and demo.sh refuses to start. Safe to run any time.

set -euo pipefail

PORTS=(8000 3000 5001)

for port in "${PORTS[@]}"; do
  pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
  if [ -n "$pids" ]; then
    printf "→ Killing PIDs on :%s → %s\n" "$port" "$pids"
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
  else
    printf "✓ :%s is free\n" "$port"
  fi
done

printf "\n✓ Demo ports reset. You can now run ./scripts/demo.sh\n"
