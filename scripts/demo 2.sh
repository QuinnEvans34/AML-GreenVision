#!/usr/bin/env bash
# One-command launcher for the GreenVision W10D3 demo.
#
# Starts three processes in parallel (output multiplexed by `concurrently`):
#   • FastAPI inference backend         http://localhost:8000
#   • Next.js dashboard                  http://localhost:3000
#   • MLflow UI (Models tab is the prize) http://localhost:5001
#
# Pre-flight checks before launch:
#   1. The three ports are free (else: tells you to run demo_reset.sh).
#   2. The Production model loads from the MLflow Registry.
#   3. web/node_modules and web/public/training_data.json are present.
#
# Usage:
#   ./scripts/demo.sh
#
# Stop everything: Ctrl+C in this terminal (kills all three).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ───────────────────────────────────────────────────────────────
# Pretty output helpers
# ───────────────────────────────────────────────────────────────
say()  { printf "\033[1;36m→\033[0m %s\n" "$*"; }
ok()   { printf "\033[1;32m✓\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m!\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m✗\033[0m %s\n" "$*"; exit 1; }

echo "════════════════════════════════════════════════════════════"
echo " GreenVision · W10D3 demo orchestrator"
echo "════════════════════════════════════════════════════════════"

# ───────────────────────────────────────────────────────────────
# 1. Port availability
# ───────────────────────────────────────────────────────────────
say "Checking ports…"
for port in 8000 3000 5001; do
  if lsof -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then
    die "Port $port is already in use. Run scripts/demo_reset.sh to free it."
  fi
done
ok "Ports 8000, 3000, 5001 are free"

# ───────────────────────────────────────────────────────────────
# 2. Production model loads
# ───────────────────────────────────────────────────────────────
say "Verifying the Production model loads from MLflow Registry…"
.venv/bin/python -c "
import mlflow, mlflow.pytorch
mlflow.set_tracking_uri('file:./mlruns')
m = mlflow.pytorch.load_model('models:/GreenVision/Production')
print(f'  Production model: {type(m).__name__}')
" || die "Production model failed to load. Run scripts/promote.py to promote a version."
ok "models:/GreenVision/Production loads cleanly"

# ───────────────────────────────────────────────────────────────
# 3. Web deps + cached training data
# ───────────────────────────────────────────────────────────────
if [ ! -d "web/node_modules" ]; then
  say "web/node_modules missing — running npm install…"
  (cd web && npm install)
fi
ok "web/node_modules is present"

if [ ! -f "web/public/training_data.json" ]; then
  warn "web/public/training_data.json is missing — regenerating…"
  PYTHONPATH=src .venv/bin/python scripts/export_mlflow_for_dashboard.py
fi
ok "web/public/training_data.json is present"

# ───────────────────────────────────────────────────────────────
# 4. Launch
# ───────────────────────────────────────────────────────────────
echo ""
echo "Starting GreenVision demo. Press Ctrl+C to stop everything."
echo ""
echo "  • Dashboard : http://localhost:3000   (the demo surface)"
echo "  • FastAPI   : http://localhost:8000   (/docs for Swagger)"
echo "  • MLflow UI : http://localhost:5001   (Models tab → GreenVision)"
echo ""

# concurrently was installed as a devDep in web/. Run it from web/'s
# node_modules so we don't need a global install.
exec web/node_modules/.bin/concurrently \
  --names "API,UI,MLflow" \
  --prefix-colors "cyan.bold,magenta.bold,yellow.bold" \
  --kill-others-on-fail \
  --handle-input \
  "PYTHONPATH=src .venv/bin/uvicorn api.main:app --port 8000" \
  "bash -c 'cd web && npm run dev'" \
  ".venv/bin/mlflow ui --backend-store-uri file:./mlruns --port 5001"
