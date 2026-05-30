#!/usr/bin/env bash
# Backup demo path: Next.js dashboard only, no FastAPI backend.
#
# Use this when the FastAPI backend is unavailable mid-presentation. The
# Analytics page still works (it reads the static training_data.json), and
# the About page is unaffected. The Diagnose page will surface the
# "Backend not reachable" alert — which is itself a demonstrable handling
# behavior (covers the "stop FastAPI manually" integration scenario from
# the rubric).
#
# Usage:
#   ./scripts/demo_static.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/web"

if [ ! -d "node_modules" ]; then
  echo "→ Installing web deps…"
  npm install
fi

if [ ! -f "public/training_data.json" ]; then
  echo "✗ public/training_data.json is missing. Generate it first:"
  echo "    PYTHONPATH=src .venv/bin/python scripts/export_mlflow_for_dashboard.py"
  exit 1
fi

echo "════════════════════════════════════════════════════════════"
echo " GreenVision · static fallback (Next.js only, no API)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  • Dashboard : http://localhost:3000"
echo "  • Analytics : http://localhost:3000/analytics  (still works)"
echo "  • Diagnose  : http://localhost:3000            (will show API alert)"
echo ""
echo "Press Ctrl+C to stop."
echo ""

exec npm run dev
