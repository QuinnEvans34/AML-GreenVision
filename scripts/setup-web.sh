#!/usr/bin/env bash
# Scaffolds the GreenVision dashboard at web/.
#
# Phase 3 of the W10 implementation plan:
#   1. Verifies Node 18+ is installed.
#   2. Runs create-next-app with all flags pre-filled (no prompts).
#   3. Initializes shadcn/ui with default New York / slate base.
#   4. Adds the shadcn components we'll use.
#   5. Installs Three.js + React Three Fiber + drei + recharts + next-themes + lucide.
#   6. Installs `concurrently` for scripts/demo.sh.
#
# Usage:
#   ./scripts/setup-web.sh
#
# Takes ~3-7 minutes depending on network speed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "────────────────────────────────────────────────"
echo " GreenVision web scaffold (Phase 3)"
echo "────────────────────────────────────────────────"

# ────────────────────────────────────────────────────────────────
# 1. Node version check
# ────────────────────────────────────────────────────────────────
if ! command -v node >/dev/null 2>&1; then
  echo "✗ Node.js is not installed. Install Node 20+ from https://nodejs.org/"
  exit 1
fi

NODE_MAJOR=$(node -v | sed 's/v//' | cut -d. -f1)
if [ "$NODE_MAJOR" -lt 18 ]; then
  echo "✗ Node 18+ required (have v$NODE_MAJOR). Upgrade Node."
  exit 1
fi
echo "✓ Node $(node -v)"
echo "✓ npm  $(npm -v)"
echo ""

# ────────────────────────────────────────────────────────────────
# 2. Refuse to clobber an existing scaffold
# ────────────────────────────────────────────────────────────────
if [ -d "web" ]; then
  echo "✗ web/ already exists."
  echo "  If you intend to re-scaffold, remove it first: rm -rf web"
  exit 1
fi

# ────────────────────────────────────────────────────────────────
# 3. Scaffold Next.js 14 App Router with all flags pre-filled
# ────────────────────────────────────────────────────────────────
echo "→ [1/4] Scaffolding Next.js…"
npx --yes create-next-app@latest web \
  --typescript \
  --tailwind \
  --app \
  --no-src-dir \
  --import-alias "@/*" \
  --use-npm \
  --eslint \
  --no-turbo
echo "✓ Next.js scaffolded at web/"
echo ""

cd web

# ────────────────────────────────────────────────────────────────
# 4. shadcn/ui init (defaults: New York style, slate base, CSS vars)
# ────────────────────────────────────────────────────────────────
echo "→ [2/4] Initializing shadcn/ui…"
npx --yes shadcn@latest init -d -y
echo "✓ shadcn/ui initialized"
echo ""

# ────────────────────────────────────────────────────────────────
# 5. Add the shadcn components we need across the dashboard
# ────────────────────────────────────────────────────────────────
echo "→ [3/4] Adding shadcn components…"
npx --yes shadcn@latest add -y \
  button \
  card \
  badge \
  alert \
  separator \
  tabs \
  progress \
  skeleton \
  dialog \
  input \
  label \
  tooltip \
  dropdown-menu \
  sonner
echo "✓ shadcn components added"
echo ""

# ────────────────────────────────────────────────────────────────
# 6. Install third-party libraries we use directly
# ────────────────────────────────────────────────────────────────
echo "→ [4/4] Installing third-party libraries…"
npm install \
  next-themes \
  three \
  @react-three/fiber \
  @react-three/drei \
  recharts \
  lucide-react

npm install --save-dev \
  @types/three \
  concurrently
echo "✓ Dependencies installed"
echo ""

# ────────────────────────────────────────────────────────────────
# 7. Done
# ────────────────────────────────────────────────────────────────
echo "────────────────────────────────────────────────"
echo " ✓ Web scaffold complete."
echo "────────────────────────────────────────────────"
echo ""
echo "Quick sanity check:"
echo "  cd web && npm run dev"
echo "  → http://localhost:3000"
echo ""
echo "Then come back here and I'll write the layout, routes, and components."
