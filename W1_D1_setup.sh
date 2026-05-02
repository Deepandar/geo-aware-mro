#!/usr/bin/env bash
# ============================================================
# W1_D1_setup.sh  —  Week 1 Day 1 Terminal Action Sequence
# Run this script FIRST, then open the notebook.
# ============================================================
# Usage:
#   chmod +x W1_D1_setup.sh
#   bash W1_D1_setup.sh
# ============================================================

set -euo pipefail   # exit on any error

# ── CONFIG — update before running ──────────────────────────
PROJECT_DIR="$HOME/geo-aware-mro"
GITHUB_USER="your-github-username"          # ← update
GITHUB_EMAIL="your-email@example.com"       # ← update
# ────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  GEO-AWARE MRO  ·  W1 D1 Setup          ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Python version gate ───────────────────────────────────
echo "→ Checking Python version..."
PY_VER=$(python --version 2>&1 | cut -d' ' -f2)
PY_MAJ=$(echo $PY_VER | cut -d. -f1)
PY_MIN=$(echo $PY_VER | cut -d. -f2)

if [ "$PY_MAJ" -lt 3 ] || [ "$PY_MIN" -lt 11 ]; then
  echo "❌ Python $PY_VER detected. Need ≥3.11."
  echo "   Run: conda activate geo-mro"
  exit 1
fi
echo "✅ Python $PY_VER OK"

# ── 2. Create project directory ──────────────────────────────
echo ""
echo "→ Creating project directory at: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
echo "✅ cd $PROJECT_DIR"

# ── 3. Git init ──────────────────────────────────────────────
echo ""
echo "→ Initialising git repo..."
if [ ! -d ".git" ]; then
  git init
  git config user.name  "$GITHUB_USER"
  git config user.email "$GITHUB_EMAIL"
  echo "✅ git init complete"
else
  echo "⏩ .git already exists — skipping init"
fi

# ── 4. Create conda env (if not already active) ──────────────
echo ""
echo "→ Checking conda env 'geo-mro'..."
if conda info --envs | grep -q "geo-mro"; then
  echo "⏩ env 'geo-mro' already exists"
  echo "   Activate with: conda activate geo-mro"
else
  echo "   env not found — will create after environment.yml is written"
  echo "   (Run notebook Steps 3-4 first, then come back)"
fi

# ── 5. Summary ───────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Terminal setup complete.                ║"
echo "║  Next: open W1_D1_environment_setup.ipynb ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "After running the notebook, come back and run:"
echo ""
echo "  conda env create -f environment.yml"
echo "  conda activate geo-mro"
echo "  git add ."
echo "  git commit -m 'chore: W1D1 — project scaffold, conda env, CI pipeline'"
echo "  git checkout -b develop && git checkout main"
echo "  git remote add origin https://github.com/$GITHUB_USER/geo-aware-mro.git"
echo "  git push -u origin main"
echo "  git push -u origin develop"
