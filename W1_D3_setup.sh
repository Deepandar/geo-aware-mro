#!/usr/bin/env bash
# ============================================================
# W1_D3_setup.sh  —  Week 1 Day 3 Terminal Action Sequence
# Run this script FIRST, then open the notebook.
# ============================================================
# What this does:
#   1. Verifies DVC ≥ 3.x is installed in the active env
#   2. Confirms you are in the correct project directory
#   3. Confirms you're on the develop branch
#   4. Creates the local DVC remote directory
#   5. Prints the post-notebook commit commands
# ============================================================
# Usage:
#   conda activate geo-mro
#   cd ~/geo-aware-mro
#   chmod +x W1_D3_setup.sh
#   bash W1_D3_setup.sh
# ============================================================

set -euo pipefail

# ── CONFIG — update before running ──────────────────────────
PROJECT_DIR="$HOME/geo-aware-mro"
LOCAL_DVC_REMOTE="$HOME/geo-mro-dvc-remote"
# ────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  GEO-AWARE MRO  ·  W1 D3 DVC Setup      ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Python + conda env gate ───────────────────────────────
echo "→ Checking Python environment..."
PY_VER=$(python --version 2>&1 | cut -d' ' -f2)
PY_MAJ=$(echo $PY_VER | cut -d. -f1)
PY_MIN=$(echo $PY_VER | cut -d. -f2)

if [ "$PY_MAJ" -lt 3 ] || [ "$PY_MIN" -lt 11 ]; then
  echo "❌ Python $PY_VER detected. Need ≥3.11."
  echo "   Run: conda activate geo-mro"
  exit 1
fi
echo "✅ Python $PY_VER OK"

CONDA_ENV=${CONDA_DEFAULT_ENV:-"(none)"}
echo "   Conda env: $CONDA_ENV"
if [ "$CONDA_ENV" != "geo-mro" ]; then
  echo "⚠️  Not in 'geo-mro' env. Continuing, but activate it first:"
  echo "   conda activate geo-mro"
fi

# ── 2. DVC version check ─────────────────────────────────────
echo ""
echo "→ Checking DVC..."
if ! command -v dvc &> /dev/null; then
  echo "❌ dvc not found on PATH."
  echo "   Fix: pip install dvc>=3.30"
  exit 1
fi

DVC_VER=$(dvc --version)
DVC_MAJ=$(echo $DVC_VER | cut -d. -f1)
if [ "$DVC_MAJ" -lt 3 ]; then
  echo "❌ DVC $DVC_VER detected. Need ≥3.x"
  echo "   Fix: pip install --upgrade dvc"
  exit 1
fi
echo "✅ DVC $DVC_VER OK"

# ── 3. Project directory check ───────────────────────────────
echo ""
echo "→ Checking project directory..."
if [ ! -d "$PROJECT_DIR" ]; then
  echo "❌ Project directory not found: $PROJECT_DIR"
  echo "   Run W1_D1_setup.sh first."
  exit 1
fi
cd "$PROJECT_DIR"
echo "✅ cd $PROJECT_DIR"

# ── 4. Git repo check ────────────────────────────────────────
echo ""
echo "→ Checking git repo..."
if [ ! -d ".git" ]; then
  echo "❌ .git not found in $PROJECT_DIR"
  echo "   Run W1_D1_setup.sh first."
  exit 1
fi
echo "✅ .git exists"

# ── 5. Branch check — must be on develop ────────────────────
echo ""
echo "→ Checking git branch..."
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
if [ "$CURRENT_BRANCH" != "develop" ]; then
  echo "⚠️  On branch '$CURRENT_BRANCH', not 'develop'."
  echo "   Switching to develop..."
  git checkout develop 2>/dev/null || git checkout -b develop
fi
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "✅ On branch: $CURRENT_BRANCH"

# ── 6. Day 2 gate: MLflow DB must exist ─────────────────────
echo ""
echo "→ Checking Day 2 completion (MLflow)..."
MLFLOW_DB=$(find . -name "mlflow.db" 2>/dev/null | head -1)
if [ -z "$MLFLOW_DB" ]; then
  echo "⚠️  mlflow.db not found — Day 2 may not be complete."
  echo "   Run W1_D2_mlflow_setup.ipynb first."
  echo "   Continuing anyway (notebook will catch this)."
else
  echo "✅ MLflow DB found: $MLFLOW_DB"
fi

# ── 7. Create local DVC remote directory ────────────────────
echo ""
echo "→ Creating local DVC remote at: $LOCAL_DVC_REMOTE"
mkdir -p "$LOCAL_DVC_REMOTE"
echo "✅ $LOCAL_DVC_REMOTE ready"

# ── 8. DVC init check ────────────────────────────────────────
echo ""
echo "→ Checking DVC init status..."
if [ ! -d ".dvc" ]; then
  echo "   .dvc/ not found — will run dvc init inside notebook"
else
  echo "⏩ .dvc/ already exists"
  DVC_CONFIG=".dvc/config"
  if [ -f "$DVC_CONFIG" ]; then
    echo "   .dvc/config:"
    cat "$DVC_CONFIG"
  fi
fi

# ── 9. Print post-notebook commit sequence ───────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Pre-notebook checks complete.                               ║"
echo "║  Next: open W1_D3_dvc_setup.ipynb and run all cells.        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "After completing the notebook, run these commands:"
echo ""
echo "  cd ~/geo-aware-mro"
echo "  git checkout develop"
echo ""
echo "  # Stage DVC configuration"
echo "  git add .dvc/.gitignore .dvc/config"
echo "  git add .dvcignore"
echo ""
echo "  # Stage pipeline files"
echo "  git add dvc.yaml params.yaml"
echo ""
echo "  # Stage data layer metadata (NOT actual data)"
echo "  git add data/raw/README.md data/raw/schema.json data/raw/.gitkeep"
echo "  git add data/processed/README.md data/processed/schema.json data/processed/.gitkeep"
echo "  git add data/external/README.md data/external/schema.json data/external/.gitkeep"
echo "  git add data/interim/README.md data/interim/schema.json data/interim/.gitkeep"
echo ""
echo "  # Stage new source file"
echo "  git add src/utils/data_io.py"
echo ""
echo "  # Commit & push"
echo "  git commit -m 'chore: W1D3 — DVC init, pipeline DAG, params, data_io'"
echo "  git push origin develop"
echo ""
echo "  # Verify DVC pipeline DAG"
echo "  dvc dag"
echo "  dvc status"
echo ""
echo "Day 4 (Thu): Dockerfile + docker-compose + GitHub Actions CI"
echo ""
