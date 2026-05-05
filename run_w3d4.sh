#!/usr/bin/env bash
# =============================================================================
# GEO-AWARE MRO DECISION INTELLIGENCE SYSTEM
# Execution Wrapper: Week 3 Day 4 (Composite Criticality Index)
# =============================================================================

set -e

echo "================================================================="
echo "  INITIATING WEEK 3 DAY 4: COMPOSITE CRITICALITY INDEX (C_i)"
echo "================================================================="

echo "[1/2] Verifying directory structure..."
mkdir -p data/processed
mkdir -p mlflow

echo "[2/2] Executing Python engine..."
python week3_day4_ci_scorer.py

echo "================================================================="
echo "  BASH: EXECUTION SUCCESSFUL"
echo "  Artifacts saved to data/processed/"
echo "================================================================="
