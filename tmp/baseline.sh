#!/bin/bash
# Baseline the same sweep against origin/main.  Anything failing on main
# is pre-existing and unrelated to any jagged work -- including our 3
# trial-only commits.
#
# Run:    bash tmp/baseline.sh   (from the helion repo root)
# Output: tmp/log_baseline.txt
#
# Restores the original branch/HEAD when done.

set -u

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
CURRENT_SHA=$(git rev-parse HEAD)

mkdir -p tmp

cleanup() {
  echo
  echo "Restoring $CURRENT_BRANCH @ $CURRENT_SHA"
  git checkout "$CURRENT_BRANCH" 2>&1 | tail -3
}
trap cleanup EXIT

git fetch origin main 2>&1 | tail -3
git checkout origin/main 2>&1 | tail -3

{
  echo "================================================"
  echo "Baseline: origin/main"
  echo "Git HEAD: $(git rev-parse --short HEAD)"
  echo "================================================"
  date
  echo

  HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1 \
  PYTHONPATH=. \
  HELION_SKIP_CACHE=1 \
  HELION_BACKEND=pallas \
  HELION_AUTOTUNE_EFFORT=none \
  python3 -m pytest test/ \
    -ra --tb=line \
    --durations=20

  echo
  echo "================================================"
  echo "Done.  Exit code: $?"
  echo "================================================"
} 2>&1 | tee tmp/log_baseline.txt
