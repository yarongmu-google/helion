#!/bin/bash
# Full Pallas regression sweep on trial.
#
# Run:    bash tmp/test.sh   (from helion repo root)
# Output: tmp/log.txt
#
# NOTE: no pytest-xdist on TPU (libtpu lockfile is single-process).

set -u

mkdir -p tmp

{
  echo "================================================"
  echo "Environment"
  echo "================================================"
  date
  uname -a
  python3 --version
  python3 -c "import jax; print('jax:', jax.__version__)" || true
  echo "Git HEAD: $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
  echo

  echo "================================================"
  echo "Full test/ suite under HELION_BACKEND=pallas"
  echo "================================================"
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
} 2>&1 | tee tmp/log.txt
