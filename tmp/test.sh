#!/bin/bash
# Full Pallas regression sweep.
#
# Runs the entire test/ suite under HELION_BACKEND=pallas so we catch
# anything the recent jagged + max-clamp + emit_pipeline-gate changes
# might have broken.  Does NOT stop at the first failure -- we want
# the full picture.
#
# Run:    bash tmp/test.sh   (from the helion repo root)
# Output: tmp/log.txt
#
# Parallelism via pytest-xdist if available; falls back to serial.

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

  # NOTE: do NOT use pytest-xdist on TPU.  libtpu uses a multi-process
  # lockfile and only one process can hold the device; xdist workers
  # 2..N fail with "Internal error when accessing libtpu multi-process
  # lockfile" -- spurious, not a code issue.

  echo "================================================"
  echo "Full test/ suite under HELION_BACKEND=pallas"
  echo "================================================"
  HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1 \
  PYTHONPATH=. \
  HELION_SKIP_CACHE=1 \
  HELION_BACKEND=pallas \
  HELION_AUTOTUNE_EFFORT=none \
  python3 -m pytest test/ \
    -ra --tb=short \
    --durations=20

  echo
  echo "================================================"
  echo "Done.  Exit code: $?"
  echo "================================================"
} 2>&1 | tee tmp/log.txt
