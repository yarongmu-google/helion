#!/bin/bash
# V1 jagged-kernel TPU verification.
#
# Runs the 4 jagged tests in test/test_examples.py with the Pallas backend.
# All 4 had @xfailIfPallas; they were removed when the V1 jagged kernels
# (sum, mean, softmax, layer_norm) were lit up on TPU.
#
# Run:    bash tmp/test.sh   (from the helion repo root)
# Output: tmp/log.txt

set -u

mkdir -p tmp

{
  echo "================================================"
  echo "Environment"
  echo "================================================"
  date
  uname -a
  python3 --version
  echo
  python3 -c "import jax; print('jax:', jax.__version__); print('devices:', jax.devices())" || true
  echo "Git HEAD: $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
  echo

  echo "================================================"
  echo "test_jagged_sum / mean / softmax / layer_norm"
  echo "================================================"
  HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1 \
  HELION_AUTOTUNE_LOG_LEVEL=DEBUG \
  PYTHONPATH=. \
  HELION_SKIP_CACHE=1 \
  HELION_BACKEND=pallas \
  HELION_AUTOTUNE_EFFORT=none \
  python3 -m pytest test/test_examples.py \
    -k "test_jagged_sum or test_jagged_mean or test_jagged_softmax or test_jagged_layer_norm" \
    -x -vv -s -ra

  echo
  echo "================================================"
  echo "Done.  Exit code: $?"
  echo "================================================"
} 2>&1 | tee tmp/log.txt
