#!/bin/bash
# V1 jagged kernel smoke test on real TPU.
#
# Run: bash tmp/test.sh   (from the helion repo root)
# Output captured in: tmp/log.txt
#
# Runs the canonical V1 jagged_sum kernel from examples/ — the existing
# ``main()`` already wires up real-scale test data
# (B=8, M=128, max_seqlen=64) and runs ``helion._testing.run_example``
# (correctness check + benchmark).
#
# This is the smallest viable end-to-end TPU validation of the work
# committed under #71/#72/#73/#74/#67/#77.  If this passes, we expand to
# jagged_mean / jagged_softmax / jagged_layer_norm.
#
# Notable env:
#   HELION_BACKEND=pallas         : force Pallas codegen path
#   (no HELION_PALLAS_INTERPRET=1!) we want real TPU, not the interpret
#                                   shim — interpret silently clamps
#                                   lax.dynamic_slice so per-item starts
#                                   and DMA-OOB-tolerance bugs hide
#   HELION_AUTOTUNE_EFFORT=none   : skip autotuning; use default config
#   HELION_PRINT_OUTPUT_CODE=1    : print the generated Pallas kernel so
#                                   the log contains the emit even if
#                                   compile succeeds (helpful when
#                                   numbers come out wrong)

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
  echo "--- JAX / Pallas ---"
  python3 -c "import jax; print('jax:', jax.__version__); print('devices:', jax.devices())" || true
  python3 -c "from jax.experimental.pallas import tpu as pltpu; print('pltpu: importable')" || true
  echo
  echo "--- Helion ---"
  python3 -c "import helion; print('helion:', getattr(helion, '__version__', '(unversioned)'))" || true
  echo "Git HEAD: $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
  echo

  # Reusable env block.  Same flags for both examples.
  COMMON_ENV=(
    HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1
    HELION_AUTOTUNE_LOG_LEVEL=DEBUG
    TORCH_COMPILE_DEBUG=1
    PYTHONPATH=.
    HELION_SKIP_CACHE=1
    HELION_BACKEND=pallas
    HELION_AUTOTUNE_EFFORT=none
    HELION_PRINT_OUTPUT_CODE=1
  )

  echo "================================================"
  echo "jagged_sum: examples/jagged_sum.py main() (regression guard)"
  echo "================================================"
  env "${COMMON_ENV[@]}" python3 examples/jagged_sum.py
  jagged_sum_rc=$?
  echo "jagged_sum exit code: $jagged_sum_rc"

  echo
  echo "================================================"
  echo "jagged_mean: examples/jagged_mean.py main()"
  echo "================================================"
  env "${COMMON_ENV[@]}" python3 examples/jagged_mean.py
  jagged_mean_rc=$?
  echo "jagged_mean exit code: $jagged_mean_rc"

  echo
  echo "================================================"
  echo "Done.  jagged_sum=$jagged_sum_rc  jagged_mean=$jagged_mean_rc"
  echo "================================================"
} 2>&1 | tee tmp/log.txt
