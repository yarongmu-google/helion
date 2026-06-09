#!/bin/bash
# Targeted: run only the trial-only-regressing tests with stderr captured
# so the BlockSizeSpec._normalize debug prints are visible in tmp/log.txt.
#
# Run:    bash tmp/test.sh   (from helion repo root)
# Output: tmp/log.txt

set -u

mkdir -p tmp

{
  echo "================================================"
  echo "Targeted Pallas regressions (with _normalize debug prints)"
  echo "Git HEAD: $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
  echo "================================================"
  date
  echo

  HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1 \
  PYTHONPATH=. \
  HELION_SKIP_CACHE=1 \
  HELION_BACKEND=pallas \
  HELION_AUTOTUNE_EFFORT=none \
  python3 -m pytest test/test_pallas.py test/test_examples.py \
    -k "test_broadcast_mask_size1_last_dim or test_emit_pipeline_loop_order or test_matmul_broadcast_bias or test_template_via_closure0" \
    -ra --tb=short -s -v

  echo
  echo "================================================"
  echo "Done.  Exit code: $?"
  echo "================================================"
} 2>&1 | tee tmp/log.txt
