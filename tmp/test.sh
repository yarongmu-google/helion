#!/bin/bash
# Verify the 6 trial-only regressions are gone AND the 4 jagged tests
# still pass after gating BlockSizeSpec clamp on an explicit is_hard_pin
# flag.
#
# Run:    bash tmp/test.sh   (from helion repo root)
# Output: tmp/log.txt

set -u

mkdir -p tmp

{
  echo "================================================"
  echo "Targeted sweep: 4 jagged + 6 trial-only regressions"
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
    -k "test_jagged_sum or test_jagged_mean or test_jagged_softmax or test_jagged_layer_norm or test_broadcast_mask_size1_last_dim or test_broadcast_mask_size1_multiple_dims or test_emit_pipeline_loop_order or test_matmul_broadcast_bias or test_template_via_closure0 or test_template_via_closure1 or test_layernorm_bwd or test_squeeze_and_excitation_net_bwd_db or test_jagged_hstu_attn_2" \
    -ra --tb=short -v

  echo
  echo "================================================"
  echo "Done.  Exit code: $?"
  echo "================================================"
} 2>&1 | tee tmp/log.txt
