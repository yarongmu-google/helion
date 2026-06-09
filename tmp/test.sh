#!/bin/bash
# M>128 sublane-alignment probe.
#
# Run:    bash tmp/test.sh   (from the helion repo root)
# Output: tmp/log.txt
#
# Goal: show that XLA itself has no problem dynamic-slicing a (N, 256)
# fp32 tensor at a runtime offset on TPU.  The constraint we see in
# helion comes specifically from Mosaic's ``tpu.memref_slice`` verifier
# rejecting unaligned offsets on an HBM memref whose XLA-picked tile
# shape is (8, 128) (sublane tile 8 because lane > 128 fp32).
#
# Test A (XLA / JAX native, expected PASS):
#   - Allocate x = jnp.ones((2048, 256), fp32) on TPU
#   - lax.dynamic_slice(x, (runtime_start, 0), (16, 128))
#   - The runtime_start is data-dependent (read from another array).
#   - Pure XLA -- no Pallas / no tpu.memref_slice.
#
# Test B (Pallas DMA, expected FAIL with the verifier error):
#   - Same x.
#   - Use a pallas_call whose body issues
#     ``pltpu.make_async_copy(x_hbm.at[pl.ds(start, 16), pl.ds(0, 128)], buf, sem)``
#     with start coming from a scalar SMEM prefetch.
#   - Reproduces the exact MLIR seen in the helion failure:
#       tpu.memref_slice(<2048x256xf32, tiled<(8,128)>, hbm>, i32, i32)
#         -> <16x128xf32, tiled<(8,128)>, hbm>
#
# Helion smoke test is commented out below so this probe runs in
# isolation -- re-enable when M>128 sublane alignment is solved.

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
  echo "Git HEAD: $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
  echo

  echo "================================================"
  echo "Test A: lax.dynamic_slice on (N, 256), runtime offset (expect PASS)"
  echo "================================================"
  python3 - <<'PY'
import jax, jax.numpy as jnp
from jax import lax

@jax.jit
def slice_at(x, start):
    return lax.dynamic_slice(x, (start, 0), (16, 128))

x = jnp.arange(2048 * 256, dtype=jnp.float32).reshape(2048, 256)
start = jnp.array(7, dtype=jnp.int32)  # data-dependent, not 8-aligned
out = slice_at(x, start)
print("OK -- out.shape:", out.shape, "out[0,0]:", float(out[0, 0]),
      "(== 7*256+0 ==", 7 * 256, ")")
PY

  echo
  echo "================================================"
  echo "Test B: Pallas DMA, same slice (expect FAIL: Mosaic alignment)"
  echo "================================================"
  python3 - <<'PY'
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def body(start_ref, x_ref, out_ref, x_buf, x_sem):
    start = start_ref[0]
    copy = pltpu.make_async_copy(
        x_ref.at[pl.ds(start, 16), pl.ds(0, 128)],
        x_buf,
        x_sem,
    )
    copy.start()
    copy.wait()
    out_ref[...] = x_buf[...]

call = pl.pallas_call(
    body,
    out_shape=jax.ShapeDtypeStruct((16, 128), jnp.float32),
    in_specs=[
        pl.BlockSpec(memory_space=pltpu.MemorySpace.SMEM),
        pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
    ],
    out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
    scratch_shapes=[
        pltpu.VMEM((16, 128), jnp.float32),
        pltpu.SemaphoreType.DMA,
    ],
)

x = jnp.arange(2048 * 256, dtype=jnp.float32).reshape(2048, 256)
start = jnp.array([7], dtype=jnp.int32)
try:
    out = call(start, x)
    out.block_until_ready()
    print("UNEXPECTED PASS -- out.shape:", out.shape)
except Exception as e:
    print("EXPECTED FAIL -- first line of error:")
    print(str(e).splitlines()[0])
PY

  echo
  echo "================================================"
  echo "Test C: same DMA wrapped in pl.multiple_of(start, 8)"
  echo "         (compiles, but the assertion is FALSE for start=7"
  echo "         so the result should be WRONG, not just unaligned)"
  echo "================================================"
  python3 - <<'PY'
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def body(start_ref, x_ref, out_ref, x_buf, x_sem):
    start = pl.multiple_of(start_ref[0], 8)  # LIES when start_ref[0] is not 8-aligned
    copy = pltpu.make_async_copy(
        x_ref.at[pl.ds(start, 16), pl.ds(0, 128)],
        x_buf,
        x_sem,
    )
    copy.start()
    copy.wait()
    out_ref[...] = x_buf[...]

call = pl.pallas_call(
    body,
    out_shape=jax.ShapeDtypeStruct((16, 128), jnp.float32),
    in_specs=[
        pl.BlockSpec(memory_space=pltpu.MemorySpace.SMEM),
        pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
    ],
    out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
    scratch_shapes=[
        pltpu.VMEM((16, 128), jnp.float32),
        pltpu.SemaphoreType.DMA,
    ],
)

x = jnp.arange(2048 * 256, dtype=jnp.float32).reshape(2048, 256)
for s in (0, 7, 8, 16):
    start = jnp.array([s], dtype=jnp.int32)
    try:
        out = call(start, x)
        out.block_until_ready()
        # Expected = first lane stripe (cols 0..127) of rows [s, s+16)
        # If the DMA actually honored start=s, out[0,0] == s*256
        # If it rounded down to a multiple of 8, out[0,0] == (s & ~7)*256
        actual = float(out[0, 0])
        truth = s * 256
        rounded = (s & ~7) * 256
        verdict = "CORRECT" if actual == truth else (
            "ROUNDED_DOWN" if actual == rounded else "OTHER"
        )
        print(f"  start={s:>2}  truth={truth:>6}  out[0,0]={actual:>7.0f}  -> {verdict}")
    except Exception as e:
        print(f"  start={s:>2}  ERROR: {str(e).splitlines()[0]}")
PY

  echo
  echo "================================================"
  echo "Test D: reshape HBM ref before slicing (expect PASS at any start)"
  echo "  Reshape (2048, 256) -> (2048*2, 128) so lane=128 exactly."
  echo "  After reshape the (8,128) tile annotation should NOT apply;"
  echo "  HBM is 1D + 32B-aligned at the hardware level."
  echo "================================================"
  python3 - <<'PY'
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def body(start_ref, x_ref, out_ref, x_buf, x_sem):
    # Each original row (256 cols) becomes 2 rows of 128 in the reshape.
    # Original row r maps to reshaped rows 2r and 2r+1.
    flat_start = start_ref[0] * 2  # reshape-row offset
    x_flat = x_ref.reshape(2048 * 2, 128)   # in-place memref_reshape
    copy = pltpu.make_async_copy(
        x_flat.at[pl.ds(flat_start, 16), pl.ds(0, 128)],  # 16 reshape-rows = 8 original rows
        x_buf,
        x_sem,
    )
    copy.start()
    copy.wait()
    out_ref[...] = x_buf[...]

call = pl.pallas_call(
    body,
    out_shape=jax.ShapeDtypeStruct((16, 128), jnp.float32),
    in_specs=[
        pl.BlockSpec(memory_space=pltpu.MemorySpace.SMEM),
        pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
    ],
    out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
    scratch_shapes=[
        pltpu.VMEM((16, 128), jnp.float32),
        pltpu.SemaphoreType.DMA,
    ],
)

x = jnp.arange(2048 * 256, dtype=jnp.float32).reshape(2048, 256)
for s in (0, 7, 8, 16):
    start = jnp.array([s], dtype=jnp.int32)
    try:
        out = call(start, x)
        out.block_until_ready()
        # After reshape, reshape-row 2s = original (s, 0:128).
        # out[0, 0] should equal x[s, 0] = s*256.
        actual = float(out[0, 0])
        truth = s * 256
        verdict = "CORRECT" if actual == truth else f"GOT {actual:.0f} expected {truth}"
        print(f"  start={s:>2}  out[0,0]={actual:>7.0f}  -> {verdict}")
    except Exception as e:
        print(f"  start={s:>2}  ERROR: {str(e).splitlines()[0]}")
PY

  echo
  echo "================================================"
  echo "Done.  Exit code: $?"
  echo "================================================"

  # -- HELION SMOKE (disabled while M>128 is open) --
  # echo "================================================"
  # echo "jagged_layer_norm: examples/jagged_layer_norm.py main()"
  # echo "================================================"
  # HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1  \
  # HELION_AUTOTUNE_LOG_LEVEL=DEBUG \
  # TORCH_COMPILE_DEBUG=1 \
  # PYTHONPATH=. \
  # HELION_SKIP_CACHE=1 \
  # HELION_BACKEND=pallas \
  # HELION_AUTOTUNE_EFFORT=none \
  # HELION_PRINT_OUTPUT_CODE=1 \
  # python3 examples/jagged_layer_norm.py
} 2>&1 | tee tmp/log.txt
