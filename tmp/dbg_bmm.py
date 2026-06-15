"""Minimal repro for jagged_dense_bmm debugging.

Usage on TPU box:
    cd ~/helion
    git checkout jagged_dense_bmm
    HELION_PRINT_OUTPUT_CODE=1 HELION_AUTOTUNE_EFFORT=none python /tmp/dbg_bmm.py 2>&1 | tee /tmp/jdb_dbg.log

Then push /tmp/jdb_dbg.log via trial.
"""

import sys
import torch

sys.path.insert(0, ".")

import helion  # noqa: E402
from helion._testing import DEVICE  # noqa: E402

# Use tiny power-of-2 shapes to take masking edges out of the picture.
B = 2
MAX_SEQ_LEN = 4
D = 8
K = 8


def main() -> None:
    from examples.jagged_dense_bmm import jagged_dense_bmm, jagged_dense_bmm_reference

    torch.manual_seed(0)
    seq_offsets, jagged, dense, bias = _make_input()
    print("[dbg] seq_offsets:", seq_offsets.tolist())
    print("[dbg] jagged.shape:", jagged.shape, "dense.shape:", dense.shape)

    ref = jagged_dense_bmm_reference(seq_offsets, jagged, dense, bias)
    print("[dbg] reference output (first item):")
    print(ref[: seq_offsets[1].item()])

    got = jagged_dense_bmm(seq_offsets, jagged, dense, bias)
    print("[dbg] helion output (first item):")
    print(got[: seq_offsets[1].item()])

    diff = (ref - got).abs()
    print("[dbg] abs-diff max:", diff.max().item(), "mean:", diff.mean().item())
    print("[dbg] per-item abs-diff max:")
    for i in range(B):
        s, e = seq_offsets[i].item(), seq_offsets[i + 1].item()
        if s == e:
            print(f"  item {i}: empty")
            continue
        print(f"  item {i} (rows [{s}:{e}]): max={diff[s:e].max().item():.4g}")


def _make_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Use Helion's LONG_INT_TYPE indirection so int64 on GPU / int32 on TPU.
    from helion._testing import LONG_INT_TYPE

    lengths = torch.randint(1, MAX_SEQ_LEN + 1, size=(B,), device=DEVICE)
    seq_offsets = torch.zeros((B + 1,), dtype=LONG_INT_TYPE, device=DEVICE)
    seq_offsets[1:] = torch.cumsum(lengths, dim=0).to(LONG_INT_TYPE)
    jagged_size = int(seq_offsets[-1].item())
    jagged = torch.empty((jagged_size, D), dtype=torch.float32, device=DEVICE).uniform_(-1.0, 1.0)
    dense = torch.empty((B, D, K), dtype=torch.float32, device=DEVICE).uniform_(-1.0, 1.0)
    bias = torch.empty((B, K), dtype=torch.float32, device=DEVICE).uniform_(-1.0, 1.0)
    return seq_offsets, jagged, dense, bias


if __name__ == "__main__":
    main()
