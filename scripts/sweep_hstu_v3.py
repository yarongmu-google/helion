"""Pinned-config sweep for jagged HSTU v3 — no autotune.

Sweeps two cohorts around the current autotune winner
(``[1, 256, 128]`` pb=False, ~2.55 ms on TPU):

  (3) ``pallas_pre_broadcast=True`` variants around the winner
  (4) block-size neighbours (smaller block_q / block_kv)

Run on TPU::

    HELION_BACKEND=pallas PYTHONPATH=. python3 scripts/sweep_hstu_v3.py

Prints per-config timings and a ranked summary.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import helion
from helion._testing import DEVICE
from helion.runtime.settings import _get_backend

if _get_backend() == "pallas":
    from helion.autotuner.benchmarking import do_bench_generic as do_bench
else:
    from helion.autotuner.benchmarking import do_bench
import helion.language as hl


def _build_kernel(config: helion.Config) -> object:
    @helion.kernel(config=config, static_shapes=True)
    def jagged_hstu_attention(
        max_seq_len: int,
        alpha: float,
        attn_scale: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
    ) -> torch.Tensor:
        H = hl.specialize(q.size(1))
        D = hl.specialize(q.size(2))
        num_sequences = seq_offsets.size(0) - 1
        out = torch.empty_like(v)

        for tile_b in hl.tile(num_sequences):
            start = seq_offsets[tile_b.begin]
            seq_lens = seq_offsets[tile_b.index + 1] - seq_offsets[tile_b]

            for tile_q in hl.jagged_tile(seq_lens):
                q_idx = start + tile_q.index
                q_blk = q[q_idx, :, :].transpose(0, 1)
                acc = hl.zeros([H, tile_q, D], dtype=torch.float32)

                for tile_kv in hl.jagged_tile(seq_lens):
                    kv_idx = start + tile_kv.index
                    k_blk = k[kv_idx, :, :].transpose(0, 1)
                    v_blk = v[kv_idx, :, :].transpose(0, 1)

                    scores = torch.bmm(q_blk, k_blk.transpose(-2, -1)) * alpha
                    scores = F.silu(scores) * attn_scale

                    causal_mask = (
                        tile_q.index.unsqueeze(1) >= tile_kv.index.unsqueeze(0)
                    )
                    scores = torch.where(causal_mask[None, :, :], scores, 0.0)

                    acc = acc + torch.bmm(scores.to(v.dtype), v_blk)

                out[q_idx, :, :] = acc.transpose(0, 1).to(out.dtype)

        return out

    return jagged_hstu_attention


def _make_inputs() -> (
    tuple[int, float, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    torch.manual_seed(0)
    num_sequences = 16
    max_seq_len = 512
    heads = 32
    head_dim = 128
    alpha = 1.0 / head_dim**2
    attn_scale = 1.0 / max_seq_len
    dtype = torch.float32
    device = torch.device(DEVICE)

    lengths = torch.randint(
        max_seq_len // 2,
        max_seq_len + 1,
        (num_sequences,),
        dtype=torch.int32,
    )
    seq_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(lengths, dim=0).to(torch.int32),
        ]
    ).to(device)
    total_len = int(seq_offsets[-1].item())

    q = torch.randn(total_len, heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_len, heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_len, heads, head_dim, dtype=dtype, device=device)
    return max_seq_len, alpha, attn_scale, q, k, v, seq_offsets


# (block_sizes, pallas_pre_broadcast)
CONFIGS: list[tuple[list[int], bool]] = [
    # Current autotune winner — baseline.
    ([1, 256, 128], False),
    # (3) pre-broadcast sweep around the winner.
    ([1, 256, 128], True),
    ([1, 128, 128], True),
    ([1, 256, 256], True),
    ([1, 128, 256], True),
    ([1, 64, 128], True),
    # (4) block-size sweep, pb=False.
    ([1, 128, 128], False),
    ([1, 256, 64], False),
    ([1, 512, 64], False),
    ([1, 64, 128], False),
    ([1, 64, 64], False),
    ([1, 128, 64], False),
    ([1, 256, 256], False),
    ([1, 512, 128], False),
    # block-size sweep, pb=True.
    ([1, 256, 64], True),
    ([1, 512, 64], True),
    ([1, 64, 64], True),
    ([1, 128, 64], True),
    ([1, 512, 128], True),
]


def main() -> None:
    args = _make_inputs()
    results: list[tuple[list[int], bool, float | None, str | None]] = []

    print(f"{'block_sizes':22s}  pb     time(ms)")
    print("-" * 50)
    for block_sizes, pb in CONFIGS:
        cfg = helion.Config(
            block_sizes=block_sizes,
            pallas_loop_type="fori_loop",
            pallas_pre_broadcast=pb,
        )
        try:
            kernel = _build_kernel(cfg)
            t_ms = do_bench(lambda: kernel(*args))
            results.append((block_sizes, pb, t_ms, None))
            print(f"{str(block_sizes):22s}  {str(pb):5s}  {t_ms:.3f}")
        except Exception as e:
            results.append((block_sizes, pb, None, type(e).__name__))
            print(f"{str(block_sizes):22s}  {str(pb):5s}  FAILED: {type(e).__name__}")

    print()
    print("===== ranked (successful only) =====")
    successful = [
        (bs, pb, t) for bs, pb, t, err in results if t is not None and err is None
    ]
    successful.sort(key=lambda r: r[2])
    for block_sizes, pb, t_ms in successful:
        print(f"{str(block_sizes):22s}  pb={pb}  {t_ms:.3f} ms")


if __name__ == "__main__":
    main()
