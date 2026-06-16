"""
Jagged HSTU Attention
=====================

Per-sequence causal SiLU-gated attention over jagged-packed Q/K/V::

    scores = silu(Q @ K^T * alpha) * attn_scale
    scores = where(causal_mask, scores, 0)
    out    = scores @ V

Tensor shapes::

    q, k, v      : [L, H, D]     (L = sum of seq_lens, jagged-packed)
    seq_offsets  : [B + 1]        (int cumulative offsets)
    output       : [L, H, D]

The kernel iterates per-sequence via ``hl.jagged_tile``, with natural
3-D indexing ``q[abs_idx, :, :]`` on the original ``[L, H, D]`` shape.
No host-side padding, no flatten, no in-kernel reshape.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


def reference_jagged_hstu_attention(
    max_seq_len: int,
    alpha: float,
    attn_scale: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    """Direct per-sequence reference — no padding, no reshape.

    Splits the jagged L dim into per-sequence slices, runs per-head bmm
    on each slice. Math invariant for the kernel.
    """
    output = torch.zeros_like(v)
    seq_lens = (seq_offsets[1:] - seq_offsets[:-1]).tolist()

    q_split = torch.split(q, seq_lens, dim=0)
    k_split = torch.split(k, seq_lens, dim=0)
    v_split = torch.split(v, seq_lens, dim=0)

    for i, (q_b, k_b, v_b) in enumerate(zip(q_split, k_split, v_split)):
        # [seq_len, H, D] -> [H, seq_len, D] for per-head bmm
        q_h = q_b.transpose(0, 1)
        k_h = k_b.transpose(0, 1)
        v_h = v_b.transpose(0, 1)

        scores = torch.bmm(q_h, k_h.transpose(-2, -1)) * alpha
        scores = F.silu(scores) * attn_scale
        n = scores.size(-1)
        causal = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
        scores = torch.where(causal[None, :, :], scores, 0.0)

        out_b = torch.bmm(scores.to(v.dtype), v_h).transpose(0, 1)
        s = int(seq_offsets[i])
        e = int(seq_offsets[i + 1])
        output[s:e] = out_b

    return output


@helion.kernel(
    static_shapes=True,
    autotune_baseline_fn=reference_jagged_hstu_attention,
)
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
        # ``tile_b.begin`` gives a scalar program index (same idiom as
        # v1's ``jagged_hstu_attn.py:114``). With scalar ``start``,
        # downstream shapes stay rank-aligned with HBM (no leading
        # tile_b phantom dim).
        start = seq_offsets[tile_b.begin]                              # scalar
        seq_lens = seq_offsets[tile_b.index + 1] - seq_offsets[tile_b]  # [tile_b]

        for tile_q in hl.jagged_tile(seq_lens):
            q_idx = start + tile_q.index                                # [tile_q]
            # Transpose at the load site so Helion hoists it outside
            # the KV loop and lowers the load-mask as a multiply
            # rather than a where+broadcast (matches v2 codegen).
            q_blk = q[q_idx, :, :].transpose(0, 1)                       # [H, tile_q, D]
            # Acc lives in dot-output orientation so the value-bmm
            # update can be added directly with no per-iter transpose.
            acc = hl.zeros([H, tile_q, D], dtype=torch.float32)

            for tile_kv in hl.jagged_tile(seq_lens):
                kv_idx = start + tile_kv.index                           # [tile_kv]
                k_blk = k[kv_idx, :, :].transpose(0, 1)                  # [H, tile_kv, D]
                v_blk = v[kv_idx, :, :].transpose(0, 1)                  # [H, tile_kv, D]

                # [H, tile_q, D] @ [H, D, tile_kv] -> [H, tile_q, tile_kv]
                scores = torch.bmm(q_blk, k_blk.transpose(-2, -1)) * alpha
                scores = F.silu(scores) * attn_scale

                causal_mask = (
                    tile_q.index.unsqueeze(1) >= tile_kv.index.unsqueeze(0)
                )                                                         # [tile_q, tile_kv]
                scores = torch.where(causal_mask[None, :, :], scores, 0.0)

                # [H, tile_q, tile_kv] @ [H, tile_kv, D] -> [H, tile_q, D]
                acc = acc + torch.bmm(scores.to(v.dtype), v_blk)

            # [H, tile_q, D] -> [tile_q, H, D] for the jagged write.
            out[q_idx, :, :] = acc.transpose(0, 1).to(out.dtype)

    return out


def main() -> None:
    # Same workload as jagged_hstu_attn_2.py so v3 perf is directly
    # comparable to v2 on TPU.
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
    L = int(seq_offsets[-1].item())

    q = torch.randn(L, heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(L, heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(L, heads, head_dim, dtype=dtype, device=device)

    run_example(
        lambda *a: jagged_hstu_attention(*a),
        lambda *a: reference_jagged_hstu_attention(*a),
        (max_seq_len, alpha, attn_scale, q, k, v, seq_offsets),
        atol=1e-2,
        rtol=1e-2,
    )


if __name__ == "__main__":
    main()
