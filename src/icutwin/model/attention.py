"""``varlen``: padding-free packed attention with PyTorch's own variable-length FlashAttention kernels
(``torch.nn.attention.varlen.varlen_attn``, PyTorch 2.13), registered as a transformers attention implementation.

It is what FlashAttention-2 does for packed batches (plan S6: padding-free packing), without the separate
``flash-attn`` package, whose builds do not cover every GPU (Blackwell sm_120 on the server). A batch is either
* packed: one row of whole windows with ``cu_seq_lens_q``/``max_length_q`` from ``data.flat_batches``; every
  window attends causally to itself only;
* right-padded (evaluation): every row is one sequence. Attention is causal, so no real token reaches the
  padding after it; the padded positions' outputs are never targets.
transformers builds no attention mask for a registered implementation it does not know (the mask would be
O(tokens^2)); none is needed here. ``pilot.attention_parity`` checks it against SDPA on separate windows.
"""

from __future__ import annotations

NAME = "varlen"


def varlen_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0,
                             cu_seq_lens_q=None, cu_seq_lens_k=None, max_length_q=None, max_length_k=None, **kwargs):
    """transformers attention interface: ``query`` (batch, heads, tokens, head_dim), ``key``/``value`` with the
    KV heads; returns (batch, tokens, heads, head_dim) and no weights."""
    import torch
    from torch.nn.attention.varlen import varlen_attn

    b, h, t, d = query.shape
    # the kernels take bf16/fp16 only, and autocast does not cast for this op as it does for SDPA: with float32
    # master weights, Qwen3's q/k norms return float32 under autocast (their weight is float32)
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else query.dtype
    dtype = dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    out_dtype = query.dtype if not torch.is_autocast_enabled("cuda") else dtype
    query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)
    if cu_seq_lens_q is None:   # right-padded rows, one sequence each
        cu_seq_lens_q = torch.arange(0, (b + 1) * t, t, dtype=torch.int32, device=query.device)
        max_length_q = t
    elif b != 1:
        raise ValueError("a packed batch is one row (data.flat_batches)")
    cu_k = cu_seq_lens_q if cu_seq_lens_k is None else cu_seq_lens_k
    out = varlen_attn(query.transpose(1, 2).reshape(b * t, h, d),
                      key.transpose(1, 2).reshape(b * t, key.shape[1], d),
                      value.transpose(1, 2).reshape(b * t, value.shape[1], d),
                      cu_seq_lens_q.to(torch.int32), cu_k.to(torch.int32), int(max_length_q),
                      int(max_length_k or max_length_q), scale=scaling, window_size=(-1, 0),
                      enable_gqa=key.shape[1] != h)
    return out.reshape(b, t, h, d).to(out_dtype), None


def register() -> None:
    from transformers import AttentionInterface

    AttentionInterface.register(NAME, varlen_attention_forward)


register()
