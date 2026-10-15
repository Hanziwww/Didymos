"""Model presets (plan S6): HF ``Qwen3ForCausalLM`` built from a config, trained from scratch.

The S, M and L presets follow the CoMET family the plan names (S 62M: 6 layers, width 768; M 119M: 12 layers,
768; L 1B: 16 layers, 2048), with GQA and a 4x SwiGLU width, tied embeddings as Qwen3's small models; parameter
counts depend on the vocabulary (``n_params``). ``ML`` is the intermediate width the M5 scaling notes named
(16 layers, 1024; 270M with the r3 vocabulary): round 2's step up from M at the same, privacy-bound passes
(``docs/m7_authenticity.md``). ``tiny`` and ``pilot`` are local sanity presets (overfit test, bounded pilot), not
candidates for the real run.
"""

from __future__ import annotations

PRESETS: dict[str, dict] = {
    "tiny": dict(hidden_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                 intermediate_size=384),
    "pilot": dict(hidden_size=256, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                  intermediate_size=1024),
    "pilot_wide": dict(hidden_size=512, num_hidden_layers=6, num_attention_heads=8, num_key_value_heads=4,
                       intermediate_size=2048),
    "S": dict(hidden_size=768, num_hidden_layers=6, num_attention_heads=12, num_key_value_heads=4,
              intermediate_size=3072),
    "M": dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=4,
              intermediate_size=3072),
    "ML": dict(hidden_size=1024, num_hidden_layers=16, num_attention_heads=16, num_key_value_heads=4,
               intermediate_size=4096),
    "L": dict(hidden_size=2048, num_hidden_layers=16, num_attention_heads=16, num_key_value_heads=8,
              intermediate_size=8192),
}
def _context() -> int:
    from icutwin.tokenization.spec import load_spec

    return int(load_spec()["windows"]["max_tokens"])


CONTEXT = _context()   # tokens per window (grammar spec windows.max_tokens); positions add room for the goal block
# and, from round 3, for the clock blocks written after the cut (a 32,768-token first window spanning 70 days holds
# about 365 of their tokens): MARGIN positions beyond CONTEXT
MARGIN = 1024


ROPE_THETA = 1_000_000.0   # Qwen3's RoPE base for its long-context models (M_main, 8k: transformers' 10,000)


def qwen3_config(preset: str, vocab_size: int, pad_token_id: int, bos_token_id: int, eos_token_id: int,
                 max_positions: int = CONTEXT + MARGIN, attn_implementation: str = "sdpa", rope_theta: float = ROPE_THETA):
    from transformers import Qwen3Config

    p = PRESETS[preset]
    cfg = Qwen3Config(vocab_size=vocab_size, max_position_embeddings=max_positions,
                      head_dim=p["hidden_size"] // p["num_attention_heads"], tie_word_embeddings=True,
                      pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                      use_cache=False, rope_parameters={"rope_type": "default", "rope_theta": float(rope_theta)}, **p)
    cfg._attn_implementation = attn_implementation
    return cfg


def n_params(preset: str, vocab_size: int) -> dict:
    """Parameter counts without building the model: embedding (tied with the output layer) and the rest."""
    p = PRESETS[preset]
    d, layers, heads, kv = p["hidden_size"], p["num_hidden_layers"], p["num_attention_heads"], p["num_key_value_heads"]
    hd = d // heads
    attn = d * heads * hd + 2 * d * kv * hd + heads * hd * d + 2 * hd   # q, k, v, o and Qwen3's q/k norms
    mlp = 3 * d * p["intermediate_size"]
    per_layer = attn + mlp + 2 * d
    body = layers * per_layer + d
    emb = vocab_size * d
    return {"embedding": emb, "non_embedding": body, "total": emb + body}
