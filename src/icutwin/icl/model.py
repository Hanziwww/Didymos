"""TrajPFN (plan Phase G, Architecture): episodes in, predictive distributions out.

``tokens → patch pooling → causal temporal encoder → rows → in-context transformer → heads``:
* **token embedder**: kind; channel = global embedding (zero when anonymized or dropped) + projected per-episode
  random identity; value by piecewise-linear encoding of its squashed value; source decile; flags; position inside
  its patch; label payload (``batch.label_features``); static key.
* **patch pooling**: multi-head attention pooling (one learned query per head, a softmax over the patch's tokens) into
  one vector per patch, plus its absolute time and gap, plus the episode's task vector. Linear in the tokens and
  identical on every device. (The plan's two set-attention layers inside a patch were replaced by this pooling:
  patches hold 1-60 tokens, cross-token interaction is left to the temporal encoder, and no padding is needed.)
* **temporal encoder**: a causal transformer over each patient's patches (pre-norm, RMSNorm, rotary positions on
  continuous time in hours, SwiGLU), the same weights for context and query patients; patients are bucketed by
  length and right-padded (causal attention never reads the padding).
* **task encoder**: the episode's task tokens (``batch._task``) through a small transformer; their mean is the task
  vector.
* **in-context transformer**: per episode the task tokens, the context's train rows (state + label embedding) and the
  test rows (state + mask). Every position attends to the task tokens and train rows only (never to a test row), so
  train rows form the context and its keys and values are independent of the queries (a bedside cache). Queries are
  scaled by QASSMax (TabICLv2): a learned function of log(number of keys) times a query gate.
* **heads** (``heads``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from icutwin.icl import batch as B
from icutwin.icl import heads as H


@dataclass
class Config:
    n_channels: int
    n_static: int
    d_tok: int = 128
    d_id: int = 32
    pool_heads: int = 4
    d_time: int = 256
    time_layers: int = 4
    time_heads: int = 4
    d_task: int = 128
    task_layers: int = 2
    task_heads: int = 4
    d_icl: int = 256
    icl_layers: int = 6
    icl_heads: int = 8
    ffn_mult: float = 8 / 3
    bins: int = 128
    ple_bins: int = 32
    rope_base: float = 10000.0
    bucket_tokens: int = 32768            # padded patch tokens per temporal-encoder bucket
    dropout: float = 0.0
    checkpointing: bool = False           # recompute transformer blocks in the backward pass (memory for time)


PRESETS = {
    "tiny": dict(d_tok=32, d_id=32, pool_heads=2, d_time=32, time_layers=1, time_heads=2, d_task=32, task_layers=1,
                 task_heads=2, d_icl=32, icl_layers=2, icl_heads=2, bins=32, ple_bins=8),
    "S": dict(d_tok=128, d_time=256, time_layers=4, time_heads=4, d_task=128, task_layers=2, d_icl=256,
              icl_layers=6, icl_heads=8),
    "M": dict(d_tok=128, d_time=384, time_layers=8, time_heads=6, d_task=192, task_layers=2, task_heads=4,
              d_icl=512, icl_layers=12, icl_heads=8),
}


def config(preset: str, n_channels: int, n_static: int, **over) -> Config:
    return Config(n_channels=n_channels, n_static=n_static, **{**PRESETS[preset], **over})


# -- building blocks ------------------------------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: float, dropout: float = 0.0):
        super().__init__()
        h = int(round(d * mult / 8)) * 8
        self.w12 = nn.Linear(d, 2 * h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1)
        return self.drop(self.w3(F.silu(a) * b))


def rope(x: torch.Tensor, pos: torch.Tensor, base: float) -> torch.Tensor:
    """Rotary embedding of ``x`` [..., L, h, dh] at real-valued positions ``pos`` [..., L] (hours)."""
    dh = x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh))
    ang = pos.float()[..., None, None] * inv                            # [..., L, 1, dh/2]
    cos, sin = torch.cos(ang), torch.sin(ang)
    x1, x2 = x.float()[..., 0::2], x.float()[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)
    return out.to(x.dtype)


class Block(nn.Module):
    """A pre-norm transformer block. ``causal``: causal self-attention (temporal encoder); else attention with an
    optional key mask (task encoder, in-context transformer) and optional QASSMax."""

    def __init__(self, d: int, heads: int, mult: float, dropout: float = 0.0, qassmax: bool = False):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.ff = SwiGLU(d, mult, dropout)
        self.drop = nn.Dropout(dropout)
        self.qassmax = qassmax
        if qassmax:
            self.base = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, d))
            self.gate = nn.Sequential(nn.Linear(self.dh, 64), nn.GELU(), nn.Linear(64, self.dh))
            nn.init.zeros_(self.base[-1].weight)          # starts as plain attention: scale 1, gate 0
            nn.init.ones_(self.base[-1].bias)
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    def forward(self, x, pos=None, rope_base: float = 1e4, causal: bool = False, key_mask=None, log_n=None):
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, n, 3, self.h, self.dh).unbind(2)
        if pos is not None:
            q, k = rope(q, pos, rope_base), rope(k, pos, rope_base)
        if self.qassmax:                                   # TabICLv2: q * base(log n) * (1 + tanh(gate(q)))
            s = self.base(log_n.view(b, 1, 1)).view(b, 1, self.h, self.dh)
            q = q * s * (1 + torch.tanh(self.gate(q)))
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        mask = None if key_mask is None else key_mask[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal and mask is None)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(b, n, d)))
        return x + self.ff(self.n2(x))


class PLE(nn.Module):
    """Piecewise-linear encoding of a squashed value on ``n`` bins over [-10, 10] plus the value itself (Gorishniy
    et al. 2022), projected to ``d``."""

    def __init__(self, n: int, d: int):
        super().__init__()
        self.register_buffer("edges", torch.linspace(-10, 10, n + 1), persistent=False)
        self.proj = nn.Linear(n + 1, d)

    def forward(self, x):
        lo, hi = self.edges[:-1], self.edges[1:]
        enc = ((x[..., None] - lo) / (hi - lo)).clamp(0, 1)
        return self.proj(torch.cat([enc, x[..., None] / 10], -1))


# -- the model ------------------------------------------------------------------------------------------------------
class TrajPFN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = c = cfg
        d = c.d_tok
        self.embed_kind = nn.Embedding(len(B.KINDS), d)
        self.embed_channel = nn.Embedding(c.n_channels + 1, d, padding_idx=c.n_channels)
        self.embed_static = nn.Embedding(c.n_static + 1, d, padding_idx=c.n_static)
        self.embed_bin = nn.Embedding(B.NO_BIN + 1, d)
        self.id_proj = nn.Linear(c.d_id, d, bias=False)
        self.value = PLE(c.ple_bins, d)
        self.flag = nn.Linear(8, d)
        self.dt = nn.Linear(2, d)
        self.lab = nn.Linear(B.F_LAB, d)
        self.tok_mlp = nn.Sequential(nn.RMSNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        # patch pooling
        self.pool_score = nn.Linear(d, c.pool_heads)
        self.pool_value = nn.Linear(d, d)
        self.patch_out = nn.Linear(d, c.d_time)
        self.patch_time = nn.Linear(4, c.d_time)
        self.patch_task = nn.Linear(c.d_task, c.d_time)
        self.time_blocks = nn.ModuleList([Block(c.d_time, c.time_heads, c.ffn_mult, c.dropout)
                                          for _ in range(c.time_layers)])
        self.time_norm = nn.RMSNorm(c.d_time)
        # task encoder
        self.task_in = nn.Linear(B.F_TASK, c.d_task)
        self.embed_task_kind = nn.Embedding(len(B.TASK_KINDS), c.d_task)
        self.task_chan = nn.Linear(d, c.d_task)
        self.task_blocks = nn.ModuleList([Block(c.d_task, c.task_heads, c.ffn_mult, c.dropout)
                                          for _ in range(c.task_layers)])
        self.task_norm = nn.RMSNorm(c.d_task)
        # rows and the in-context transformer
        self.row_state = nn.Linear(c.d_time, c.d_icl)
        self.row_label = nn.Linear(B.F_LAB, c.d_icl)
        self.row_fc_value = PLE(c.ple_bins, c.d_icl)
        self.row_fc_step = nn.Embedding(256, c.d_icl)
        self.row_fc_chan = nn.Linear(d, c.d_icl)
        self.embed_role = nn.Embedding(3, c.d_icl)                  # task token, train row, test row
        self.mask_token = nn.Parameter(torch.zeros(c.d_icl))
        self.icl_task = nn.Linear(c.d_task, c.d_icl)
        self.icl_blocks = nn.ModuleList([Block(c.d_icl, c.icl_heads, c.ffn_mult, c.dropout, qassmax=True)
                                         for _ in range(c.icl_layers)])
        self.icl_norm = nn.RMSNorm(c.d_icl)
        self.heads = H.Heads(c.d_icl, c.bins, d)

    # -- channels ------------------------------------------------------------------------------------------------
    def channel(self, gid: torch.Tensor, slot: torch.Tensor, rand_id: torch.Tensor) -> torch.Tensor:
        """Global embedding (zero for the sentinel) + the per-episode random identity (zero without a slot)."""
        e = self.embed_channel(gid)
        if len(rand_id):
            r = rand_id[slot.clamp_min(0)]
            e = e + self.id_proj(r) * (slot >= 0).unsqueeze(-1).to(r.dtype)
        return e

    def _run(self, blk: nn.Module, x: torch.Tensor, **kw) -> torch.Tensor:
        if self.cfg.checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(lambda t: blk(t, **kw), x, use_reentrant=False)
        return blk(x, **kw)

    # -- stages --------------------------------------------------------------------------------------------------
    def embed_tokens(self, b: dict) -> torch.Tensor:
        val = b["tok_val"]
        e = (self.embed_kind(b["tok_kind"]) + self.channel(b["tok_gid"], b["tok_slot"], b["rand_id"])
             + self.embed_static(b["tok_skey"]) + self.embed_bin(b["tok_bin"])
             + self.value(val) * b["tok_has"].unsqueeze(-1) + self.flag(b["tok_flag"])
             + self.dt(torch.stack([b["tok_dt"], b["tok_dt"] ** 2], -1)) + self.lab(b["tok_lab"]))
        return e + self.tok_mlp(e)

    def pool_patches(self, x: torch.Tensor, b: dict, task_vec: torch.Tensor) -> torch.Tensor:
        """Multi-head attention pooling of each patch's tokens (a softmax per patch and head), then the patch's
        time features and its episode's task vector."""
        idx = b["tok_patch"]
        n_p = len(b["patch_end_h"])
        hp = self.cfg.pool_heads
        s = self.pool_score(x).float()                                           # [N, hp]
        m = torch.full((n_p, hp), -torch.inf, device=x.device).scatter_reduce(
            0, idx[:, None].expand(-1, hp), s, "amax", include_self=True)
        w = torch.exp(s - m[idx])
        den = torch.zeros(n_p, hp, device=x.device).index_add(0, idx, w)
        w = w / den[idx].clamp_min(1e-12)
        v = self.pool_value(x).float().view(len(x), hp, -1)
        pooled = torch.zeros(n_p, hp, v.shape[-1], device=x.device).index_add(0, idx, w[..., None] * v)
        pooled = pooled.flatten(1).to(x.dtype)
        t = b["patch_end_h"]
        tf = torch.stack([torch.sign(t), torch.log1p(t.abs()) / 5, torch.log1p(b["patch_gap_h"].clamp_min(0)),
                          torch.ones_like(t)], -1)
        ep = b["pat_ep"][b["patch_pat"]]
        return self.patch_out(pooled) + self.patch_time(tf) + self.patch_task(task_vec[ep])

    def encode_time(self, p: torch.Tensor, b: dict) -> torch.Tensor:
        """The causal temporal encoder over each patient's patches, in length buckets; returns [P, d_time]."""
        off = b["pat_off"].tolist()
        lens = [off[i + 1] - off[i] for i in range(len(off) - 1)]
        order = sorted(range(len(lens)), key=lambda i: -lens[i])
        out = torch.zeros_like(p)
        pos_all = b["patch_end_h"]
        i = 0
        while i < len(order):
            lmax = lens[order[i]]
            n_b = max(1, min(len(order) - i, self.cfg.bucket_tokens // max(lmax, 1)))
            pats = order[i:i + n_b]
            i += n_b
            if lmax == 0:
                continue
            idx = torch.full((len(pats), lmax), -1, dtype=torch.long, device=p.device)
            for r, pi in enumerate(pats):
                idx[r, :lens[pi]] = torch.arange(off[pi], off[pi + 1], device=p.device)
            ok = idx >= 0
            x = torch.where(ok[..., None], p[idx.clamp_min(0)], torch.zeros((), dtype=p.dtype, device=p.device))
            pos = torch.where(ok, pos_all[idx.clamp_min(0)], torch.zeros((), device=p.device))
            for blk in self.time_blocks:
                x = self._run(blk, x, pos=pos, rope_base=self.cfg.rope_base, causal=True)
            out[idx[ok]] = self.time_norm(x)[ok].to(out.dtype)
        return out

    def encode_task(self, b: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(task tokens padded [B, T, d_task], their mask [B, T], the task vector [B, d_task])."""
        n_ep = int(b["n_episodes"])
        x = (self.task_in(b["task_feat"]) + self.embed_task_kind(b["task_kind"])
             + self.task_chan(self.channel(b["task_gid"], b["task_slot"], b["rand_id"])))
        xs, mask = _pad_by(x, b["task_ep"], n_ep)
        for blk in self.task_blocks:
            xs = blk(xs, key_mask=mask)
        xs = self.task_norm(xs)
        vec = (xs * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return xs, mask, vec

    def rows(self, h: torch.Tensor, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """(train rows [R_tr, d_icl], test rows [R_te, d_icl])."""
        tr = self.row_state(h[b["tr_patch"]]) + self.row_label(b["tr_lab"]) + self.embed_role.weight[1]
        if b["tr_yf"].numel():                                          # a forecast row's values as its label
            u, m = b["tr_yf"], b["tr_mf"].to(h.dtype)
            n, hf, cf = u.shape
            chan = self.forecast_channels(b)[b["tr_ep"]]                    # [R, C, d_tok]
            cell = (self.row_fc_value(u) + self.row_fc_step(torch.arange(hf, device=h.device))[None, :, None, :]
                    + self.row_fc_chan(chan)[:, None, :, :])
            tr = tr + (cell * m[..., None]).sum((1, 2)) / m.sum((1, 2)).clamp_min(1)[:, None]
        te = self.row_state(h[b["te_patch"]]) + self.mask_token + self.embed_role.weight[2]
        return tr, te

    def forecast_channels(self, b: dict) -> torch.Tensor:
        return self.channel(b["ep_fc_gid"], b["ep_fc_slot"], b["rand_id"])

    def icl(self, tr: torch.Tensor, te: torch.Tensor, task: torch.Tensor, task_mask: torch.Tensor,
            b: dict) -> torch.Tensor:
        """The in-context transformer; returns the test rows' final states [R_te, d_icl]."""
        n_ep = int(b["n_episodes"])
        tk = self.icl_task(task) + self.embed_role.weight[0]
        trs, trm = _pad_by(tr, b["tr_ep"], n_ep)
        tes, tem = _pad_by(te, b["te_ep"], n_ep)
        x = torch.cat([tk, trs, tes], 1)
        keys = torch.cat([task_mask, trm, torch.zeros_like(tem)], 1)
        log_n = torch.log(keys.sum(1).float().clamp_min(1)).to(x.dtype)
        for blk in self.icl_blocks:
            x = self._run(blk, x, key_mask=keys, log_n=log_n)
        x = self.icl_norm(x)
        z = x[:, tk.shape[1] + trs.shape[1]:]
        return z[tem]

    def forward(self, b: dict) -> dict:
        task, task_mask, task_vec = self.encode_task(b)
        tok = self.embed_tokens(b)
        p = self.pool_patches(tok, b, task_vec)
        h = self.encode_time(p, b)
        tr, te = self.rows(h, b)
        z = self.icl(tr, te, task, task_mask, b)
        fc = self.forecast_channels(b) if b["te_yf"].numel() else None
        loss, out = H.row_losses(self.heads, z, b, fc)
        return {"z": z, "row_loss": loss, "out": out, "h": h}

    def loss(self, b: dict, res: dict | None = None) -> tuple[torch.Tensor, dict]:
        res = res if res is not None else self.forward(b)
        per_ep, has = H.episode_mean(res["row_loss"], b["te_ep"], int(b["n_episodes"]))
        total = per_ep[has].mean() if has.any() else res["z"].sum() * 0
        return total, {"per_episode": per_ep.detach(), "has": has, **res}


def _pad_by(x: torch.Tensor, group: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of ``x`` grouped by ``group`` (sorted or not) into [n, L, d] with a mask, in their order of appearance."""
    counts = torch.bincount(group, minlength=n)
    lmax = int(counts.max()) if len(group) else 0
    out = torch.zeros(n, max(lmax, 0), x.shape[-1], dtype=x.dtype, device=x.device)
    mask = torch.zeros(n, max(lmax, 0), dtype=torch.bool, device=x.device)
    if len(group):
        order = torch.argsort(group, stable=True)
        g = group[order]
        start = torch.cumsum(counts, 0) - counts
        rank = torch.arange(len(g), device=x.device) - start[g]
        out[g, rank] = x[order]
        mask[g, rank] = True
    return out, mask


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def describe(cfg: Config) -> dict:
    return asdict(cfg)
