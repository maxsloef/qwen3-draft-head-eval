"""EAGLE-3 draft head, re-implemented for scoring (not serving).

All four public Qwen3-8B heads share this layout: fc (3*4096 -> 4096) over three trunk layers,
one decoder layer whose attention reads cat(embedding, feature) (8192 wide), a final norm and
a 32K-token output layer with a draft->target id offset table (d2t).

Conventions checked against SafeAILab/EAGLE (cnets.py) and RedHatAI's eagle3.py:
- position i gets the trunk feature at i and the embedding of token i+1, and predicts token i+2;
- later draft steps reuse the head's own pre-norm output as the feature (no fc), with the next token;
- the residual stream is the feature before hidden_norm (SafeAILab) or after it (RedHatAI, norm_before_residual).
"""
import math

import torch
import torch.nn.functional as F

N_HEADS, N_KV, HEAD_DIM, EPS, THETA = 32, 8, 128, 1e-6, 1e6


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rope(x, pos):
    """x [B,H,T,d]; pos [T] or [B,T] (absolute positions)."""
    d = x.shape[-1]
    inv = 1.0 / (THETA ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = pos[..., None].float() * inv
    if ang.dim() == 3:
        ang = ang[:, None]
    cos, sin = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class Head:
    def __init__(self, sd, base_embed, norm_before_residual, device):
        pref = "midlayer." if any(k.startswith("midlayer.") for k in sd) else "layers.0."
        g = lambda k: sd[k].to(device=device, dtype=torch.float32)
        self.fc = g("fc.weight")
        self.hn, self.iln, self.pln = g(pref + "hidden_norm.weight"), g(pref + "input_layernorm.weight"), g(pref + "post_attention_layernorm.weight")
        self.wq, self.wk, self.wv, self.wo = (g(pref + f"self_attn.{n}_proj.weight") for n in "qkvo")
        self.wg, self.wu, self.wd = (g(pref + f"mlp.{n}_proj.weight") for n in ("gate", "up", "down"))
        self.norm, self.lm = g("norm.weight"), g("lm_head.weight")
        self.own_embed = "embed_tokens.weight" in sd
        self.embed = sd["embed_tokens.weight"].to(device) if self.own_embed else base_embed
        d2t = sd["d2t"].to(device).long()
        self.tids = torch.arange(d2t.numel(), device=device) + d2t      # draft index -> target token id
        self.t2d = sd["t2d"].to(device).bool()                          # target token id -> is it draftable
        self.nbr = norm_before_residual

    def _inputs(self, tok, hid, nbr):
        emb = self.embed[tok].float()
        if nbr:
            hid = rms(hid, self.hn)
            res = hid
        else:
            res = hid
            hid = rms(hid, self.hn)
        x = torch.cat([rms(emb, self.iln), hid], -1)
        B, T, _ = x.shape
        rep = N_HEADS // N_KV
        q = (x @ self.wq.T).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = (x @ self.wk.T).view(B, T, N_KV, HEAD_DIM).transpose(1, 2).repeat_interleave(rep, dim=1)
        v = (x @ self.wv.T).view(B, T, N_KV, HEAD_DIM).transpose(1, 2).repeat_interleave(rep, dim=1)
        return q, k, v, res

    def _finish(self, res, a):
        B, _, T, _ = a.shape
        h = res + a.transpose(1, 2).reshape(B, T, -1) @ self.wo.T
        m = rms(h, self.pln)
        return h + (F.silu(m @ self.wg.T) * (m @ self.wu.T)) @ self.wd.T

    def logits(self, g):
        return rms(g, self.norm) @ self.lm.T

    def step1(self, tok_next, feats3, nbr=None, keep=True):
        """tok_next [B,T] = token i+1 at position i; feats3 [B,T,3*4096]. Returns pre-norm output g [B,T,4096]."""
        nbr = self.nbr if nbr is None else nbr
        q, k, v, res = self._inputs(tok_next, feats3 @ self.fc.T, nbr)
        pos = torch.arange(q.shape[2], device=q.device)
        q, k = rope(q, pos), rope(k, pos)
        if keep:
            self.kr, self.vr, self.kc, self.vc = k, v, [], []
        return self._finish(res, F.scaled_dot_product_attention(q, k, v, is_causal=True))

    def step_next(self, tok, g_prev, s):
        """Draft step s>=2 for every anchor at once. tok [B,T] = token i+s; g_prev = this anchor's previous output.
        The query at anchor i sees the real keys at positions <= i plus this anchor's own earlier draft steps."""
        q, k, v, res = self._inputs(tok, g_prev, self.nbr)
        B, H, T, d = q.shape
        pos = torch.arange(T, device=q.device)[None].expand(B, T) + (s - 1)
        q, k = rope(q, pos), rope(k, pos)
        self.kc.append(k)
        self.vc.append(v)
        real = q @ self.kr.transpose(-1, -2) / math.sqrt(d)
        real = real.masked_fill(torch.ones(T, T, device=q.device, dtype=torch.bool).triu(1), float("-inf"))
        chain = torch.stack([(q * kc).sum(-1) for kc in self.kc], -1) / math.sqrt(d)
        w = torch.cat([real, chain], -1).softmax(-1)
        a = w[..., :T] @ self.vr
        for c, vc in enumerate(self.vc):
            a = a + w[..., T + c, None] * vc
        return self._finish(res, a)

    def clear(self):
        self.kr = self.vr = self.kc = self.vc = None
