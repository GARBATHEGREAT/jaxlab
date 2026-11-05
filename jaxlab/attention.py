"""Causal sliding-window attention, two ways.

Shapes are (T, H, D): time, heads, head dimension. Position i attends to keys
j with i - window < j <= i (so `window` keys including itself).

* full_window_attention: builds the whole T x T score matrix and masks it.
  Cost grows as O(T^2) in both compute and memory.
* local_attention: splits time into blocks of `window` and lets each block see
  only itself and the previous block (2 * window keys), so cost is O(T * window).
  All shapes are static, which is what XLA (and TPUs) want.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def full_window_attention(q, k, v, window: int):
    t, _, d = q.shape
    scores = jnp.einsum("thd,shd->hts", q, k) / jnp.sqrt(d).astype(q.dtype)
    i = jnp.arange(t)[:, None]
    j = jnp.arange(t)[None, :]
    mask = (j <= i) & (j > i - window)
    scores = jnp.where(mask[None], scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("hts,shd->thd", weights, v)


def local_attention(q, k, v, window: int):
    t, h, d = q.shape
    if t % window:
        raise ValueError(f"sequence length {t} must be a multiple of window {window}")
    nb = t // window
    qb = q.reshape(nb, window, h, d)
    kb = k.reshape(nb, window, h, d)
    vb = v.reshape(nb, window, h, d)
    # Keys/values for block b are [block b-1, block b]; block -1 is zeros.
    kk = jnp.concatenate([jnp.concatenate([jnp.zeros_like(kb[:1]), kb[:-1]], 0), kb], axis=1)
    vv = jnp.concatenate([jnp.concatenate([jnp.zeros_like(vb[:1]), vb[:-1]], 0), vb], axis=1)
    scores = jnp.einsum("bthd,bshd->bhts", qb, kk) / jnp.sqrt(d).astype(q.dtype)
    # Within a block, query offset t may see key slot s when t < s <= t + window.
    tq = jnp.arange(window)[:, None]
    ks = jnp.arange(2 * window)[None, :]
    allowed = (ks > tq) & (ks <= tq + window)
    # The first block has no previous block: its first `window` key slots are padding.
    first_block = jnp.arange(nb)[:, None, None] == 0
    allowed = allowed[None] & ~(first_block & (ks < window)[None])
    scores = jnp.where(allowed[:, None], scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhts,bshd->bthd", weights, vv)
    return out.reshape(t, h, d)
