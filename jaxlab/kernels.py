"""A Pallas RMSNorm kernel plus a plain-JAX reference.

Layout choices follow TPU tiling rules: the block's last dimension is the full
hidden size (a multiple of 128 lanes) and the row block is a multiple of 8
sublanes. By default the kernel runs with `interpret=True`, which executes the
same kernel body on CPU so it can be tested anywhere. It has NOT been run on a
real TPU; pass `interpret=False` on a TPU VM to compile it with Mosaic.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def rmsnorm_ref(x, w, eps: float = 1e-6):
    xf = x.astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (xf * inv * w.astype(jnp.float32)).astype(x.dtype)


def _rmsnorm_kernel(x_ref, w_ref, o_ref, *, eps):
    x = x_ref[...].astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    o_ref[...] = (x * inv * w_ref[...].astype(jnp.float32)).astype(o_ref.dtype)


def rmsnorm_pallas(x, w, eps: float = 1e-6, block_rows: int = 8, interpret: bool = True):
    rows, hidden = x.shape
    if rows % block_rows:
        raise ValueError(f"rows {rows} must be a multiple of block_rows {block_rows}")
    if hidden % 128:
        raise ValueError(f"hidden {hidden} must be a multiple of 128 for TPU lane tiling")
    return pl.pallas_call(
        partial(_rmsnorm_kernel, eps=eps),
        grid=(rows // block_rows,),
        in_specs=[
            pl.BlockSpec((block_rows, hidden), lambda i: (i, 0)),
            pl.BlockSpec((1, hidden), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((block_rows, hidden), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=interpret,
    )(x, w.reshape(1, hidden))
