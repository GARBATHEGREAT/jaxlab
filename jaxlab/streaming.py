"""Streaming causal 1-D convolution with carried state.

A streaming model must give the same answer chunk by chunk as it would on the
whole signal. The only thing carried between chunks is the last (K - 1) input
frames, so the loop is a pure function of (state, chunk) and compiles to a
single `lax.scan` with static shapes.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

_DN = ("NWC", "WIO", "NWC")


def causal_conv_offline(x, kernel):
    """x: (T, C_in); kernel: (K, C_in, C_out). Output (T, C_out), left-padded with zeros."""
    k = kernel.shape[0]
    xp = jnp.pad(x, ((k - 1, 0), (0, 0)))
    return lax.conv_general_dilated(xp[None], kernel, (1,), "VALID", dimension_numbers=_DN)[0]


def init_state(kernel):
    k, c_in, _ = kernel.shape
    return jnp.zeros((k - 1, c_in), kernel.dtype)


def stream_step(state, chunk, kernel):
    """One chunk: chunk (n, C_in) -> (new_state, out (n, C_out))."""
    k = kernel.shape[0]
    joined = jnp.concatenate([state, chunk], axis=0)
    out = lax.conv_general_dilated(joined[None], kernel, (1,), "VALID", dimension_numbers=_DN)[0]
    new_state = joined[joined.shape[0] - (k - 1):] if k > 1 else state
    return new_state, out


def causal_conv_streaming(x, kernel, chunk: int):
    """Process x in fixed-size chunks with one compiled scan. T must divide by chunk."""
    t, c_in = x.shape
    if t % chunk:
        raise ValueError(f"length {t} must be a multiple of chunk {chunk}")
    chunks = x.reshape(t // chunk, chunk, c_in)

    def body(state, c):
        return stream_step(state, c, kernel)

    _, outs = lax.scan(body, init_state(kernel), chunks)
    return outs.reshape(t, -1)
