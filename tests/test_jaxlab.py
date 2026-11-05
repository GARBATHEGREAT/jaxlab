import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxlab.attention import full_window_attention, local_attention
from jaxlab.kernels import rmsnorm_pallas, rmsnorm_ref
from jaxlab.streaming import causal_conv_offline, causal_conv_streaming


def numpy_window_attention(q, k, v, window):
    """Slow loop reference, written independently of the JAX versions."""
    t, h, d = q.shape
    out = np.zeros_like(q)
    for i in range(t):
        lo = max(0, i - window + 1)
        for head in range(h):
            s = q[i, head] @ k[lo : i + 1, head].T / np.sqrt(d)
            s = np.exp(s - s.max())
            out[i, head] = (s / s.sum()) @ v[lo : i + 1, head]
    return out


def rand(key, *shape, dtype=jnp.float32):
    return jax.random.normal(jax.random.PRNGKey(key), shape, dtype)


@pytest.mark.parametrize("t,window", [(32, 8), (64, 16), (48, 48)])
def test_attention_matches_numpy_reference(t, window):
    q, k, v = (rand(i, t, 2, 8) for i in range(3))
    ref = numpy_window_attention(*(np.asarray(a) for a in (q, k, v)), window)
    np.testing.assert_allclose(full_window_attention(q, k, v, window), ref, atol=1e-5)
    np.testing.assert_allclose(local_attention(q, k, v, window), ref, atol=1e-5)


def test_first_position_attends_only_to_itself():
    q, k, v = (rand(i, 16, 1, 4) for i in range(3))
    out = local_attention(q, k, v, 4)
    np.testing.assert_allclose(out[0], v[0], atol=1e-6)


def test_local_attention_rejects_ragged_length():
    q = rand(0, 30, 1, 4)
    with pytest.raises(ValueError):
        local_attention(q, q, q, 8)


def test_local_attention_does_not_retrace_for_same_shapes():
    traces = []

    @jax.jit
    def f(q, k, v):
        traces.append(1)
        return local_attention(q, k, v, 8)

    q, k, v = (rand(i, 32, 2, 8) for i in range(3))
    f(q, k, v).block_until_ready()
    f(q + 1, k, v).block_until_ready()
    assert len(traces) == 1


@pytest.mark.parametrize("chunk", [1, 4, 16, 64])
@pytest.mark.parametrize("ksize", [1, 3, 5])
def test_streaming_equals_offline(chunk, ksize):
    x = rand(1, 64, 6)
    kernel = rand(2, ksize, 6, 5)
    np.testing.assert_allclose(
        causal_conv_streaming(x, kernel, chunk), causal_conv_offline(x, kernel), atol=1e-5
    )


def test_streaming_is_causal():
    x = rand(1, 32, 4)
    kernel = rand(2, 5, 4, 3)
    base = causal_conv_streaming(x, kernel, 8)
    changed = causal_conv_streaming(x.at[20:].add(5.0), kernel, 8)
    np.testing.assert_array_equal(base[:20], changed[:20])


@pytest.mark.parametrize("dtype,atol", [(jnp.float32, 1e-5), (jnp.bfloat16, 3e-2)])
def test_pallas_rmsnorm_matches_reference(dtype, atol):
    x = rand(3, 16, 256, dtype=dtype)
    w = rand(4, 256, dtype=dtype)
    got = rmsnorm_pallas(x, w).astype(jnp.float32)
    want = rmsnorm_ref(x, w).astype(jnp.float32)
    np.testing.assert_allclose(got, want, atol=atol)


def test_pallas_rmsnorm_rejects_bad_tiling():
    with pytest.raises(ValueError):
        rmsnorm_pallas(rand(0, 10, 256), rand(1, 256))
    with pytest.raises(ValueError):
        rmsnorm_pallas(rand(0, 16, 100), rand(1, 100))
