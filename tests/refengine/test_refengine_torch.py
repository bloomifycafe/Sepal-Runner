"""Optional cross-check of the primitives against their PyTorch equivalents (float64, CPU). Skipped without PyTorch."""
import numpy as np
import pytest

from sepal.refengine.attention import grouped_query_attention
from sepal.refengine.deltanet import causal_conv
from sepal.refengine.primitives import rms_norm, sigmoid, silu, softplus, swiglu_mlp

torch = pytest.importorskip("torch")
F = torch.nn.functional
TOLERANCE = 1e-12


@pytest.fixture
def rng():
    return np.random.default_rng(11)


def t(array):
    return torch.from_numpy(np.ascontiguousarray(array))


def causal_attention(q, keys, values):
    """PyTorch's causal attention on [tokens, heads, dim] arrays, each kv head repeated for its group of query heads."""
    def repeat(x):
        return t(x).transpose(0, 1).repeat_interleave(q.shape[1] // x.shape[1], dim=0)   # [heads, tokens, dim]
    return F.scaled_dot_product_attention(t(q).transpose(0, 1), repeat(keys), repeat(values), is_causal=True).transpose(0, 1).numpy()


def test_elementwise_match_torch(rng):
    x = 4 * rng.standard_normal(64)
    np.testing.assert_allclose(silu(x), F.silu(t(x)).numpy(), atol=TOLERANCE)
    np.testing.assert_allclose(sigmoid(x), torch.sigmoid(t(x)).numpy(), atol=TOLERANCE)
    np.testing.assert_allclose(softplus(x), F.softplus(t(x)).numpy(), atol=TOLERANCE)


def test_rms_norm_matches_torch(rng):
    x, w = rng.standard_normal((6, 24)), 0.1 * rng.standard_normal(24)
    expected = F.rms_norm(t(x), (24,), weight=t(1.0 + w), eps=1e-6).numpy()
    np.testing.assert_allclose(rms_norm(x, w, 1e-6), expected, atol=TOLERANCE)


def test_swiglu_matches_torch(rng):
    x, gate, up, down = rng.standard_normal((5, 24)), rng.standard_normal((40, 24)), rng.standard_normal((40, 24)), rng.standard_normal((24, 40))
    expected = F.linear(F.silu(F.linear(t(x), t(gate))) * F.linear(t(x), t(up)), t(down)).numpy()
    np.testing.assert_allclose(swiglu_mlp(x, gate, up, down), expected, atol=TOLERANCE)


def test_causal_conv_matches_torch_conv1d(rng):
    """A left-padded depthwise Conv1d over the whole sequence, and the same from a cached tail."""
    channels, taps, tokens, split = 10, 4, 9, 5
    inputs, kernel = rng.standard_normal((tokens, channels)), rng.standard_normal((channels, taps))
    expected = F.conv1d(t(inputs.T[None]), t(kernel[:, None, :]), padding=taps - 1, groups=channels)[0, :, :tokens].T.numpy()
    whole, _ = causal_conv(inputs, kernel, np.zeros((taps - 1, channels)))
    np.testing.assert_allclose(whole, expected, atol=TOLERANCE)
    _, tail = causal_conv(inputs[:split], kernel, np.zeros((taps - 1, channels)))
    continued, _ = causal_conv(inputs[split:], kernel, tail)
    np.testing.assert_allclose(continued, expected[split:], atol=TOLERANCE)


def test_grouped_query_attention_matches_torch(rng):
    """Causal scaled-dot-product attention with each kv head repeated for its group of query heads."""
    tokens, heads, kv_heads, dim = 7, 4, 2, 16
    q = rng.standard_normal((tokens, heads, dim))
    keys, values = rng.standard_normal((2, tokens, kv_heads, dim))
    np.testing.assert_allclose(grouped_query_attention(q, keys, values, 0, dim ** -0.5), causal_attention(q, keys, values), atol=TOLERANCE)


def test_attention_over_a_parent_context_matches_torch(rng):
    """A branch attends to all of its parent's keys and causally to its own: the last rows of one causal pass."""
    context, tokens, heads, kv_heads, dim = 5, 4, 4, 2, 16
    q = rng.standard_normal((context + tokens, heads, dim))
    keys, values = rng.standard_normal((2, context + tokens, kv_heads, dim))
    expected = causal_attention(q, keys, values)[context:]
    np.testing.assert_allclose(grouped_query_attention(q[context:], keys, values, context, dim ** -0.5), expected, atol=TOLERANCE)
