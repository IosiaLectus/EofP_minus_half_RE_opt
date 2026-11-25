"""Tests for src/matrix_log.py: matrix logarithm with autograd support."""

import numpy as np
import pytest
import scipy.linalg
import torch

from matrix_log import logm


def _random_spd(n=4, seed=42):
    """A random symmetric positive-definite matrix (real log well-defined)."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, n))
    return m @ m.T + n * np.eye(n)


def test_logm_of_identity_is_zero():
    eye = torch.eye(3, dtype=torch.float64)
    result = logm(eye)
    assert result.shape == (3, 3)
    assert torch.allclose(result, torch.zeros_like(eye), atol=1e-10)


def test_logm_matches_scipy_directly():
    a = _random_spd()
    expected = scipy.linalg.logm(a)
    result = logm(torch.from_numpy(a)).numpy()
    assert np.allclose(result, expected, atol=1e-8)


def test_logm_round_trips_through_expm():
    a = _random_spd(n=5, seed=7)
    result = logm(torch.from_numpy(a)).numpy()
    reconstructed = scipy.linalg.expm(result)
    assert np.allclose(reconstructed, a, atol=1e-8)


def test_logm_gradient_flows():
    x = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    a = x @ x.T + 3 * torch.eye(3, dtype=torch.float64)  # SPD in x
    loss = logm(a).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.any(x.grad != 0)


def test_logm_rejects_non_square_matrix():
    a = torch.randn(2, 3, dtype=torch.float64)
    with pytest.raises(AssertionError):
        logm(a)