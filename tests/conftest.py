"""Shared fixtures for the test suite."""

import pytest
import torch


@pytest.fixture
def make_state():
    """Factory for reproducible, normalised random pure states.

    ``make_state((2, 2, 3, 3), seed=1)`` returns a complex128 tensor of that
    shape with unit 2-norm.
    """

    def _make(dims, seed=0):
        gen = torch.Generator().manual_seed(seed)
        psi = torch.randn(tuple(dims), dtype=torch.cdouble, generator=gen)
        return psi / torch.linalg.vector_norm(psi)

    return _make


@pytest.fixture
def bell_state():
    """(|00> + |11>)/sqrt(2) as a (2, 2) complex128 tensor."""
    psi = torch.zeros(2, 2, dtype=torch.cdouble)
    psi[0, 0] = psi[1, 1] = 2 ** -0.5
    return psi
