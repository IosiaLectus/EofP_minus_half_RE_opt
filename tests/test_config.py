"""Tests for src/config.py: Pauli matrices and the 2D identity."""

import pytest
import torch

from config import id_2d, pauli_x, pauli_y, pauli_z


@pytest.mark.parametrize("pauli", [pauli_x, pauli_y, pauli_z])
def test_pauli_is_hermitian(pauli):
    assert torch.allclose(pauli, pauli.mH)


@pytest.mark.parametrize("pauli", [pauli_x, pauli_y, pauli_z])
def test_pauli_squares_to_identity(pauli):
    assert torch.allclose(pauli @ pauli, id_2d, atol=1e-6)


@pytest.mark.parametrize(
    "a, b",
    [(pauli_x, pauli_y), (pauli_y, pauli_z), (pauli_z, pauli_x)],
)
def test_pauli_anticommutation(a, b):
    assert torch.allclose(a @ b + b @ a, torch.zeros_like(a), atol=1e-6)


def test_id_2d_is_identity():
    z = torch.zeros((2, 2), dtype=id_2d.dtype)
    assert torch.allclose(id_2d @ pauli_x, pauli_x)
    assert torch.allclose(z @ id_2d, z)