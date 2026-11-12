"""Tests for the small numerical/formatting helpers in src/utilities.py, and
for source-level hygiene of the modules."""

import warnings
from pathlib import Path

import numpy as np
import pytest
import tensornetwork as tn
import torch

from utilities import (
    RenyiEntropy,
    diff_entanglement_renyi,
    find_optimum_difference,
    print_ket,
    round_tensor,
)

SRC = Path(__file__).resolve().parents[1] / "src"


def _renyi_np(p, q):
    p = np.asarray(p, dtype=float)
    if q == 1:
        return float(-(p * np.log2(p)).sum())
    return float(np.log2((p**q).sum()) / (1 - q))


# --------------------------------------------------------------------------
# round_tensor
# --------------------------------------------------------------------------
def test_round_tensor_rounds_real_and_imaginary_parts():
    x = torch.tensor([[0.12345 + 0.98765j, -0.0004 + 0.5j]], dtype=torch.cfloat)
    out = round_tensor(x, 3)
    expected = torch.tensor([[0.123 + 0.988j, 0.0 + 0.5j]], dtype=torch.cfloat)
    assert torch.allclose(out, expected, atol=1e-6)


def test_round_tensor_preserves_shape_and_dtype_and_is_idempotent():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 2, dtype=torch.cfloat)
    once = round_tensor(x, 2)
    assert once.shape == x.shape and once.dtype == x.dtype
    assert torch.equal(round_tensor(once, 2), once)


def test_round_tensor_does_not_modify_its_input():
    x = torch.tensor([0.123456 + 0j], dtype=torch.cfloat)
    before = x.clone()
    round_tensor(x, 2)
    assert torch.equal(x, before)


def test_round_tensor_default_is_three_decimals():
    x = torch.tensor([0.123456 + 0j], dtype=torch.cfloat)
    assert float(round_tensor(x).real[0]) == pytest.approx(0.123, abs=1e-6)


# --------------------------------------------------------------------------
# print_ket
# --------------------------------------------------------------------------
@pytest.fixture
def two_qubit_state():
    return torch.tensor([[0.5 + 0j, 0.5 - 0.5j], [0.0 + 0j, 0.1234 + 0j]], dtype=torch.cfloat)


def test_print_ket_formats_positive_and_negative_imaginary_parts(two_qubit_state):
    out = print_ket(two_qubit_state, decimals=2, entries_per_line=1)
    lines = out.strip("\n").split("\n")
    assert len(lines) == 4
    assert lines[0] == r"(0.5 + 0.0 i )\ket{0, 0}"
    # the negative imaginary part is rendered as "- 0.5 i"
    assert lines[1].lstrip(" + ") == r"(0.5 - 0.5 i )\ket{0, 1}"
    assert lines[3].lstrip(" + ") == r"(0.12 + 0.0 i )\ket{1, 1}"


def test_print_ket_without_decimals_does_not_round(two_qubit_state):
    out = print_ket(two_qubit_state)
    assert "0.1234" in out  # float32 representation starts with 0.1234


def test_print_ket_respects_entries_per_line(two_qubit_state):
    assert print_ket(two_qubit_state, decimals=2, entries_per_line=2).count("\n") == 2
    assert print_ket(two_qubit_state, decimals=2, entries_per_line=4).count("\n") == 1


def test_print_ket_accepts_tensornetwork_nodes(two_qubit_state):
    assert print_ket(tn.Node(two_qubit_state), decimals=2) == print_ket(
        two_qubit_state, decimals=2
    )


def test_print_ket_labels_every_basis_state_of_a_four_party_state():
    out = print_ket(torch.ones(2, 2, 2, 2, dtype=torch.cfloat), decimals=1)
    assert out.count(r"\ket{") == 16
    assert r"\ket{1, 1, 1, 1}" in out and r"\ket{0, 0, 0, 0}" in out


def test_print_ket_currently_lists_zero_amplitudes(two_qubit_state):
    """Characterisation test: print_ket has a ``if r_coeff==0 and i_coeff==0:
    pass`` that looks like it was meant to skip zero terms, but it is a no-op.
    Update this test if that behaviour is changed on purpose."""
    assert r"(0.0 + 0.0 i )\ket{1, 0}" in print_ket(two_qubit_state, decimals=2)


# --------------------------------------------------------------------------
# find_optimum_difference / diff_entanglement_renyi
# --------------------------------------------------------------------------
@pytest.mark.parametrize("q", [1, 2, 3])
def test_diff_entanglement_renyi_is_ee_minus_half_the_renyi_entropy(q):
    spec = torch.tensor([0.7, 0.3], dtype=torch.float64)
    got = diff_entanglement_renyi(torch.tensor(0.2, dtype=torch.float64), spec, q)
    assert float(got) == pytest.approx(0.2 - _renyi_np([0.7, 0.3], q) / 2)


def test_diff_entanglement_renyi_defaults_to_q_equal_one():
    spec = torch.tensor([0.9, 0.1], dtype=torch.float64)
    ee = torch.tensor(1.0, dtype=torch.float64)
    assert float(diff_entanglement_renyi(ee, spec)) == pytest.approx(
        float(diff_entanglement_renyi(ee, spec, q_value=1))
    )


def test_find_optimum_difference_finds_the_most_negative_difference():
    ee = {"a": [torch.tensor(0.2), torch.tensor(0.9)], "b": [torch.tensor(0.1)]}
    spec = {
        "a": [torch.tensor([0.5, 0.5]), torch.tensor([0.9, 0.1])],
        "b": [torch.tensor([0.7, 0.3])],
    }
    result = find_optimum_difference(ee, spec, [1, 2])
    for q in (1, 2):
        candidates = [
            0.2 - _renyi_np([0.5, 0.5], q) / 2,
            0.9 - _renyi_np([0.9, 0.1], q) / 2,
            0.1 - _renyi_np([0.7, 0.3], q) / 2,
        ]
        assert float(result[q]) == pytest.approx(min(candidates), abs=1e-6)


def test_find_optimum_difference_returns_one_entry_per_renyi_index():
    ee = {"a": [torch.tensor(0.0)]}
    spec = {"a": [torch.tensor([0.5, 0.5])]}
    assert set(find_optimum_difference(ee, spec, [1, 2, 3])) == {1, 2, 3}


def test_find_optimum_difference_agrees_with_renyi_entropy_helper():
    spec = torch.tensor([0.6, 0.4], dtype=torch.float64)
    ee = {"a": [torch.tensor(0.1, dtype=torch.float64)]}
    got = find_optimum_difference(ee, {"a": [spec]}, [2])[2]
    assert float(got) == pytest.approx(0.1 - float(RenyiEntropy(spec, 2)) / 2)

