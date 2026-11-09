"""Tests for the gap function and the optimisation drivers in src/utilities.py.

The optimisers are only run for a handful of epochs on 4 qubits so the whole
module stays fast; seeds are fixed so every assertion is deterministic.
"""

import pytest
import torch

import utilities
from utilities import (
    BoundViolationFromUnitary,
    FindBoundViolation,
    FindBoundViolationLooped,
    generate_classifier_dataset,
)

pytestmark = pytest.mark.filterwarnings("ignore:.*logm input matrix.*")

DIMS = [2, 2, 2, 2]


@pytest.fixture
def random_unitary():
    torch.manual_seed(0)
    a = torch.randn(16, 16, dtype=torch.cfloat)
    q, _ = torch.linalg.qr(a)
    return q


@pytest.fixture(params=["Adam", "SGD"])
def optimizer_name(request, monkeypatch):
    """Run the test once per optimiser branch (optim_toggle is imported by
    value into utilities, so patch it there)."""
    monkeypatch.setattr(utilities, "optim_toggle", request.param)
    return request.param


# --------------------------------------------------------------------------
# BoundViolationFromUnitary
# --------------------------------------------------------------------------
@pytest.mark.parametrize("svd", [True, False])
def test_identity_unitary_gives_a_product_state_with_zero_gap(svd):
    # The default initial state is the uniform superposition, a product state.
    gap = BoundViolationFromUnitary(torch.eye(16, dtype=torch.cfloat), svd=svd)
    assert float(gap) == pytest.approx(0.0, abs=1e-5)


def test_return_state_gives_a_normalised_four_party_state(random_unitary):
    gap, state = BoundViolationFromUnitary(random_unitary, return_state=True)
    assert tuple(state.shape) == tuple(DIMS)
    assert float(torch.linalg.vector_norm(state)) == pytest.approx(1.0, abs=1e-5)
    assert gap.dim() == 0


def test_return_state_false_returns_only_the_gap(random_unitary):
    assert torch.is_tensor(BoundViolationFromUnitary(random_unitary))


def test_custom_initial_state_is_used(random_unitary):
    init = torch.zeros(16, dtype=torch.cfloat)
    init[3] = 1.0
    _, state = BoundViolationFromUnitary(
        random_unitary, initial_state=init, return_state=True
    )
    assert torch.allclose(state.reshape(16), random_unitary[:, 3], atol=1e-6)
    _, default_state = BoundViolationFromUnitary(random_unitary, return_state=True)
    assert not torch.allclose(state, default_state)


@pytest.mark.parametrize("q", [1, 2])
def test_gap_is_the_same_with_and_without_svd(random_unitary, q):
    with_svd = BoundViolationFromUnitary(random_unitary, Renyi_index=q, svd=True)
    no_svd = BoundViolationFromUnitary(random_unitary, Renyi_index=q, svd=False)
    assert float(with_svd) == pytest.approx(float(no_svd), abs=1e-4)


def test_gap_is_invariant_under_a_phase_in_the_unitary(random_unitary):
    # A global phase cannot change any entropy.
    gap = BoundViolationFromUnitary(random_unitary, svd=False)
    phased = BoundViolationFromUnitary(1j * random_unitary, svd=False)
    assert float(gap) == pytest.approx(float(phased), abs=1e-5)


@pytest.mark.parametrize("svd", [True, False])
def test_gap_is_differentiable_with_respect_to_the_generator(svd):
    torch.manual_seed(1)
    pre = torch.triu(torch.rand(16, 16, dtype=torch.cfloat) / 10, diagonal=1)
    pre.requires_grad = True
    unitary = torch.matrix_exp(pre - pre.conj().transpose(0, 1))
    BoundViolationFromUnitary(unitary, svd=svd).backward()
    assert pre.grad is not None
    assert torch.isfinite(pre.grad).all()
    assert bool((pre.grad != 0).any())


# --------------------------------------------------------------------------
# FindBoundViolation
# --------------------------------------------------------------------------
def test_find_bound_violation_returns_gap_and_normalised_state(optimizer_name):
    torch.manual_seed(0)
    gap, state = FindBoundViolation(DIMS, epochs=2, svd=False)
    assert torch.is_tensor(gap) and gap.dim() == 0
    assert tuple(state.shape) == tuple(DIMS)
    assert float(torch.linalg.vector_norm(state.detach())) == pytest.approx(1.0, abs=1e-4)


def test_find_bound_violation_runs_with_svd(optimizer_name):
    torch.manual_seed(0)
    gap, state = FindBoundViolation(DIMS, epochs=2, svd=True)
    assert torch.isfinite(gap)


def test_find_bound_violation_is_reproducible_given_a_seed():
    torch.manual_seed(7)
    g1, s1 = FindBoundViolation(DIMS, epochs=3, svd=False)
    torch.manual_seed(7)
    g2, s2 = FindBoundViolation(DIMS, epochs=3, svd=False)
    assert float(g1) == float(g2)
    assert torch.equal(s1, s2)


def test_find_bound_violation_lowers_the_gap():
    torch.manual_seed(123)
    g_start, _ = FindBoundViolation(DIMS, epochs=1, lr=1e-2, svd=False)
    torch.manual_seed(123)
    g_end, _ = FindBoundViolation(DIMS, epochs=60, lr=1e-2, svd=False)
    assert float(g_end) < float(g_start)


# --------------------------------------------------------------------------
# FindBoundViolationLooped
# --------------------------------------------------------------------------
def test_looped_search_returns_one_stack_of_states_per_q():
    torch.manual_seed(0)
    found = FindBoundViolationLooped(DIMS, num_restarts=2, qlist=[1, 2], epochs=1, svd=False)
    assert set(found) == {1, 2}
    for states in found.values():
        assert tuple(states.shape) == (2, *DIMS)
        assert not states.requires_grad


def test_looped_search_rounds_states_to_three_decimals():
    torch.manual_seed(0)
    states = FindBoundViolationLooped(DIMS, num_restarts=1, qlist=[1], epochs=1, svd=False)[1]
    for part in (states.real, states.imag):
        scaled = part * 1000
        assert torch.allclose(scaled, scaled.round(), atol=1e-2)


def test_looped_search_writes_outfile_when_requested(tmp_path):
    torch.manual_seed(0)
    outfile = tmp_path / "states.pth"
    found = FindBoundViolationLooped(
        DIMS, num_restarts=1, qlist=[1], epochs=1, outfile=str(outfile), svd=False
    )
    assert outfile.exists()
    loaded = torch.load(outfile)
    assert torch.equal(loaded[1], found[1])


def test_looped_search_writes_nothing_without_outfile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    FindBoundViolationLooped(DIMS, num_restarts=1, qlist=[1], epochs=1, svd=False)
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# generate_classifier_dataset
# --------------------------------------------------------------------------
def test_classifier_dataset_without_epsilon_minimises_the_gap(optimizer_name):
    torch.manual_seed(5)
    gap, state = generate_classifier_dataset(epochs=2)
    assert torch.isfinite(gap)
    assert tuple(state.shape) == tuple(DIMS)


def test_classifier_dataset_with_epsilon_drives_the_gap_towards_epsilon():
    epsilon = 0.3
    torch.manual_seed(5)
    g_start, _ = generate_classifier_dataset(epochs=1, lr=1e-2, epsilon=epsilon)
    torch.manual_seed(5)
    g_end, _ = generate_classifier_dataset(epochs=100, lr=1e-2, epsilon=epsilon)
    assert abs(float(g_end) - epsilon) < abs(float(g_start) - epsilon)


def test_classifier_dataset_with_epsilon_runs_on_both_optimisers(optimizer_name):
    torch.manual_seed(5)
    gap, _ = generate_classifier_dataset(epochs=2, epsilon=0.1)
    assert torch.isfinite(gap)
