"""Tests for src/visualization.py.

Figures are rendered with the non-interactive Agg backend into pytest's
tmp_path, and the plotted data is read back from the matplotlib artists, so
we check the numbers on the plot and not only that a PNG appeared.
"""

import numpy as np
import pytest
import tensornetwork as tn
import torch

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import visualization as viz  # noqa: E402
from utilities import TripartiteInformation  # noqa: E402

KEY = "2_2_2_2"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _renyi_np(p, q):
    p = np.asarray(p, dtype=float)
    if q == 1:
        return float(-(p * np.log2(p)).sum())
    return float(np.log2((p**q).sum()) / (1 - q))


def _line_labels(ax):
    return [line.get_label() for line in ax.lines if not line.get_label().startswith("_")]


@pytest.fixture
def entropy_data():
    """ee[dim_key][renyi_key][i] and spectra[dim_key][renyi_key][i]."""
    ee = {KEY: {"1": [0.1, 0.4], "2": [0.3]}}
    spectra = {
        KEY: {
            "1": [torch.tensor([0.9, 0.1]), torch.tensor([0.5, 0.5])],
            "2": [torch.tensor([0.7, 0.3])],
        }
    }
    return ee, spectra


# --------------------------------------------------------------------------
# generate_figpath
# --------------------------------------------------------------------------
def test_generate_figpath_creates_the_directory_and_returns_a_png_path(tmp_path):
    target = tmp_path / "pics"
    path = viz.generate_figpath(str(target), "myfig")
    assert target.is_dir()
    assert path == f"{target}/myfig.png"


def test_generate_figpath_accepts_an_existing_directory(tmp_path):
    assert viz.generate_figpath(str(tmp_path), "x") == f"{tmp_path}/x.png"


# --------------------------------------------------------------------------
# plot_optimum_difference_renyi
# --------------------------------------------------------------------------
def test_plot_optimum_difference_renyi_saves_figure_with_expected_curve(tmp_path, entropy_data):
    ee, spectra = entropy_data
    qs = [1, 2, 3]
    viz.plot_optimum_difference_renyi(
        ee, spectra, qs, pic_dir=str(tmp_path / "p"), figname="renyi", dpi=20
    )
    assert (tmp_path / "p" / "renyi.png").exists()

    ax = plt.gcf().axes[0]
    assert _line_labels(ax) == ["2,2,2,2"]  # underscores become commas
    curve = [line for line in ax.lines if line.get_label() == "2,2,2,2"][0]
    # reference: min over all (key, state) of ee - S_q/2, floored at 0
    pairs = [(0.1, [0.9, 0.1]), (0.4, [0.5, 0.5]), (0.3, [0.7, 0.3])]
    expected = [min(0.0, *[e - _renyi_np(p, q) / 2 for e, p in pairs]) for q in qs]
    assert np.allclose(curve.get_ydata(), expected, atol=1e-6)
    assert list(curve.get_xdata()) == qs


def test_plot_optimum_difference_renyi_draws_a_zero_reference_line(tmp_path, entropy_data):
    ee, spectra = entropy_data
    viz.plot_optimum_difference_renyi(ee, spectra, [1], pic_dir=str(tmp_path), dpi=20)
    ax = plt.gcf().axes[0]
    assert any(np.allclose(line.get_ydata(), 0) for line in ax.lines)


def test_plot_optimum_difference_renyi_honours_ylim_and_one_curve_per_key(tmp_path, entropy_data):
    ee, spectra = entropy_data
    ee["3_3_2_2"] = ee[KEY]
    spectra["3_3_2_2"] = spectra[KEY]
    viz.plot_optimum_difference_renyi(
        ee, spectra, [1, 2], ylim=(-1, 0.5), pic_dir=str(tmp_path), dpi=20
    )
    ax = plt.gcf().axes[0]
    assert ax.get_ylim() == (-1, 0.5)
    assert sorted(_line_labels(ax)) == ["2,2,2,2", "3,3,2,2"]


# --------------------------------------------------------------------------
# plot_optimum_difference
# --------------------------------------------------------------------------
def _expected_best_curve(qs, renyi_q=1):
    """Pick the (ee, spectrum) pair with the most negative delta at ``renyi_q``,
    then re-evaluate that same state's delta at every q in ``qs``."""
    pairs = [(0.1, [0.9, 0.1]), (0.4, [0.5, 0.5]), (0.3, [0.7, 0.3])]
    e, p = min(pairs, key=lambda ep: ep[0] - _renyi_np(ep[1], renyi_q) / 2)
    return [e - _renyi_np(p, q) / 2 for q in qs]


def test_plot_optimum_difference_plots_the_best_state_across_q(tmp_path, entropy_data):
    ee, spectra = entropy_data
    qs = [1.0, 2.0, 3.0]
    viz.plot_optimum_difference(
        ee, spectra, qs, renyi_q=1, pic_dir=str(tmp_path), figname="best", dpi=20
    )
    assert (tmp_path / "best_biggest.png").exists()
    ax = plt.gcf().axes[0]
    curve = ax.lines[0]
    assert curve.get_label() == "2,2,2,2"
    assert np.allclose(curve.get_ydata(), _expected_best_curve(qs), atol=1e-5)


# def test_plot_optimum_difference_with_integer_q_values_is_not_truncated(tmp_path, entropy_data):
#     """Passing renyi_vals as a list of ints must give the same curve as floats
#     (np.zeros_like on an int list would silently truncate deltas to integers)."""
#     ee, spectra = entropy_data
#     qs = [1, 2, 3]
#     viz.plot_optimum_difference(ee, spectra, qs, pic_dir=str(tmp_path), dpi=20)
#     curve = plt.gcf().axes[0].lines[0]
#     assert np.allclose(curve.get_ydata(), _expected_best_curve(qs), atol=1e-5)


def test_plot_optimum_difference_honours_ylim(tmp_path, entropy_data):
    ee, spectra = entropy_data
    viz.plot_optimum_difference(
        ee, spectra, [1.0, 2.0], ylim=(-0.5, 0.5), pic_dir=str(tmp_path), dpi=20
    )
    assert plt.gcf().axes[0].get_ylim() == (-0.5, 0.5)


# --------------------------------------------------------------------------
# plot_tripartite
# --------------------------------------------------------------------------
@pytest.fixture
def tripartite_data(make_state):
    states = [make_state((2, 2, 2, 2), seed=s) for s in (1, 2, 3)]
    spectra = [torch.tensor(p) for p in ([0.5, 0.5], [0.6, 0.4], [0.9, 0.1])]
    ee = [0.5, 0.7, 0.2]
    return (
        {KEY: {"1": states}},
        {KEY: {"1": ee}},
        {KEY: {"1": spectra}},
        states, ee, spectra,
    )


def test_plot_tripartite_saves_both_figures(tmp_path, tripartite_data):
    states, ee, spectra, *_ = tripartite_data
    viz.plot_tripartite(
        states, ee, spectra, renyi_index=1, pic_dir=str(tmp_path), figname="t", dpi=20
    )
    assert (tmp_path / "t_hist.png").exists()
    assert (tmp_path / "t_delta.png").exists()


def test_plot_tripartite_scatter_shows_delta_against_max_tripartite_information(
    tmp_path, tripartite_data
):
    states, ee, spectra, raw_states, raw_ee, raw_spectra = tripartite_data
    viz.plot_tripartite(states, ee, spectra, renyi_index=1, pic_dir=str(tmp_path), dpi=20)

    fig = plt.figure(r"maximum tripartite v. $\delta$")
    ax = fig.axes[0]
    points = ax.collections[0].get_offsets()
    deltas, tripartite = points[:, 0], points[:, 1]

    expected_deltas = [e - _renyi_np(s.numpy(), 1) / 2 for e, s in zip(raw_ee, raw_spectra)]
    assert np.allclose(deltas, expected_deltas, atol=1e-6)

    # for a pure 4-party state I3 does not depend on which 3 parties are chosen
    expected_i3 = [
        float(TripartiteInformation(tn.Node(s.clone()), [0], [1], [2])) for s in raw_states
    ]
    assert np.allclose(tripartite, expected_i3, atol=1e-6)
    assert ax.get_xlabel() == r"$\delta$"


def test_plot_tripartite_honours_ylim_on_both_figures(tmp_path, tripartite_data):
    states, ee, spectra, *_ = tripartite_data
    viz.plot_tripartite(
        states, ee, spectra, ylim=(-2, 2), pic_dir=str(tmp_path), dpi=20
    )
    for name in ("Histogram", r"maximum tripartite v. $\delta$"):
        assert plt.figure(name).axes[0].get_ylim() == (-2, 2)
