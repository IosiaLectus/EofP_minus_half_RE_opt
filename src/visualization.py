"""Module used to do all the visualization for the project."""
import tensornetwork as tn
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import data_dir
from utilities import find_optimum_difference, diff_entanglement_renyi
from utilities import TripartiteInformation, RenyiEntropy


def generate_figpath(pic_dir, figname):
    """Generate the necessary directory and path for the generated figures."""
    if not os.path.isdir(pic_dir):
        os.mkdir(pic_dir)
    figpath = pic_dir + f"/{figname}" + ".png"

    return figpath


def plot_optimum_difference_renyi(entang_entropy_AA, refl_spec_AB,
                                  renyi_indices, figsize=(8, 5), ylim=None,
                                  pic_dir="../pics", figname="fig", dpi=600):
    """Plot at each Renyi index the optimum difference between entanglement
    entropy and half the Renyi entropy at that index."""
    plt.figure(figsize=figsize)
    plt.axhline(0.0, color='k', linestyle=':')
    for dim_key in entang_entropy_AA.keys():
        entanglement_entropy = entang_entropy_AA[dim_key]
        reflected_ent_spectra = refl_spec_AB[dim_key]
        opt_difference = find_optimum_difference(entanglement_entropy,
                                                 reflected_ent_spectra,
                                                 renyi_indices)
        plt.plot(renyi_indices, opt_difference.values(),
                 label=dim_key.replace("_",","))
    plt.grid()
    plt.legend()
    plt.xlabel('Renyi index q')
    plt.ylabel('$\delta$')
    if ylim:
        plt.ylim(ylim)
    figpath = generate_figpath(pic_dir, figname)
    plt.savefig(figpath, dpi=dpi)


def plot_optimum_difference(entanglement_entropy_AA, reflected_spectrum_AB,
                            renyi_vals, renyi_q=1, figsize=(8, 5.5), ylim=None,
                            pic_dir="../pics", figname="fig", dpi=600):
    """Plot the difference of entanglement entropy with half the
    reflected entropy over a list of q values for different Renyi
    q entropies. We find the state with that optimizes the above equation
    and use it for other q values.
    """
    plt.figure("Minimum $\delta$", figsize=figsize)
    for dim_key in entanglement_entropy_AA.keys():
        opt_diff_min = 10
        renyi_ind_opt = 0
        opt_index = 0
        for renyi_ind in entanglement_entropy_AA[dim_key].keys():
            entang_ = torch.tensor(entanglement_entropy_AA[dim_key][renyi_ind])
            spectra = reflected_spectrum_AB[dim_key][renyi_ind]
            # Calculate the reflected entropy at a specific Renyi entropy
            # with index renyi_q.
            for i_spec, spec in enumerate(spectra):
                diff_ = diff_entanglement_renyi(entang_[i_spec], spec, renyi_q)
                if diff_ < opt_diff_min:
                    opt_diff_min = diff_
                    opt_index = i_spec
                    renyi_ind_opt = renyi_ind
        opt_entan = entanglement_entropy_AA[dim_key][renyi_ind_opt][opt_index]
        opt_ref_spec = reflected_spectrum_AB[dim_key][renyi_ind_opt][opt_index]
        l_diff_min = np.zeros_like(renyi_vals)
        for i_renyi, renyi_val in enumerate(renyi_vals):
            l_diff_min[i_renyi] = diff_entanglement_renyi(opt_entan,
                                                          opt_ref_spec,
                                                          renyi_val)
        plt.plot(renyi_vals, l_diff_min, label=dim_key.replace("_",","))
    plt.grid()
    plt.legend()
    plt.xlabel('Renyi entropy index q')
    plt.ylabel('$\delta$')
    if ylim:
        plt.ylim(ylim)
    figpath = generate_figpath(pic_dir, figname + "_biggest")
    plt.savefig(figpath, dpi=dpi)


def plot_tripartite(states, entanglement_ent, reflected_ent_spectra,
                    renyi_index=1, figsize=(8, 5.5), ylim=None,
                    pic_dir="../pics", figname="fig", dpi=600):
    """Plot the tripartite information of the state."""
    plt.figure("maximum tripartite v. $\delta$", figsize=figsize)
    plt.figure("Histogram", figsize=figsize)
    l_indices = [[[a_ind], [b_ind], [c_ind]] for a_ind in range(2)
                 for b_ind in range(a_ind + 1, 3)
                 for c_ind in range(b_ind + 1, 4)]
    for dim_key in states.keys():
        tripartite = np.zeros(len(states[dim_key][f"{renyi_index}"]))
        deltas = np.zeros_like(tripartite)
        for i_state, state in enumerate(states[dim_key][f"{renyi_index}"]):
            spec = reflected_ent_spectra[dim_key][f"{renyi_index}"][i_state]
            entang_ent = entanglement_ent[dim_key][f"{renyi_index}"][i_state]
            reflected_entropy = RenyiEntropy(spec, Renyi_index=renyi_index)
            deltas[i_state] = entang_ent - reflected_entropy / 2
            tripartite_max = 0
            for indices in l_indices:
                A_indices = indices[0]
                B_indices = indices[1]
                C_indices = indices[2]
                tripartite_ = TripartiteInformation(tn.Node(state), A_indices,
                                                    B_indices, C_indices,
                                                    renyi_index)
                if tripartite_ > tripartite_max:
                    tripartite_max = tripartite_
            tripartite[i_state] = tripartite_max
        plt.figure("maximum tripartite v. $\delta$", figsize=figsize)
        plt.scatter(deltas, tripartite, label=dim_key.replace("_",","))
        plt.figure("Histogram", figsize=figsize)
        plt.hist(tripartite, label=dim_key.replace("_",","), density=True)
    plt.axvline(x=0, c="black")
    plt.legend()
    plt.xlabel("Frequency")
    plt.ylabel("Maximum Tripartite Mutual Information")
    if ylim:
        plt.ylim(ylim)
    figpath = generate_figpath(pic_dir, figname + "_hist")
    plt.savefig(figpath, dpi=dpi)

    plt.figure("maximum tripartite v. $\delta$", figsize=figsize)
    plt.axhline(y=0, c="black")
    plt.legend()
    plt.xlabel("$\delta$")
    plt.ylabel("Maximum Tripartite Mutual Information")
    if ylim:
        plt.ylim(ylim)
    figpath = generate_figpath(pic_dir, figname + "_delta")
    plt.savefig(figpath, dpi=dpi)
    
