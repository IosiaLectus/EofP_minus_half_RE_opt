#!/usr/bin/python3

import torch
import tensornetwork as tn
import numpy as np
from copy import deepcopy
from functools import partial
from tqdm import tqdm, trange
from config import optim_toggle
from itertools import product
from matrix_log import logm #import matrix log with autograd


tn.set_default_backend("pytorch")
torch.autograd.set_detect_anomaly(True)


def partial_trace(t, traced_out_parties):
    """
    Writing a function to compute partial trace. This function takes
    for argument a torch tensor t, and a list of traced out parties, in
    the form of a zero-indexed list. For example, if we have a density
    matrix on 2 qubits, and we want to trace out the second qubit, then
    traced_out_parties takes the value [1].
    """
    rank = len(list(t.size()))
    einsum_string = ""
    for x in range(rank):
        einsum_string += chr(x + ord('A'))
    for party in traced_out_parties:
        einsum_string = einsum_string.replace(einsum_string[party], einsum_string[(party + (rank//2)) % rank])
    
    return torch.einsum(einsum_string, t)


def HLS_state(beta, dim_list=(3, 3, 2, 3, 3, 2)):
    """The family of states considered in the arXiv paper 2302.10208."""
    density_mat = torch.zeros(dim_list)
    density_mat[0, 0, 0, 0, 0, 0] = beta
    density_mat[1, 1, 0, 1, 1, 0] = beta
    density_mat[2, 0, 0, 2, 0, 0] = beta
    density_mat[2, 1, 0, 2, 1, 0] = beta 
    density_mat[0, 2, 0, 0, 2, 0] = 1.0
    density_mat[1, 2, 1, 1, 2, 1] = 1.0
    # normalize
    norm_factor = (4 * beta + 2) * (1.0 + 0.0j)
    density_mat = density_mat / norm_factor

    return tn.Node(density_mat)


def reduced_HLS_state(beta):
    reduced_HLS_state = partial_trace(HLS_state(beta).tensor, [2])
    return tn.Node(reduced_HLS_state)


def HLS_reflected_spectrum(beta):
    ev1 = (1+beta)/(4*beta + 2)
    ev2 = (1 + 3*beta + np.sqrt(1 - 2*beta + 9*beta*beta))/(8*beta + 4)
    ev3 = (1 + 3*beta - np.sqrt(1 - 2*beta + 9*beta*beta))/(8*beta + 4)
    spec = torch.Tensor(sorted([ev1, ev2, ev3], reverse=True))
    return spec


def HLS_reduced_reflected_spectrum(beta):
    ev1 = (beta)/(4*beta + 2)
    ev2 = (2 + 3*beta + np.sqrt(4 - 4*beta + 9*beta*beta))/(8*beta + 4)
    ev3 = (2 + 3*beta - np.sqrt(4 - 4*beta + 9*beta*beta))/(8*beta + 4)
    spec = torch.Tensor(sorted([ev1, ev2, ev3], reverse=True))
    return spec


# Do SVD (= eigenvalue decomp?) on density matrix
def split_density_mat(density_mat):
    dims = list(density_mat.tensor.size())
    nindices = len(dims)
    if nindices % 2 == 1:
        print("invalid density matrix: odd number of indices")
        return
    nparties = int(nindices / 2)
    if not dims[:nparties] == dims[nparties:]:
        print("invalid density matrix: dimensions unbalanced")
        return
    left_indices = [density_mat[i] for i in range(nparties)]
    right_indices = [density_mat[i + nparties] for i in range(nparties)]
    # Do a SVD
    left, mid, right, sing_vals = tn.split_node_full_svd(density_mat,
                                                         left_indices,
                                                         right_indices)
    return left, mid, right


# Returns the pieces needed to build purifications
def MinimalPurification(density_mat, reg=10**-12):
    left, mid, right = split_density_mat(density_mat)
    new_mid = tn.Node(torch.sqrt(mid.tensor + reg*torch.ones_like(mid.tensor)))
    new_left = tn.Node(left.tensor)
    return new_left, new_mid


def BuildPurification(left, mid, right):
    # create the edges
    left[-1] ^ mid[0]
    mid[1] ^ right[-1]
    # contract the edges
    purification = left @ mid
    purification = purification @ right
    return purification


# Get the spectrum of the reduced density matrix of a pure state on certain indices
def EntanglementSpectrum(state, A_indices):
    dims = list(state.tensor.size())
    nparties = len(dims)
    # TODO: Should it be if i not in A_indices?
    B_indices = [i for i in range(nparties) if not i in A_indices]
    left, mid, right, sing_vals = tn.split_node_full_svd(state,
                                                         [state[i] for i in
                                                          A_indices],
                                                         [state[i] for i in
                                                          B_indices])
    # From SVD read off the singular values and hence the entropy
    sing_vals = torch.einsum('ii->i', mid.tensor)
    probs = torch.conj(sing_vals) * sing_vals
    # Force spectrum to be real and normalized
    probs = probs.real
    probs = probs / torch.sum(probs)
    return probs


# Shannon Entropy of a classical probability distribution
def ShannonEntropy(probs):
    log_probs = torch.log(probs)
    log_probs = torch.nan_to_num(log_probs, nan=0.0, neginf=0.0, posinf=0.0)
    h = -(1 / np.log(2)) * torch.sum(probs * log_probs)
    if torch.isnan(h):
        print(f"Problematic probs are {probs}\n")
    return h


# Renyi entropy of a classical probability distribujtion
def RenyiEntropy(probs, Renyi_index=1):
    if Renyi_index == 1:
        return ShannonEntropy(probs)
    x = torch.sum(probs**Renyi_index)
    x = torch.log(x)/np.log(2)
    x = x/(1 - Renyi_index)
    return x

# Compute entanglement entropy of system defined by A_indices without SVD
def EntanglementEntropyNoSVD(state, A_indices, Renyi_index=1):
    dims = list(state.tensor.size())
    nparties = len(dims)
    B_indices = [i for i in range(nparties) if not i in A_indices]
    A_dim = np.product([dims[i] for i in A_indices])
    B_dim = np.product([dims[i] for i in B_indices])
    tot_dim = A_dim*B_dim 
    state_tensor = state.tensor
    state_tensor = state_tensor.reshape([tot_dim])
    rho = torch.einsum('i,j->ij', state_tensor.conj(), state_tensor)
    rho = rho.reshape(dims+dims)
    rho = partial_trace(rho, B_indices)
    rho = rho.reshape([A_dim, A_dim])
    if Renyi_index==1:
        log_rho = logm(rho).type(rho.type())
        rho_log_rho = torch.einsum('ik,kj->ij', rho, log_rho)
        ret = - torch.einsum('ii->',rho_log_rho)
    else:
        if Renyi_index%1==0:
            rho_to_alpha = torch.matrix_power(rho, int(Renyi_index))
        else:
            log_rho = logm(rho).type(rho.type())
            rho_to_alpha = torch.linalg.matrix_exp(Renyi_index*log_rho)
        ret =  torch.log(torch.einsum('ii->',rho_to_alpha))/(1 - Renyi_index)
    ret = ret/np.log(2)
    return torch.real(ret)

def test_EntanglementEntropyNoSVD(q=1):
    state = torch.rand(2,2,4,4)
    state_norm = torch.sqrt((state**2).sum())
    state = state/state_norm
    state = (1+0j)*state
    state = tn.Node(state)
    ee_no_SVD = float(EntanglementEntropyNoSVD(state, [0,1], Renyi_index=q))
    ee_SVD = float(EntanglementEntropy(state, [0,1], Renyi_index=q))
    return ee_SVD, ee_no_SVD



# Compute entanglement entropy of system defined by A_indices
def EntanglementEntropy(state, A_indices, Renyi_index=1, svd=True):
    if not svd:
        return EntanglementEntropyNoSVD(state, A_indices, Renyi_index)
    probs = EntanglementSpectrum(state, A_indices)
    return RenyiEntropy(probs, Renyi_index)


def DensityMatrixSpectrum(density_mat):
    left, mid, right = split_density_mat(density_mat)
    probs = torch.einsum('ii->i', mid.tensor)
    # Force spectrum to be real and normalized
    probs = probs.real
    probs = probs / torch.sum(probs)
    return probs


# Compute Von Neuman entropy
def VNEntropy(density_mat, Renyi_index=1):
    probs = DensityMatrixSpectrum(density_mat)
    return RenyiEntropy(probs, Renyi_index=1)


# Computed the reflected entropy spectrum (I think?)
def ReflectedEntropySpectrum(density_mat, A_indices):
    # get relevant part of SVD
    left, mid = MinimalPurification(density_mat)
    dims = list(left.tensor.size())
    nparties = len(dims) - 1
    right = tn.Node(left.tensor.clone().conj())
    purification = BuildPurification(left, mid, right)
    # Get correct indices for SVD
    top_indices = [i for i in A_indices] + [i + nparties for i in A_indices]

    return EntanglementSpectrum(purification, top_indices), purification

# Compute the reflected entropy without doing an SVD
def ReflectedEntropyNoSVD(density_mat, A_indices, Renyi_index=1, svd=True):
    dims = list(density_mat.tensor.size())
    nparties = len(dims)//2
    tot_dim = np.product([dims[i] for i in range(nparties)])
    rho = density_mat.tensor.reshape([tot_dim, tot_dim])
    log_rho = logm(rho).type(rho.type())
    purification = torch.matrix_exp(0.5*log_rho)
    purification = purification.reshape(dims)
    purification = tn.Node(purification)
    top_indices = [i for i in A_indices] + [i + nparties for i in A_indices]
    return EntanglementEntropyNoSVD(purification, top_indices, Renyi_index), purification


# Computed the reflected entropy (I think?)
def ReflectedEntropy(density_mat, A_indices, Renyi_index=1, svd=True):
    if not svd:
        return ReflectedEntropyNoSVD(density_mat, A_indices, Renyi_index)
    else:
        probs, purification = ReflectedEntropySpectrum(density_mat, A_indices)
        return RenyiEntropy(probs, Renyi_index), purification


def TripartiteInformation(state, A_indices, B_indices, C_indices,
                          Renyi_index=1):
    """Calculate tripartite information."""    
    ee_A   = EntanglementEntropy(state, A_indices, Renyi_index)
    ee_B   = EntanglementEntropy(state, B_indices, Renyi_index)
    ee_C   = EntanglementEntropy(state, B_indices, Renyi_index)
    ee_AB  = EntanglementEntropy(state, A_indices + B_indices, Renyi_index)
    ee_AC  = EntanglementEntropy(state, A_indices + C_indices, Renyi_index)
    ee_BC  = EntanglementEntropy(state, B_indices + C_indices, Renyi_index)
    ee_ABC = EntanglementEntropy(state, A_indices + B_indices + C_indices,
                                 Renyi_index)
    tpi = ee_A + ee_B + ee_C - ee_AB - ee_AC - ee_BC + ee_ABC
    
    return tpi


def BoundViolationFromUnitary(unitary, subsystem_dims = [2, 2, 2, 2],
                              Renyi_index=1, svd=True, initial_state=None,
                              return_state=False):
    tot_dims = np.prod(subsystem_dims)
    if not torch.is_tensor(initial_state):
        initial_state = (1.+0.j) * torch.ones(tot_dims)
        initial_state = initial_state / np.sqrt(initial_state @ initial_state.conj())
    state = torch.einsum('ij,j->i', unitary, initial_state)
    state = state.reshape(tuple(subsystem_dims))
    ee = EntanglementEntropy(tn.Node(state), [0, 2], svd=svd)
    rhoAB = torch.einsum('ijab, klab -> ijkl', state, state.conj())
    reflected_entropy, reflected_state = ReflectedEntropy(tn.Node(rhoAB), [0], Renyi_index=Renyi_index, svd=svd)
    gap = ee - (reflected_entropy/2)
    if return_state:
        return gap, state
    return gap


def FindBoundViolation(subsystem_dims = [2, 2, 2, 2], Renyi_index=1,
                       initial_state=None, svd=True, epochs=2000, lr=1e-3):
    # subsystem_dims gives the dimensions of the 4 subsystems (as a list) in the order A, B, A', B'
    # Build the unitary
    # This is actually what will be optimized
    tot_dim = np.prod(subsystem_dims)
    pre_hermitian = torch.triu(torch.rand(tot_dim, tot_dim,
                                          dtype=torch.cfloat) / 10, diagonal=1)
    pre_hermitian.requires_grad = True

    gap_func = partial(BoundViolationFromUnitary,
                       subsystem_dims=subsystem_dims, Renyi_index=Renyi_index,
                       initial_state=initial_state, svd=svd)

    gap_list = []
    # pytorch optimizer
    if optim_toggle == "SGD":
        optimizer = torch.optim.SGD([pre_hermitian], lr=lr, momentum=0.5)
    elif optim_toggle == "Adam":
        optimizer = torch.optim.Adam([pre_hermitian], lr=lr)
    for i in trange(epochs):
        optimizer.zero_grad()
        # From this random matrix we will construct an anti-hermitian matrix
        anti_hermitian = pre_hermitian - pre_hermitian.conj().transpose(0,1)
        # and we will exponentiate the anti-hermitian matrix to get the unitary matrix
        unitary = torch.matrix_exp(anti_hermitian)
        gap = gap_func(unitary)
        gap_list.append(gap)
        #if i % 50 == 0 or i<50:
        #    print(f"\nEE at step {i}: {ee}")
        #    print(pre_hermitian[0,1])
        gap.backward(retain_graph=True)
        optimizer.step()

    return BoundViolationFromUnitary(unitary, subsystem_dims=subsystem_dims,
                                     Renyi_index=Renyi_index,
                                     return_state=True)


def FindBoundViolationLooped(subsystem_dims=[2, 2, 2, 2], num_restarts=10,
                             qlist=[1], outfile=None, epochs=2000, svd=True):
    """
    Do FindBoundViolation with num_restarts restarts for each q in
    qlist.
    """
    found_states = {q: [] for q in qlist}
    for q in qlist:
        for i in range(num_restarts):
            gap, state = FindBoundViolation(subsystem_dims=subsystem_dims,
                                            epochs=epochs, Renyi_index=q,
                                            svd=svd)
            state = state.detach()
            state = round_tensor(state, 3)
            found_states[q].append(state.unsqueeze(0))

    found_states = {q: torch.cat(found_states[q], 0) for q in qlist}
    if outfile:
        torch.save(found_states, outfile)
    return found_states


def round_tensor(tns, decimals=3):
    """
    Define a function to round a for real (unlike torch.round, which
    doesn't change the tensor but only how it is displayed).
    """
    out = torch.zeros_like(tns)
    dims = list(tns.shape)
    tuples_of_indices = [ [i] for i in range(dims[0])]
    for n in range(1,len(dims)):
        tuples_of_indices = [x + [i] for x in tuples_of_indices for i in range(dims[n]) ]
    tuples_of_indices = [tuple(x) for x in tuples_of_indices]
    for x in tuples_of_indices:
        entry_real = float(tns[x].real)
        entry_imag = float(tns[x].imag)
        entry_real = round(entry_real, decimals)
        entry_imag = round(entry_imag, decimals)
        out[x] = (1+0j)*entry_real + (0+1j)*entry_imag
    return out


def print_ket(state, decimals=None, entries_per_line=2):
    if not torch.is_tensor(state):
        state = state.tensor
    dims = list(state.shape)
    tuples_of_indices = [ [i] for i in range(dims[0])]
    for n in range(1,len(dims)):
        tuples_of_indices = [x + [i] for x in tuples_of_indices for i in range(dims[n]) ]
    tuples_of_indices = [tuple(x) for x in tuples_of_indices]
    latex = ''
    entry_count = 0
    for x in tuples_of_indices:
        entry_count += 1
        inside_ket = f'{x[0]}'
        for n in range(1,len(x)):
            inside_ket += f', {x[n]}'
        ket = '\ket{' + inside_ket + '}'
        coeff = state[x]
        r_coeff = float(coeff.real)
        i_coeff = float(coeff.imag)
        if not decimals is None:
            r_coeff = round(r_coeff, decimals)
            i_coeff = round(i_coeff, decimals)
        if r_coeff==0 and i_coeff==0:
            pass
        #coeff = 1.0+3.0j
        if i_coeff<0:
            latex += f' + ({r_coeff} - {-i_coeff} i ){ket}'
        else:
            latex += f' + ({r_coeff} + {i_coeff} i ){ket}'
        if entry_count%entries_per_line==0:
            latex += '\n'
    latex = latex[3:]
    return latex


def find_optimum_difference(entanglement_entropy, reflected_ent_spectra,
                            renyi_indices):
    """Identify the state for which we have the optimum difference between
    entanglement entropy and half the reflected entropy. This is done over
    the entire range of the provided Renyi indices.
    """
    optimum_difference = {}
    for renyi_index in renyi_indices:
        reflected_entropy = np.zeros_like(reflected_ent_spectra)
        opt_diff = 0
        for renyi_key in reflected_ent_spectra.keys():
            for i_spec, spec in enumerate(reflected_ent_spectra[renyi_key]):
                reflected_entropy = RenyiEntropy(spec, Renyi_index=renyi_index)
                entang_entropy = entanglement_entropy[renyi_key][i_spec]
                diff_ = entang_entropy - reflected_entropy / 2
                if diff_.min() < opt_diff:
                    opt_diff = diff_.min()
        optimum_difference[renyi_index] = opt_diff
    
    return optimum_difference


def diff_entanglement_renyi(entanglement_entropy, reflected_ent_spectrum,
                            q_value=1):
    """Calculate the difference of the optimal entanglement entropy."""
    half_renyi = RenyiEntropy(reflected_ent_spectrum, Renyi_index=q_value) / 2
    
    return entanglement_entropy - half_renyi


def generate_classifier_dataset(subsystem_dims = [2, 2, 2, 2], Renyi_index=1,
                                initial_state=None, epochs=2001, lr=1e-3,
                                epsilon=None):
    """Generate a dataset for making a classifier """
    tot_dim = np.prod(subsystem_dims)
    pre_hermitian = torch.triu(1e-1 * torch.rand(tot_dim, tot_dim,
                                                 dtype=torch.cfloat),
                                                 diagonal=1)
    pre_hermitian.requires_grad = True

    gap_func = partial(BoundViolationFromUnitary,
                       subsystem_dims=subsystem_dims, Renyi_index=Renyi_index,
                       initial_state=initial_state, return_state=True)
    # pytorch optimizer
    if optim_toggle == "SGD":
        optimizer = torch.optim.SGD([pre_hermitian], lr=lr, momentum=0.5)
    elif optim_toggle == "Adam":
        optimizer = torch.optim.Adam([pre_hermitian], lr=lr)
    # TODO: Potential removal.
    # l_gap = torch.zeros(epochs)
    state_shape = tuple([epochs] + subsystem_dims)
    # TODO: Potential removal.
    # l_states = torch.zeros(state_shape, dtype=torch.complex128)
    for epoch in trange(epochs):
        optimizer.zero_grad()
        # From this random matrix we will construct an anti-hermitian
        # matrix.
        anti_hermitian = pre_hermitian - pre_hermitian.conj().transpose(0,1)
        # and we will exponentiate the anti-hermitian matrix to get the
        # unitary matrix
        unitary = torch.matrix_exp(anti_hermitian)
        gap = gap_func(unitary)
        # TODO: Potential removal.
        # l_gap[epoch] = gap[0]
        # l_states[epoch] = gap[1]
        # Optimize the gap to be equal with a certain epsilon. If epsilon
        # is None then the optimizer tries to find the minimum gap, which means
        # make it a negative number with a large absolute value. 
        if epsilon:
            delta_opt = torch.abs(gap[0] - epsilon)
            delta_opt.backward(retain_graph=True)
        else:
            gap[0].backward(retain_graph=True)
        optimizer.step()

    return gap[0], gap[1]


