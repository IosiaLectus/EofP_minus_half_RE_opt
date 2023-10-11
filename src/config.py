import torch


data_dir = "../data"
optim_toggle = "Adam"
pauli_x = torch.tensor([[0 + 0j, 1 + 0j], [1 + 0j, 0 + 0j]],
                       dtype=torch.cfloat)
pauli_y = torch.tensor([[0 + 0j, 0 - 1j], [0 + 1j, 0 + 0j]],
                       dtype=torch.cfloat)
pauli_z = torch.tensor([[1 + 0j, 0 + 0j], [0 + 0j, -1 + 0j]],
                       dtype=torch.cfloat)
id_2d = torch.tensor([[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]], dtype=torch.cfloat)