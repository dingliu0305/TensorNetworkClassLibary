from MPS_Class import MpsOpenBoundaryClass as Mob
import Hamiltonian_Module as Hm
import numpy as np


print('Preparation the parameters and MPS')
para = dict()
# Physical parameters
para['jx'] = 1
para['jy'] = 1
para['jz'] = 1
para['hx'] = 0
para['hz'] = 0
# Numerical parameters
para['l'] = 10  # Length of MPS
para['chi'] = 8  # Virtual bond dimension cut-off
para['d'] = 2  # Physical bond dimension
para['sweep_time'] = 20  # sweep time
# Fixed parameters
para['tau'] = 1e-4  # a shift to ensure the GS energy is the lowest
para['break_tol'] = 1e-10  # tolerance for breaking the loop

# Initialize MPS
A = Mob(para['l'], para['d'], para['chi'])
A.orthogonalize_mps(para['l'],0)

# Prepare Hamiltonian
hamilt = Hm.hamiltonian_heisenberg(para['jx'], para['jy'], para['jz'], para['hx'], para['hz'])
interact_index, nh = Hm.interactions_full_connection_two_body(para['l'])

print('Starting sweep')
for t in range(0, para['sweep_time']):
    # t-th sweep
    for n in range(0, para['l']):
        # update the n-th tensor
        pass


