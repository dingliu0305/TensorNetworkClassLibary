from MPS_Class import MpsOpenBoundaryClass as Mob
import Hamiltonian_Module as Hm
import numpy as np
import scipy.sparse.linalg as la


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
# obtain spin operators
op_half = Hm.spin_operators('half')
operators = [op_half['id'], op_half['sx'], op_half['sy'], op_half['sz'], op_half['su'], op_half['sd']]
para['nh'] = para['l'] - 1  # number of two-body interactions
# define interaction index
# index1[n, 1]-th operator is at the index[n, 0]-th site
index1 = np.mat(np.arange(0, para['l']))
index1 = np.vstack((index1, 6 * np.ones((1, para['l'])))).T.astype(int)
# index2[n, 2]-th operator is at the index[n, 0]-th site
# index2[n, 3]-th operator is at the index[n, 1]-th site
index_position = Hm.interactions_nearest_neighbor_1d(para['l'])
index2 = Hm.interactions_position2full_index_heisenberg_two_body(index_position)
# define the coefficients for one-body terms
operators.append(-para['hx']*op_half['sx'] - para['hz']*op_half['sz'])  # the 6th operator for field
coeff1 = np.ones((para['l'], 1))
coeff2 = np.ones((index2.shape[0], 1))

# Initialize MPS
A = Mob(length=para['l'], d=para['d'], chi=para['chi'], debug=1)
A.orthogonalize_mps(para['l']-1, 0)

# Prepare Hamiltonian
hamilt = Hm.hamiltonian_heisenberg(para['jx'], para['jy'], para['jz'], para['hx'], para['hz'])
interact_index, nh = Hm.interactions_full_connection_two_body(para['l'])

n = 0
A.mps[n] = A.update_tensor_handle_dmrg_1site(A.mps[n], n, operators, index1, index2, coeff1, coeff2, para['tau'])



# print('Starting sweep')
# for t in range(0, para['sweep_time']):
#     # t-th sweep
#     for n in range(0, para['l']):
        # update the n-th tensor

        # fun = la.aslinearoperator(lambda tensor: A.update_tensor_handle_dmrg_1site(tensor, n, operators, index1, index2, coeff1, coeff2, para['tau']))
        # tmp, A.mps[n] = la.eigs(A=fun, k=1, which='LM')
