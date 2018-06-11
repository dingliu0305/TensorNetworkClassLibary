from MPS_Class import MpsOpenBoundaryClass as Mob
from Parameters import parameter_dmrg
import Hamiltonian_Module as Hm
import numpy as np
import scipy.sparse.linalg as la
import time

is_debug = True


def get_bond_energies(eb_full, positions, index2):
    nl = positions.shape[0]
    nh = eb_full.size
    eb = np.zeros((nl, 1))
    for i in range(0, nh):
        p = (index2[i, 0] == positions[:, 0]) * (index2[i, 1] == positions[:, 1])
        p = np.nonzero(p)
        eb[p] += eb_full[i]
    return eb


def dmrg_finite_size(para=None):
    t_start = time.time()
    info = dict()
    print('Preparation the parameters and MPS')
    if para is None:
        para = parameter_dmrg()
    # obtain spin operators
    op_half = Hm.spin_operators(para['spin'])
    operators = [op_half['id'], op_half['sx'], op_half['sy'], op_half['sz'], op_half['su'], op_half['sd']]

    # define interaction index
    # index1[n, 1]-th operator is at the index[n, 0]-th site
    index1 = np.mat(np.arange(0, para['l']))
    index1 = np.vstack((index1, 6 * np.ones((1, para['l'])))).T.astype(int)

    # index2[n, 2]-th operator is at the index[n, 0]-th site
    # index2[n, 3]-th operator is at the index[n, 1]-th site
    if para['lattice'] == 'chain':
        para['positions_h2'] = Hm.positions_nearest_neighbor_1d(para['l'], para['bound_cond'])
    elif para['lattice'] == 'square':
        para['positions_h2'] = Hm.positions_nearest_neighbor_square(
            para['square_width'], para['square_height'], para['bound_cond'])
    index2 = Hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
    para['nh'] = index2.shape[0]  # number of two-body interactions
    # define the coefficients for one-body terms
    operators.append(-para['hx']*op_half['sx'] - para['hz']*op_half['sz'])  # the 6th operator for magnetic fields
    coeff1 = np.ones((index1.shape[0], 1))
    coeff2 = np.ones((index2.shape[0], 1))
    for i in range(0, index2.shape[0]):
        if (i % 3) == 0:
            coeff2[i, 0] = para['jxy'] / 2
            coeff2[i + 1, 0] = para['jxy'] / 2
            coeff2[i + 2, 0] = para['jz']
    # Initialize MPS
    A = Mob(length=para['l'], d=para['d'], chi=para['chi'], way='qr', ini_way='r', debug=is_debug)
    A.correct_orthogonal_center(para['ob_position'])
    print('Starting to sweep ...')
    e0_total = 0
    info['convergence'] = 1
    ob = dict()
    for t in range(0+1, para['sweep_time']+1):
        if_ob = ((t % para['dt_ob']) == 0) or t == (para['sweep_time'] - 1)
        if if_ob:
            print('In the %d-th sweep ...' % t)
        for n in range(para['ob_position']+1, para['l']):
            if para['if_print_detail']:
                print('update the %d-th tensor from left to right...' % n)
            A.correct_orthogonal_center(n)  # move the orthogonal tensor to n
            h_effect, s = A.effective_hamiltonian_dmrg(n, operators, index1, index2, coeff1, coeff2)
            A.mps[n] = la.eigs(np.eye(h_effect.shape[0])-para['tau']*h_effect,
                               k=1, which='LM', v0=A.mps[n].reshape(-1, 1).copy())[1].reshape(s)
            if para['is_real']:
                A.mps[n] = A.mps[n].real
        for n in range(para['l']-2, -1, -1):
            if para['if_print_detail']:
                print('update the %d-th tensor from right to left...' % n)
            A.correct_orthogonal_center(n)
            h_effect, s = A.effective_hamiltonian_dmrg(n, operators, index1, index2, coeff1, coeff2)
            A.mps[n] = la.eigs(np.eye(h_effect.shape[0])-para['tau']*h_effect,
                               k=1, which='LM', v0=A.mps[n].reshape(-1, 1).copy())[1].reshape(s)
            if para['is_real']:
                A.mps[n] = A.mps[n].real
        for n in range(1, para['ob_position']):
            if para['if_print_detail']:
                print('update the %d-th tensor from left to right...' % n)
            A.correct_orthogonal_center(n)
            h_effect, s = A.effective_hamiltonian_dmrg(n, operators, index1, index2, coeff1, coeff2)
            A.mps[n] = la.eigs(np.eye(h_effect.shape[0])-para['tau']*h_effect,
                               k=1, which='LM', v0=A.mps[n].reshape(-1, 1).copy())[1].reshape(s)
            if para['is_real']:
                A.mps[n] = A.mps[n].real

        if if_ob:
            ob['eb_full'] = A.observe_bond_energy(index2, coeff2, operators)
            ob['mx'] = A.observe_magnetization(operators[1])
            ob['mz'] = A.observe_magnetization(operators[3])
            ob['e_per_site'] = (sum(ob['eb_full']) - para['hx']*sum(ob['mx']) - para['hz']*sum(ob['mz']))/A.length
            info['convergence'] = abs(ob['e_per_site'] - e0_total)
            if info['convergence'] < para['break_tol']:
                print('Converged at the %d-th sweep with error = %g of energy per site.' % (t, info['convergence']))
                break
            else:
                print('Convergence error of energy per site = %g' % info['convergence'])
                e0_total = ob['e_per_site']
        if t == para['sweep_time'] - 1 and info['convergence'] > para['break_tol']:
            print('Not converged with error = %g of eb per bond' % info['convergence'])
    ob['eb'] = get_bond_energies(ob['eb_full'], para['positions_h2'], index2)
    A.calculate_entanglement_spectrum()
    A.calculate_entanglement_entropy()
    info['t_cost'] = time.time() - t_start
    print('Simulation finished in %g seconds' % info['t_cost'])
    A.report_yourself()
    # A.check_orthogonality_by_tensors(tol=1e-14)
    # time.sleep(0.05)
    print(ob['eb_full'])
    # print(index2)
    # print(coeff2)
    # print(operators[4])
    # print(operators[5])
    # print(para['jxy'])
    return ob, A, info, para
