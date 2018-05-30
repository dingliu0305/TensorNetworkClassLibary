# include the functions that relate to Hamiltonian's and gates
import numpy as np
from scipy.special import comb

sx = np.zeros((2, 2))
sy = np.zeros((2, 2), dtype=np.complex)
sz = sx
sx[0, 1] = 0.5
sx[1, 0] = 0.5
sy[0, 1] = 0.5*1j
sy[1, 0] = -0.5*1j
sz[0, 0] = 0.5
sz[1, 1] = -0.5


def spin_operators(spin):
    op = dict()
    if spin == 'half' or abs(spin - 0.5)<1e-10:
        op['sx'] = np.zeros((2, 2))
        op['sy'] = np.zeros((2, 2), dtype=np.complex)
        op['sz'] = np.zeros((2, 2))
        op['sx'][0, 1] = 0.5
        op['sx'][1, 0] = 0.5
        op['sy'][0, 1] = 0.5 * 1j
        op['sy'][1, 0] = -0.5 * 1j
        op['sz'][0, 0] = 0.5
        op['sz'][1, 1] = -0.5
    return op


def hamiltonian_heisenberg(jx, jy, jz, hx, hz):
    op = spin_operators(0.5)
    hamilt = jx*np.kron(op['sx'], op['sx']) + jy*np.kron(op['sy'], op['sy']).real + jz*np.kron(op['sz'], op['sz'])
    hamilt += hx*(np.kron(np.eye(2), op['sx']) + np.kron(op['sx'], np.eye(2)))
    hamilt += hz*(np.kron(np.eye(2), op['sz']) + np.kron(op['sz'], np.eye(2)))
    return hamilt


def interactions_full_connection_two_body(l):
    # return the interactions of the fully connected two-body Hamiltonian
    # interact: [first_site, second_site]
    ni = comb(l, 2)
    interact = np.zeros((int(ni), 2))
    n = 0
    for n1 in range(0, l):
        for n2 in range(n1+1, l):
            interact[n, :] = [n1, n2]
            n += 1
    return interact, ni
