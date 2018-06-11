import Tensor_Basic_Module as T_module
import numpy as np
from Basic_Functions_SJR import empty_list, trace_stack, sort_list, print_error
from termcolor import colored, cprint


class MpsOpenBoundaryClass:

    def __init__(self, length, d, chi, way='svd', ini_way='r', debug=0):
        self.phys_dim = d
        self.decomp_way = way  # 'svd' or 'qr'
        self.length = length
        # self.orthogonality:  -1: left2right; 0: not orthogonal or center; 1: right2left
        self.orthogonality = np.zeros((length, 1))
        self.center = -1  # orthogonal center; -1 means no center
        self.lm = empty_list(length-1, np.zeros(0))
        self.ent = np.zeros((self.length-1, 1))
        if ini_way == 'r':  # randomly initialize MPS
            self.mps = T_module.random_open_mps(length, d, chi)
        elif ini_way == '1':  # initialize MPS as eyes
            self.mps = T_module.ones_open_mps(length, d, chi)
        self.virtual_dim = np.ones((length + 1,)).astype(int) * chi
        self.virtual_dim[0] = 1
        self.virtual_dim[-1] = 1
        self._debug = debug  # if in debug mode

    def report_yourself(self):
        print('center: ' + str(self.center))
        print('orthogonality:' + str(self.orthogonality.T))
        print('virtual bond dimensions: ' + str(self.virtual_dim))
        for n in range(0, self.length-1):
            print('lm[%d] = ' % n + str(self.lm[n]))
        for n in range(0, self.length-1):
            print('ent[%d] = ' % n + str(self.ent[n]))

    # Orthogonalize the MPS from the l0-th to l1-th site (l0<l1)
    def orthogonalize_mps(self, l0, l1):
        if l0 < l1:  # Orthogonalize MPS from left to rigth
            for n in range(l0, l1):
                self.mps[n], mat, self.virtual_dim[n+1], lm = \
                    T_module.left2right_decompose_tensor(self.mps[n], self.decomp_way)
                if lm.size > 0 and self.center > -1:
                    self.lm[n] = lm.copy()
                self.mps[n+1] = T_module.absorb_matrix2tensor(self.mps[n + 1], mat, 0)
            self.orthogonality[l0:l1] = -1
            self.orthogonality[l1] = 0
        elif l0 > l1:  # Orthogonalize MPS from right to left
            for n in range(l0, l1, -1):
                self.mps[n], mat, self.virtual_dim[n], lm =\
                    T_module.right2left_decompose_tensor(self.mps[n], self.decomp_way)
                if lm.size > 0 and self.center > -1:
                    self.lm[n-1] = lm.copy()
                self.mps[n-1] = T_module.absorb_matrix2tensor(self.mps[n - 1], mat, 2)
            self.orthogonality[l0:l1:-1] = 1
            self.orthogonality[l1] = 0

    # transfer the MPS into the central orthogonal form with the center lc
    def central_orthogonalization(self, lc, l0=0, l1=-1):
        # NOTE: recommend to use correct_orthogonal_center in the code
        if l1 == -1:
            l1 = self.length-1
        self.orthogonalize_mps(l0, lc)
        self.orthogonalize_mps(l1, lc)

    # move the orthogonal center at p
    def correct_orthogonal_center(self, p=-1):
        # if p<0 (default) and there is no center, automatically find a new center
        if p < -0.5 and self.center < -0.5:
            p = self.check_orthogonal_center(if_print=False)
        elif p < -0.5:
            p = self.center
        if self.center < -0.5:
            self.central_orthogonalization(p)
        elif self.center != p:
            self.orthogonalize_mps(self.center, p)
        self.center = p

    # calculate the environment (two-body terms)
    def environment_s1_s2(self, p, operators, positions):
        # p is the center and the position of the tensor to be updated
        # the two operators are at positions[0] and positions[1]
        if self._debug:
            self.check_orthogonal_center(p)
        v_left = np.zeros(0)
        v_right = np.zeros(0)
        if positions[0] > positions[1]:
            positions = sort_list(positions, [1, 0])
            operators = sort_list(operators, [1, 0])
        if p < positions[0]:
            v_left = np.eye(self.virtual_dim[p])
            v_right = T_module.bound_vec_operator_right2left(self.mps[positions[1]], operators[1], v_right)
            v_right = self.contract_v_l0_to_l1(positions[1]-1, positions[0], v_right)
            v_right = T_module.bound_vec_operator_right2left(self.mps[positions[0]], operators[0], v_right)
            v_right = self.contract_v_l0_to_l1(positions[0] - 1, p, v_right)
            v_middle = np.eye(self.mps[p].shape[1])
        elif p > positions[1]:
            v_left = T_module.bound_vec_operator_left2right(self.mps[positions[0]], operators[0], v_left)
            v_left = self.contract_v_l0_to_l1(positions[0]+1, positions[1], v_left)
            v_left = T_module.bound_vec_operator_left2right(self.mps[positions[1]], operators[1], v_left)
            v_left = self.contract_v_l0_to_l1(positions[1] + 1, p, v_left)
            v_right = np.eye(self.virtual_dim[p + 1])
            v_middle = np.eye(self.mps[p].shape[1])
        elif p == positions[0]:
            v_left = np.eye(self.virtual_dim[p])
            v_right = T_module.bound_vec_operator_right2left(self.mps[positions[1]], operators[1], v_right)
            v_right = self.contract_v_l0_to_l1(positions[1] - 1, p, v_right)
            v_middle = operators[0]
        elif p == positions[1]:
            v_left = T_module.bound_vec_operator_left2right(self.mps[positions[0]], operators[0], v_left)
            v_left = self.contract_v_l0_to_l1(positions[0] + 1, p, v_left)
            v_right = np.eye(self.virtual_dim[p + 1])
            v_middle = operators[1]
        else:
            v_left = T_module.bound_vec_operator_left2right(self.mps[positions[0]], operators[0], v_left)
            v_left = self.contract_v_l0_to_l1(positions[0] + 1, p, v_left)
            v_right = T_module.bound_vec_operator_right2left(self.mps[positions[1]], operators[1], v_right)
            v_right = self.contract_v_l0_to_l1(positions[1] - 1, p, v_right)
            v_middle = np.eye(self.mps[p].shape[1])
        return v_left, v_middle, v_right

    # calculate the environment (one-body terms)
    def environment_s1(self, p, operator, position):
        # p is the position of the tensor to be updated
        # the operator is at positions
        if self._debug:
            self.check_orthogonal_center(p)
            self.check_virtual_bond_dimensions()
        v_left = np.zeros(0)
        v_right = np.zeros(0)
        if p < position:
            v_left = np.eye(self.virtual_dim[p])
            v_right = T_module.bound_vec_operator_right2left(self.mps[position], operator, v_right)
            v_right = self.contract_v_l0_to_l1(position - 1, p, v_right)
            v_middle = np.eye(self.mps[p].shape[1])
        elif p > position:
            v_left = T_module.bound_vec_operator_left2right(self.mps[position], operator, v_left)
            v_left = self.contract_v_l0_to_l1(position + 1, p, v_left)
            v_right = np.eye(self.virtual_dim[p + 1])
            v_middle = np.eye(self.mps[p].shape[1])
        else:  # p == position
            v_left = np.eye(self.virtual_dim[p])
            v_right = np.eye(self.virtual_dim[p + 1])
            v_middle = operator
        return v_left, v_middle, v_right

    # update the boundary vector v by contracting from l0 to l1 without operators
    def contract_v_l0_to_l1(self, l0, l1, v=np.zeros(0)):
        if l0 < l1:
            for n in range(l0, l1):
                v = T_module.bound_vec_operator_left2right(tensor=self.mps[n], v=v)
        elif l0 > l1:
            for n in range(l0, l1, -1):
                v = T_module.bound_vec_operator_right2left(tensor=self.mps[n], v=v)
        return v

    def effective_hamiltonian_dmrg(self, p, operators, index1, index2, coeff1, coeff2, tol=1e-12):
        if self._debug and p != self.center:
            print_error('CenterError: the tensor must be at the orthogonal center before '
                        'defining the function handle', 'magenta')
        nh1 = index1.shape[0]
        s = [self.virtual_dim[p], self.phys_dim, self.virtual_dim[p+1]]
        dim = np.prod(s)
        h_effect = np.zeros((dim, dim))
        for n in range(0, nh1):
            # if the coefficient is too small, ignore its contribution
            if abs(coeff1[n]) > tol and np.linalg.norm(operators[index1[n, 1]].reshape(1, -1)) > tol:
                v_left, v_middle, v_right = self.environment_s1(p, operators[index1[n, 1]], index1[n, 0])
                if self._debug:
                    self.check_environments(v_left, v_middle, v_right, p)
                h_effect += coeff1[n] * np.kron(np.kron(v_left, v_middle), v_right)
        nh2 = index2.shape[0]  # number of two-body Hamiltonians
        for n in range(0, nh2):
            # if the coefficient is too small, ignore its contribution
            if abs(coeff2[n]) > tol:
                v_left, v_middle, v_right = \
                    self.environment_s1_s2(p, [operators[index2[n, 2]], operators[index2[n, 3]]], index2[n, :2])
                if self._debug:
                    self.check_environments(v_left, v_middle, v_right, p)
                h_effect += coeff2[n] * np.kron(np.kron(v_left, v_middle), v_right)
        h_effect = (h_effect + h_effect.conj().T)/2
        return h_effect, s

    def update_tensor_handle_dmrg_1site(self, tensor, p, operators, index1, index2, coeff1, coeff2, tau, tol=1e-12):
        # Very inefficient!!!
        # function handle to put in eigs, to update the p-th tensor
        # index1: one-body interactions, index2: two-body interactions
        # one-body terms: index1[n, 1]-th operator is at the index1[n, 0]-th site
        # tne-body terms: index2[n, 2]-th operator is at the index2[n, 0]-th site
        # tne-body terms: index2[n, 3]-th operator is at the index2[n, 1]-th site
        if self._debug and p != self.center:
            print_error('CenterError: the tensor must be at the orthogonal center before '
                        'defining the function handle', 'magenta')
        tensor = tensor.reshape(self.virtual_dim[p], self.phys_dim, self.virtual_dim[p+1])
        tensor1 = tensor.copy()
        nh1 = index1.shape[0]  # number of two-body Hamiltonians
        for n in range(0, nh1):
            op = operators[index1[n, 1]]
            # if the coefficient is too small, ignore its contribution
            if abs(coeff1[n]) > tol and np.linalg.norm(op.reshape(1, -1)) > tol:
                v_left, v_middle, v_right = self.environment_s1(p, op, index1[n, 0])
                if self._debug:
                    self.check_environments(v_left, v_middle, v_right, p)
                tensor1 -= tau * coeff1[n] * T_module.absorb_matrices2tensor(tensor, [v_left, v_middle, v_right])
        nh2 = index2.shape[0]  # number of two-body Hamiltonians
        for n in range(0, nh2):
            # if the coefficient is too small, ignore its contribution
            if abs(coeff2[n]) > tol:
                op = [operators[index2[n, 2]], operators[index2[n, 3]]]
                v_left, v_middle, v_right = self.environment_s1_s2(p, op, index2[n, :2])
                if self._debug:
                    self.check_environments(v_left, v_middle, v_right, p)
                tensor1 -= tau * coeff2[n] * T_module.absorb_matrices2tensor(tensor, [v_left, v_middle, v_right])
        return tensor1.reshape(-1, 1)

# ========================================================
    # observation functions
    def calculate_entanglement_spectrum(self, if_fast=True):
        # NOTE: this function will central orthogonalize the MPS
        _way = self.decomp_way
        _center = self.center
        self.decomp_way = 'svd'
        if if_fast:
            p0 = self.length - 1
            p1 = 0
            for n in range(0, self.length - 1):
                if self.lm[n].size == 0:
                    p0 = min(p0, n)
                    p1 = max(p1, n)
            self.correct_orthogonal_center(p0)
            if _center > -0.5:
                self.correct_orthogonal_center(p1+1)
            self.correct_orthogonal_center(_center)
        else:
            self.correct_orthogonal_center(0)
            self.correct_orthogonal_center(self.length-1)
            self.correct_orthogonal_center(_center)
        self.decomp_way = _way

    def calculate_entanglement_entropy(self):
        for i in range(0, self.length - 1):
            if self.lm[i].size == 0:
                self.ent[i] = -1
            else:
                self.ent[i] = T_module.entanglement_entropy(self.lm[i])

    def observation_s1(self, operator, position):
        if position > self.center:
            v = T_module.bound_vec_operator_right2left(self.mps[position], operator)
            v = self.contract_v_l0_to_l1(position - 1, self.center-1, v)
        else:
            v = T_module.bound_vec_operator_left2right(self.mps[position], operator)
            v = self.contract_v_l0_to_l1(position + 1, self.center + 1, v)
        return np.trace(v)

    def observation_s1_s2(self, operators, positions):
        if self._debug:
            self.check_mps_norm1()
        if positions[0] > positions[1]:
            operators = sort_list(operators, [1, 0])
            positions = sort_list(positions, [1, 0])
        if self.center < positions[0]:
            v = self.contract_v_l0_to_l1(self.center, positions[0])
        else:
            v = np.zeros(0)
        v = T_module.bound_vec_operator_left2right(self.mps[positions[0]], operators[0], v)
        v = self.contract_v_l0_to_l1(positions[0] + 1, positions[1], v)
        v = T_module.bound_vec_operator_left2right(self.mps[positions[1]], operators[1], v)
        if positions[1] < self.center:
            v = self.contract_v_l0_to_l1(positions[1] + 1, self.center + 1, v)
        return np.trace(v)

    def observe_magnetization(self, s_operator):
        mag = np.zeros((self.length, 1))
        for i in range(0, self.length):
            mag[i] = self.observation_s1(s_operator, i)
        return mag

    def observe_bond_energy(self, index2, coeff2, operators):
        nh = index2.shape[0]
        eb = np.zeros((nh, 1))
        for n in range(0, nh):
            op = [operators[index2[n, 2]], operators[index2[n, 3]]]
            eb[n] = coeff2[n] * self.observation_s1_s2(op, index2[n, :2])
        return eb

    def norm_mps(self):
        # calculate the norm of an MPS
        if self._debug:
            lc = self.check_orthogonal_center()
            if lc != self.center:
                cprint('CenterError: center should be at %d but at %d' % self.center, lc, 'magenta')
                trace_stack()
        if self.center < -0.5:
            v = self.contract_v_l0_to_l1(0, self.length)
            norm = v[0, 0]
        else:
            norm = np.linalg.norm(self.mps[self.center].reshape(1, -1))
        return norm

    def full_coefficients_mps(self, tol_memory=20):
        tot_size_log2 = self.length * np.log2(self.phys_dim) - 5
        if tot_size_log2 > tol_memory:
            cprint('The memory cost of the total coefficients is too large (a lot more than %d Mb). '
                   'Stop calculation' % tot_size_log2, 'magenta')
            cprint('If you want to calculate anyway, please input a larger \'tol_memory\'', 'cyan')
            x = None
        else:
            s = self.mps[0].shape
            x = self.mps[0].reshape(s[0]*s[1], s[2])
            d0 = s[2]
            for n in range(1, self.length):
                s = self.mps[n].shape
                x = x.dot(self.mps[n].reshape(s[0], s[1]*s[2]))
                x.reshape(d0*s[1], s[2])
                d0 = s[2]
        return x.reshape(-1, 1)

# ===========================================================
# functions to show properties

# ===========================================================
# Checking functions
    def check_orthogonal_center(self, expected_center=-2, if_print=True):
        # Check if MPS has the correct center, or at the expected center
        # if not, find the correct center, or recommend a new center while it is not central orthogonal
        # NOTE: no central-orthogonalization in this function, only recommendation
        if self.center > -0.5:
            left = self.orthogonality[:self.center]
            right = self.orthogonality[self.center+1:]
            if not(np.prod(left == -1) and np.prod(right == 1)):
                if if_print:
                    cprint(colored('self.center is incorrect. Change it to -1', 'magenta'))
                    trace_stack()
                self.center = -1
        if self.center < -0.5:
            left = np.nonzero(self.orthogonality == -1)
            left = left[0][-1]
            right = np.nonzero(self.orthogonality == 1)
            right = right[0][0]
            if np.prod(self.orthogonality[:left+1]) and np.prod(self.orthogonality[right:]) and left + 2 == right:
                self.center = left+1
                if if_print:
                    cprint(colored('self.center is corrected to %g' % self.center, 'cyan'))
            else:
                if if_print:
                    cprint(colored('MPS is not central orthogonal. self.center remains -1', 'cyan'))
        else:
            left = self.center - 1
        if self.center > -0.5:
            if expected_center > -0.5 and expected_center != self.center:
                cprint('The center is at %d, not the expected position at %d' % (self.center, expected_center))
            recommend_center = self.center
        else:
            # if not central-orthogonal, recommend the tensor after the last left-orthogonal one as the new center
            recommend_center = left + 1
        return recommend_center

    def check_orthogonality_by_tensors(self, tol=1e-20, is_print=True):
        incorrect_ort = list()
        for n in range(0, self.length):
            if self.orthogonality[n] == -1:
                is_ort = T_module.check_orthogonality(self.mps[n], [0, 1], 2, tol=tol)
            elif self.orthogonality[n] == 1:
                is_ort = T_module.check_orthogonality(self.mps[n], 0, [1, 2], tol=tol)
            else:
                is_ort = True
            if not is_ort:
                incorrect_ort.append(n)
        if is_print:
            if incorrect_ort.__len__() == 0:
                print('The orthogonality of all tensors are marked correctly by self.orthogonality')
            else:
                cprint('In self.orthogonality, the orthogonality of the following tensors is incorrect:', 'magenta')
                cprint(str(incorrect_ort), 'cyan')
        return incorrect_ort

    def check_environments(self, vl, vm, vr, n):
        # check if the environments of the n-th tensor have consistent dimensions
        is_bug0 = False
        bond = str()
        if vl.shape[0] != vl.shape[1]:
            is_bug0 = True
            bond = 'LEFT'
        if vm.shape[0] != vm.shape[1]:
            is_bug0 = True
            bond = 'MIDDLE'
        if vr.shape[0] != vr.shape[1]:
            is_bug0 = True
            bond = 'RIGHT'
        if is_bug0:
            cprint('EnvError: for the %d-th tensor, the ' % n + bond + ' v is not square', 'magenta')

        is_bug = False
        if vl.shape[0] != self.virtual_dim[n]:
            is_bug = True
            bond = 'LEFT'
        if vr.shape[0] != self.virtual_dim[n + 1]:
            is_bug = True
            bond = 'RIGHT'
        if vm.shape[0] != self.mps[n].shape[1]:
            bond = 'MIDDLE'
            is_bug = True
        if is_bug:
            cprint('EnvError: for the %d-th tensor, the ' % n + bond + ' v has inconsistent dimension', 'magenta')
        if is_bug0 or is_bug:
            trace_stack()

    def check_virtual_bond_dimensions(self):
        is_error = False
        for n in range(1, self.length):
            if self.virtual_dim[n] != self.mps[n].shape[0] or self.virtual_dim[n] != self.mps[n-1].shape[2]:
                cprint('VirBondDimError: inconsistent dimension detected for the %d-th virtual bond' % n, 'magenta')
                is_error = True
        if is_error:
            trace_stack()

    def check_mps_norm1(self, if_print=False):
        # check if the MPS is norm-1
        norm = self.norm_mps()
        if abs(norm - 1) > 1e-14:
            print_error('The norm is MPS is %g away from 1' % abs(norm - 1))
        if if_print:
            cprint('The norm of MPS is %g' % norm, 'cyan')




