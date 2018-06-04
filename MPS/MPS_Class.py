import Tensor_Basic_Module as T_module
import numpy as np
from Basic_Functions_SJR import empty_list, trace_stack
from termcolor import colored, cprint


class MpsOpenBoundaryClass:

    def __init__(self, length, d, chi, way="svd", debug=0):
        self.mps = T_module.random_open_mps(length, d, chi)
        self.decomp_way = way  # 'svd' or 'qr'
        self.length = length
        # self.orthogonality:  -2: not orthogonal; -1: left2right; 0: not orthogonal but center; 1: right2left
        self.orthogonality = -2 * np.zeros((length, 1))
        self.center = -1  # orthogonal center; -1 means no center
        self.virtual_dim = np.ones((length+1, )).astype(int)*chi
        self.virtual_dim[0] = 1
        self.virtual_dim[-1] = 1
        self.lm = empty_list(length-1)
        self._debug = debug  # if in debug mode

    def report_yourself(self):
        print('center: ' + str(self.center))
        print('orthogonality:' + str(self.orthogonality.T))
        print('virtual bond dimensions: ' + str(self.virtual_dim))

    # Orthogonalize the MPS from the l0-th to l1-th site (l0<l1)
    def orthogonalize_mps(self, l0, l1):
        if l0 < l1:  # Orthogonalize MPS from left to rigth
            for n in range(l0, l1):
                self.mps[n], mat, self.virtual_dim[n+1], lm =\
                    T_module.left2right_decompose_tensor(self.mps[n], self.decomp_way)
                if lm.size > 0 and self.center > 0:
                    self.lm[n] = lm
                self.orthogonality[n] = -1
                self.mps[n+1] = T_module.absorb_matrix2tensor(self.mps[n + 1], mat, 0)
        elif l0 > l1:  # Orthogonalize MPS from right to left
            for n in range(l0, l1, -1):
                self.mps[n], mat, self.virtual_dim[n], lm =\
                    T_module.right2left_decompose_tensor(self.mps[n], self.decomp_way)
                if lm.size > 0 and self.center > 0:
                    self.lm[n-1] = lm
                self.orthogonality[n] = 1
                self.mps[n-1] = T_module.absorb_matrix2tensor(self.mps[n - 1], mat, 2)
        self.center = l1

    # transfer the MPS into the central orthogonal form with the center lc
    def central_orthogonalization(self, lc, l0=0, l1=-1):
        if l1 == -1:
            l1 = self.length-1
        self.orthogonalize_mps(l0, lc)
        self.orthogonalize_mps(l1, lc)
        self.orthogonality[lc] = 0
        if self._debug:
            self.check_orthogonal_center()
            if self.center == -1:
                cprint('Some bugs in central orthogonalizaiton', 'red')

    # put the orthogonal center at p
    def correct_orthogonal_center(self, p):
        if self.center < 0:
            self.orthogonalize_mps(0, p)
            self.orthogonalize_mps(self.length-1, p)
        elif self.center != p:
            self.orthogonalize_mps(self.center-1, p)
        self.orthogonality[p] = 0

    # calculate the environment (two-body terms)
    def environment_s1_s2(self, p, operators, positions):
        # p is the position of the tensor to be updated
        # the two operators are at positions[0] and positions[1]
        self.correct_orthogonal_center(p)
        v_left = np.zeros(0)
        v_right = np.zeros(0)
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
        self.correct_orthogonal_center(p)
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

    def update_tensor_handle_dmrg_1site(self, tensor, p, operators, index1, index2, coeff1, coeff2, tau, tol=1e-12):
        # function handle to put in eigs, to update the p-th tensor
        # index1: one-body interactions, index2: two-body interactions
        # index1[n, 1]-th operator is at the index2[n, 0]-th site
        # index2[n, 2]-th operator is at the index2[n, 0]-th site
        # index2[n, 3]-th operator is at the index2[n, 1]-th site
        nh1 = index1.shape[0]  # number of two-body Hamiltonians
        for n in range(0, nh1):
            if abs(coeff1[n]) > tol:  # if the coefficient is too small, ignore its contribution
                op = operators[index1[n, 1]]
                v_left, v_middle, v_right = self.environment_s1(p, op, index1[n, 0])
                if self._debug:
                    self.check_environments(v_left, v_middle, v_right, n)
                tensor -= tau * coeff1[n] * T_module.absorb_matrices2tensor(tensor, [v_left, v_middle, v_right])
        nh2 = index2.shape[0]  # number of two-body Hamiltonians
        for n in range(0, nh2):
            if abs(coeff2[n]) > tol:  # if the coefficient is too small, ignore its contribution
                op = [operators[index2[n, 2]], operators[index2[n, 3]]]
                v_left, v_middle, v_right = self.environment_s1_s2(p, op, index2[n, :2])
                tensor -= tau * coeff2[n] * T_module.absorb_matrices2tensor(tensor, [v_left, v_middle, v_right])
        return tensor

    def check_orthogonal_center(self):
        if self.center > 0:
            left = self.orthogonality[:self.center]
            right = self.orthogonality[self.center+1:]
            if not(np.prod(left == -1) and np.prod(right == 1)):
                cprint(colored('self.center is incorrect. Change it to -1', 'magenta'))
                self.center = -1
        if self.center < 0:
            left = np.nonzero(self.orthogonality == -1)
            left = left[0][-1]
            right = np.nonzero(self.orthogonality == 1)
            right = right[0][0]
            if np.prod(self.orthogonality[:left+1]) and np.prod(self.orthogonality[right:]) and left + 2 == right:
                self.center = left+1
                cprint(colored('self.center is corrected to %g' % self.center, 'cyan'))
            else:
                cprint(colored('MPS is not central orthogonal. self.center remains -1', 'cyan'))

    def check_environments(self, vl, vm, vr, n):
        # check if the environments of the n-th tensor have consistent dimensions
        if vl.shape[0] != self.virtual_dim[n]:
            cprint('For the environments of the n-th tensor, the LEFT v has inconsistent dimension', 'magenta')
            trace_stack()
        if vr.shape[0] != self.virtual_dim[n + 1]:
            cprint('For the environments of the n-th tensor, the RIGHT v has inconsistent dimension', 'magenta')
            trace_stack()
        if vm.shape[0] != self.mps[n].shape[1]:
            cprint('For the environments of the n-th tensor, the MIDDLE v has inconsistent dimension', 'magenta')
            trace_stack()



