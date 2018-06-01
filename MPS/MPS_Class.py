import MPS_Basic_Module as Mm
import numpy as np


class MpsOpenBoundaryClass:

    def __init__(self, l, d, chi, way="svd"):
        self.mps = Mm.random_open_mps(l, d, chi)
        self.decomp_way = way  # 'svd' or 'qr'
        self.length = l
        self.center = -1  # orthogonal center; -1 means no center
        self.virtual_dim = np.ones((1, l+1)) * chi

    # Orthogonalize the MPS from the l0-th to l1-th site (l0<l1)
    def orthogonalize_mps(self, l0, l1):
        if l0 < l1:  # Orthogonalize MPS from left to rigth
            for n in range(l0, l1):
                self.mps[n], mat, self.virtual_dim[n] = Mm.left2right_decompose_tensor(self.mps[n], self.decomp_way)
                self.mps[n+1] = Mm.absorb_matrix2tensor(self.mps[n+1], mat, 1)
        elif l0 > l1:  # Orthogonalize MPS from right to left
            for n in range(l0, l1, -1):
                self.mps[n], mat, self.virtual_dim[n-1] = Mm.right2left_decompose_tensor(self.mps[n], self.decomp_way)
                self.mps[n - 1] = Mm.absorb_matrix2tensor(self.mps[n - 1], mat, 3)
        self.center = l1

    # transfer the MPS into the central orthogonal form with the center lc
    def central_orthogonalization(self, lc, l0=0, l1=0):
        if l1 == 0:
            l1 = self.length
        self.orthogonalize_mps(l0, lc)
        self.orthogonalize_mps(l1, lc)

    # put the orthogonal center at p
    def correct_orthogonal_center(self, p):
        if self.center < 0:
            self.orthogonalize_mps(0, p)
            self.orthogonalize_mps(self.length, p)
        elif self.center != p:
            self.orthogonalize_mps(self.center, p)

    # calculate the environment
    def environment_s1_s2(self, p, operators, positions):
        # p is the position of the tensor to be updated
        # the two operators are at positions[0] and positions[1]
        self.correct_orthogonal_center(p)
        v_left = np.zeros(0)
        v_right = np.zeros(0)
        if p < positions[0]:
            v_left = np.eye(self.virtual_dim[p-1])
            v_right = Mm.bound_vec_operator_right2left(self.mps[positions[1]], operators[1], v_right)
            v_right = self.contract_v_l0_to_l1(positions[1]-1, positions[0], v_right)
            v_right = Mm.bound_vec_operator_right2left(self.mps[positions[0]], operators[0], v_right)
            v_right = self.contract_v_l0_to_l1(positions[0] - 1, p, v_right)
            v_middle = np.eye(self.mps[p].size[1])
        elif p > positions[1]:
            v_left = Mm.bound_vec_operator_left2right(self.mps[positions[0]], operators[0], v_left)
            v_left = self.contract_v_l0_to_l1(positions[0]+1, positions[1], v_left)
            v_left = Mm.bound_vec_operator_left2right(self.mps[positions[1]], operators[1], v_left)
            v_left = self.contract_v_l0_to_l1(positions[1] + 1, p, v_left)
            v_right = np.eye(self.virtual_dim[p])
            v_middle = np.eye(self.mps[p].size[1])
        elif p == positions[0]:
            v_left = np.eye(self.virtual_dim[p - 1])
            v_right = Mm.bound_vec_operator_right2left(self.mps[positions[1]], operators[1], v_right)
            v_right = self.contract_v_l0_to_l1(positions[1] - 1, p, v_right)
            v_middle = operators[0]
        elif p == positions[1]:
            v_left = Mm.bound_vec_operator_left2right(self.mps[positions[0]], operators[0], v_left)
            v_left = self.contract_v_l0_to_l1(positions[0] + 1, p, v_left)
            v_right = np.eye(self.virtual_dim[p])
            v_middle = operators[1]
        else:
            v_left = Mm.bound_vec_operator_left2right(self.mps[positions[0]], operators[0], v_left)
            v_left = self.contract_v_l0_to_l1(positions[0] + 1, p, v_left)
            v_right = Mm.bound_vec_operator_right2left(self.mps[positions[1]], operators[1], v_right)
            v_right = self.contract_v_l0_to_l1(positions[1] - 1, p, v_right)
            v_middle = np.eye(self.mps[p].size[1])
        return v_left, v_middle, v_right

    # update the boundary vector v by contracting from l0 to l1 without operators
    def contract_v_l0_to_l1(self, l0, l1, v=np.zeros(0)):
        if l0 < l1:
            for n in range(l0, l1):
                v = Mm.bound_vec_operator_left2right(tensor=self.mps[n], v=v)
        elif l0 < l1:
            for n in range(l1, l0, -1):
                v = Mm.bound_vec_operator_right2left(tensor=self.mps[n], v=v)
        return v

    def update_tensor_handle(self, p, operators, index):
        # function handle to put in eigs, to update the p-th tensor
        # index[n, 0]-th operator is at the index[n, 2]-th site
        # index[n, 1]-th operator is at the index[n, 3]-th site
        nh = index.shape[1]
        for n in range(0, nh):
            op = [operators[index[n, 0]], operators[index[n,1]]]
            v_left, v_middle, v_right = self.environment_s1_s2(p, op, index[n, 2:3])


