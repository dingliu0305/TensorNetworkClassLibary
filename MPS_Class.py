import numpy as np
import MPS_Basic_Module as mm


class MpsOpenBoundaryClass:

    def __init__(self, l, d, chi, way="svd"):
        self.mps = mm.random_open_mps(l, d, chi)
        self.decomp_way = way
        self.length = l

    def left2right_orthogonalization(self, l0, l1):
        for n in range(l0, l1):
            self.mps[n], mat = mm.left2right_decompose_tensor(self.mps[n], self.decomp_way)
            self.mps[n+1] = mm.absorb_matrix2tensor(self.mps[n+1], mat, 1)

    def right2left_orthogonalization(self, l1, l0):
        for n in range(l1, l0, -1):
            self.mps[n], mat = mm.right2left_decompose_tensor(self.mps[n], self.decomp_way)
            self.mps[n-1] = mm.absorb_matrix2tensor(self.mps[n - 1], mat, 3)

    def central_orthogonalization(self,lc):
        self.left2right_orthogonalization(0, lc)
        self.right2left_orthogonalization(self.length, lc)
