import numpy as np


def random_open_mps(l, d, chi):
    # Create a random MPS with open boundary condition
    # l: length; d: physical dimension; chi: virtual dimension
    mps = list(range(0, l))
    mps[0] = np.random.randn(1, d, chi)
    mps[l-1] = np.random.randn(chi, d, 1)
    for n in range(1, l-1):
        mps[n] = np.random.randn(chi, d, chi)
    return mps


def left2right_decompose_tensor(tensor, way):
    # Transform a local tensor to left2right orthogonal form
    s1 = tensor.shape
    tensor = tensor.reshape(s1[0] * s1[1], s1[2])
    if way == 1 or way == "svd":
        # Use SVD decomposition
        tensor, lm, v = np.linalg.svd(tensor)
        lm = np.diag(lm)
        v = np.dot(lm, v)
    else:
        # Use QR decomposition
        tensor, v = np.linalg.qr(tensor)
    s2 = tensor.shape
    tensor = tensor[:, :s2[1]].reshape(s1[0], s1[1], s2[1])
    return tensor, v


def right2left_decompose_tensor(tensor, way):
    # Transform a local tensor to left2right orthogonal form
    s1 = np.shape(tensor)
    tensor = np.reshape(tensor, (s1[0], s1[1]*s1[2]))
    if way == 1 or way == "svd":
        # Use SVD decomposition
        tensor, lm, v = np.linalg.svd(tensor.T)
        lm = np.diag(lm)
        v = np.dot(lm, v)
    else:
        # Use QR decomposition
        tensor, v = np.linalg.qr(tensor.T)
    s2 = np.shape(tensor)
    tensor = tensor[:, :s2[1]].T.reshape(s2[1], s1[1], s1[2])
    return tensor, v


def absorb_matrix2tensor(tensor, mat, bond):
    s = tensor.shape
    if bond == 1:
        tensor1 = mat.dot(tensor.reshape(s[0], s[1]*s[2]))
    elif bond == 3:
        tensor1 = tensor.reshape(s[0]*s[1], s[2]).dot(mat.T)
    else:
        tensor1 = mat.dot(tensor.transpose(1, 0, 2).reshape(s[1], s[0]*s[2]))
        tensor1 = tensor1.reshape(s[1], s[0], s[2]).transpose(1, 0, 2)
    if bond != 2:
        tensor1.reshape(s)
    return tensor1


def bound_vec_operator_left2right(tensor, op=np.zeros(0), v=np.zeros(0), normalize=1):
    s = tensor.size
    if op.size != 0:
        tensor1 = tensor.transpose(0, 2, 1).reshape(s[0]*s[2], s[1]).dot(op)
        tensor1.reshape(s[0], s[2], s[1]).transpose(0, 2, 1)
    else:
        tensor1 = tensor
    if v.size == 0:
        tensor = tensor.reshape(s[0]*s[1], s[2]).conj()
        tensor1 = tensor1.reshape(s[0]*s[1], s[2])
        v1 = tensor.T.dot(tensor1)
    else:
        tensor1 = v.dot(tensor1.reshape(s[0], s[1]*s[2]))
        v1 = tensor.conj().reshape(s[0]*s[1], s[2]).T.dot(tensor1.reshape(s[0]*s[1], s[2]))
    if normalize:
        v1 = v1/np.linalg.norm(v1.reshape(s[0]*s[1]*s[2], 1))
    return v1
