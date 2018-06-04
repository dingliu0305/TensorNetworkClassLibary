import numpy as np
from Basic_Functions_SJR import sort_list
from termcolor import cprint
is_debug = 1


def random_open_mps(l, d, chi):
    # Create a random MPS with open boundary condition
    # l: length; d: physical dimension; chi: virtual dimension
    mps = list(range(0, l))
    mps[0] = np.random.randn(1, d, chi)
    mps[l-1] = np.random.randn(chi, d, 1)
    for n in range(1, l-1):
        mps[n] = np.random.randn(chi, d, chi)
    return mps


def decompose_tensor_one_bond(tensor, n, way):
    # decompose a matrix from te n-th bond of the tensor
    # the resulting tensor is an isometry
    s1 = tensor.shape
    dim = tensor.ndim
    index1 = np.append(np.arange(0, n-1), np.arange(n+1, dim))
    tensor = tensor.permute(np.append(index1, n))
    tensor = tensor.reshape(np.prod(s1[index1]), s1[n])
    d_min = min(np.prod(s1[index1]), s1[n])
    if way == 1 or way == "svd":
        # Use SVD decomposition
        tensor, lm, v = np.linalg.svd(tensor)
        v = np.dot(np.diag(lm), v[:d_min, :])
    else:
        # Use QR decomposition
        tensor, v = np.linalg.qr(tensor)
    tensor = tensor.reshape(np.append(s1[index1], d_min))
    permute_back = np.append(np.append(np.arange(0, n-1), dim), np.arange(n, dim-1))
    tensor = tensor.transpose(permute_back)
    return tensor, v, d_min


def left2right_decompose_tensor(tensor, way):
    # Transform a local tensor to left2right orthogonal form
    # for manipulate MPS
    s1 = tensor.shape
    dim = min(s1[0]*s1[1], s1[2])
    tensor = tensor.reshape(s1[0] * s1[1], s1[2])
    if way == 1 or way == "svd":
        # Use SVD decomposition
        tensor, lm, v = np.linalg.svd(tensor)
        v = np.dot(np.diag(lm), v[:dim, :])
    else:
        # Use QR decomposition
        tensor, v = np.linalg.qr(tensor)
        lm = np.zeros(0)
    tensor = tensor[:, :dim].reshape(s1[0], s1[1], dim)
    v = v.T
    return tensor, v, dim, lm


def right2left_decompose_tensor(tensor, way):
    # Transform a local tensor to left2right orthogonal form
    # for manipulate MPS
    s1 = np.shape(tensor)
    tensor = tensor.reshape(s1[0], s1[1]*s1[2])
    dim = min(s1[0], s1[1]*s1[2])
    if way == 1 or way == "svd":
        # Use SVD decomposition
        tensor, lm, v = np.linalg.svd(tensor.T)
        v = np.dot(np.diag(lm), v[:dim, :])
    else:
        # Use QR decomposition
        tensor, v = np.linalg.qr(tensor.T)
        lm = np.zeros(0)
    tensor = tensor[:, :dim].T.reshape(dim, s1[1], s1[2])
    v = v.T
    return tensor, v, dim, lm


def absorb_matrix2tensor(tensor, mat, bond):
    # generally, recommend to use the function 'absorb_matrices2tensor'
    # contract the 1st bond of mat with tensor
    is_id = is_identity(mat)
    if is_id:  # if the matrix is an identity, accelerate the computation
        tensor1 = tensor * is_id
    else:
        s = np.array(tensor.shape)
        nd = tensor.ndim
        if bond == 0:
            tensor1 = mat.T.dot(tensor.reshape(s[0], np.prod(s[1:nd])))
            s[0] = mat.shape[1]
        elif bond == nd-1:
            tensor1 = tensor.reshape(np.prod(s[0:nd-1]), s[nd-1]).dot(mat)
            s[-1] = mat.shape[1]
        else:
            ind = np.arange(0, bond)
            ind = np.append(ind, np.arange(bond+1, nd))
            tensor1 = tensor.transpose(np.append(ind, bond)).reshape(np.prod(s(ind)), s(bond)).dot(mat)
            tensor1 = tensor1.reshape(s[np.append(ind, bond)])
            ind = np.arange(0, bond)
            ind = np.append(np.append(ind, nd-1), np.arange(bond, nd-1))
            tensor1.transpose(ind)
        if bond == 0 or bond == nd-1:
            tensor1 = tensor1.reshape(s)
    return tensor1


def absorb_matrices2tensor_full_fast(tensor, mats):
    # generally, recommend to use the function 'absorb_matrices2tensor'
    # each bond will have a matrix to contract with
    # the matrices must be in the right order
    # contract the 1st bond of mat with tensor
    nb = tensor.ndim
    s = np.array(tensor.shape)
    if is_debug:
        for n in range(0, nb):
            if mats[n].shape[1] != s[n]:
                cprint('Error: the %d-th matrix has inconsistent dimension with the tensor' % n, 'magenta')
    for n in range(nb-1, -1, -1):
        tensor = tensor.reshape(np.prod(s[:nb-1]), s[nb-1]).dot(mats[n])
        s[-1] = mats[n].shape[1]
        ind = np.append(nb-1, np.arange(0, nb-1))
        tensor = tensor.reshape(s).transpose(ind)
        s = s[ind]
    return tensor


def absorb_matrices2tensor(tensor, mats, bonds=-1, mat_bond=-1):
    # default: contract the 1st bond of mat with tensor
    nm = len(mats)  # number of matrices to be contracted
    if bonds < 0:  # set default of bonds: contract all matrices in order, starting from the 0th bond
        bonds = np.arange(0, nm)
    if mat_bond < 0:  # set default of mat_bond: contract the 1st bond of each matrix
        mat_bond = np.zeros((nm, 1))
    for i in range(0, nm):  # permute if the second bond of a matrix is to be contracted
        if mat_bond[i] == 1:
            mats[i] = mats[i].T
    nb = bonds.size
    # check if full_fast function can be used
    if np.array_equiv(np.sort(bonds), np.arange(0, nb)):
        order = np.argsort(bonds)
        mats = sort_list(mats, order)
        # this full_fast function can be used when each bond has a matrix which are arranged in the correct order
        tensor = absorb_matrices2tensor_full_fast(tensor, mats)
    else:
        for i in range(0, nb):
            tensor = absorb_matrix2tensor(tensor, mats[i], bonds[i])
    return tensor


def bound_vec_operator_left2right(tensor, op=np.zeros(0), v=np.zeros(0), normalize=1):
    s = tensor.shape
    if op.size != 0:  # deal with the operator
        tensor1 = tensor.transpose(0, 2, 1).reshape(s[0]*s[2], s[1]).dot(op)
        tensor1 = tensor1.reshape(s[0], s[2], s[1]).transpose(0, 2, 1)
    else:  # no operator
        tensor1 = tensor
    if v.size == 0:  # no input boundary vector v
        tensor = tensor.reshape(s[0]*s[1], s[2]).conj()
        tensor1 = tensor1.reshape(s[0]*s[1], s[2])
        v1 = tensor.T.dot(tensor1)
    else:  # there is an input boundary vector v
        tensor1 = v.dot(tensor1.reshape(s[0], s[1]*s[2]))
        v1 = tensor.conj().reshape(s[0]*s[1], s[2]).T.dot(tensor1.reshape(s[0]*s[1], s[2]))
    if normalize:
        v1 = v1/np.linalg.norm(v1.reshape(s[2]*s[2], 1))
    return v1


def bound_vec_operator_right2left(tensor, op=np.zeros(0), v=np.zeros(0), normalize=1):
    s = tensor.shape
    if op.size != 0:  # deal with the operator
        tensor1 = tensor.transpose(0, 2, 1).reshape(s[0]*s[2], s[1]).dot(op)
        tensor1 = tensor1.reshape(s[0], s[2], s[1]).transpose(0, 2, 1)
    else:  # no operator
        tensor1 = tensor
    if v.size == 0:  # no input boundary vector v
        tensor = tensor.reshape(s[0], s[1]*s[2]).conj()
        tensor1 = tensor1.reshape(s[0], s[1]*s[2])
        v1 = tensor.dot(tensor1.T)
    else:  # there is an input boundary vector v
        tensor = tensor.reshape(s[0]*s[1], s[2]).conj().dot(v)
        v1 = tensor.reshape(s[0], s[1]*s[2]).dot(tensor1.reshape(s[0], s[1]*s[2]).T)
    if normalize:
        v1 = v1/np.linalg.norm(v1.reshape(s[0]*s[0], 1))
    return v1


def transfer_matrix_mps(tensor):
    s = tensor.shape
    tmp = tensor.transpose(0, 2, 1).reshape(s[0]*s[2], s[1])
    tm = tmp.conj().dot(tmp.T).reshape(s[0], s[2], s[0], s[2])
    tm = tm.transpose(0, 2, 1, 3).reshape(s[0]*s[0], s[2]*s[2])
    return tm


def is_identity(mat, tol=1e-15):
    # if mat is an identity, return c with mat = c*I
    # if not, return False
    c = mat[0, 0]
    s = mat.shape
    if c < tol or mat.ndim != 2 or s[0] != s[1]:
        is_id = False
    else:
        tmp = mat/c
        tmp -= np.eye(mat.shape[0])
        error = np.linalg.norm(tmp)
        if error < tol:
            is_id = c
        else:
            is_id = False
    return is_id
