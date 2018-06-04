# Notes (about some frequently used BIF):
# np.hstack((m1,m2)): 合并两个矩阵，成（d, d1+d2）
# np.vstack((m1,m2))：合并两个矩阵， 成（d1+d2，d）

import numpy as np
import inspect
import sys
import traceback
from termcolor import cprint


def sort_list(a, order):
    a1 = [a[i] for i in order]
    return a1


def empty_list(n):
    # make a empty list of size n
    a = list(range(0, n))
    for n in range(0, n):
        a[n] = []
    return a


def arg_find_array(arg, n=1, which='first'):
    # find the first/last n True's in the arg
    # like the "find" function in Matlab
    # the input must be array or ndarray
    x = np.nonzero(arg)
    length = x[0].size
    num = min(length, n)
    dim = arg.ndim
    y = np.zeros((dim, num))
    if which == 'last':
        for i in range(0, dim):
            y[i, :] = x[i][length-num:length]
    else:
        for i in range(0, dim):
            y[i, :] = x[i][:num]
    return y


def trace_stack():
    # print the line and file name where this function is used
    info = inspect.stack()
    ns = info.__len__()
    for ns in range(2, ns):
        cprint('in ' + str(info[ns][1]) + ' at line ' + str(info[ns][2]), 'green')



