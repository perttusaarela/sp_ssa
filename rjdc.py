import numpy as np

def do_python_rjdc(X, kpmaxit, w, eps, results):
    from ctypes import c_int, c_double, CDLL
    lib_path = './c_src/init.so'
    add_lib = CDLL(lib_path)
    py_rjdc = add_lib.rjdc
    size_X = len(X)
    size_kpmaxit = 3
    size_w = len(w)
    size_eps = 1
    size_results = len(results)
    X_in = (c_double * size_X)(*X)
    kpmaxit_in = (c_int* size_kpmaxit)(*kpmaxit)
    w_in = (c_double * size_w)(*w)
    eps_in = (c_double * size_eps)(*eps)
    results_out = (c_double * size_results)(*results)
    py_rjdc(X_in, kpmaxit_in, w_in, eps_in, results_out)
    return results_out[:]


def joint_diagonalization(X, weight=None, maxiter=1000, eps=1e-6):

    kp, p = X.shape
    k = kp // p
    assert k * p == kp

    if weight is None:
        weight = np.ones(k, dtype=float)

    res = do_python_rjdc(X.flatten(order="F"), [k, p, maxiter], weight, [eps], np.zeros(p**2+1, dtype=float))

    iter = res[-1]
    assert iter < maxiter, "maxiter reached without convergence"

    V = np.asarray(res[:-1]).reshape([p,p]).transpose()
    D = []
    for i in range(k):
        matrix = X[i * p:(i + 1) * p, :]
        D.append(V.T @ (matrix.T @ V))

    return V, D, iter
