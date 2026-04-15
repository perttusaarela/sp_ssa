import numpy as np
from pyssaBSS import joint_diagonalization
from functools import partial

from utils import sample_mean, sample_covariance, standardize_data
from spatio_temporal import get_unique_spatial_locations


def to_projector(mat):
    AtA = mat.T @ mat
    if np.linalg.matrix_rank(AtA) != mat.shape[0]:
        inv = np.linalg.pinv(AtA)
    else:
        inv = np.linalg.inv(AtA)
    return mat @ inv @ mat.T


def compare_as_projectors(mat1, mat2):
    assert mat1.shape == mat2.shape
    p1 = to_projector(mat1)
    p2 = to_projector(mat2)
    sub = p1 - p2
    return 0.5 * np.linalg.norm(sub, ord="fro") ** 2

class STSSA:
    def __init__(self, data, num_non_stationary=0):
        white_signals, whitener = standardize_data(data)
        self.data = white_signals
        self.whitener = whitener
        self.nonstationary_dim = num_non_stationary
        self.aux = None

    def set_nonstationary_dim(self, num_non_stationary):
        self.nonstationary_dim = num_non_stationary

    def sir(self, segments):
        res = stssa_sir(self.data, segments)
        ss, ns = res.get_subspaces(self.whitener, self.nonstationary_dim)
        return ss, ns

    def save(self, segments):
        res = stssa_save(self.data, segments)
        ss, ns = res.get_subspaces(self.whitener, self.nonstationary_dim)
        return ss, ns

    def lcor(self, coords, segments, kernel=("b", 2.2)):
        res = stssa_lcor(self.data, coords, segments, kernel)
        ss, ns = res.get_subspaces(self.whitener, self.nonstationary_dim)
        return ss, ns

    def comb(self, coords, segments, kernel=("b", 2.2)):
        res = stssa_comb(self.data, coords, segments, kernel)
        ss, ns = res.get_subspaces(self.whitener, self.nonstationary_dim)
        self.aux = res.aux
        return ss, ns
    
    
class STSSAResultsObject:
    def __init__(self, m_mat=None, diagonalizer=None, diagonal=None):
        self.m_mat = m_mat
        self.diagonalizer = diagonalizer
        self.diagonal = diagonal
        self.aux = {}

    def get_subspaces(self, whitener: np.ndarray, num_non_stationary: int):
        V = self.diagonalizer
        ns_eigs = V[:, :num_non_stationary].T
        ss_eigs = V[:, num_non_stationary:].T
        non_stationary_projector = ns_eigs @ whitener
        stationary_projector = ss_eigs @ whitener
        return stationary_projector, non_stationary_projector

    def sort_by_magnitude(self):
        abs_diagonal = np.abs(self.diagonal)
        perm = np.argsort(abs_diagonal)[::-1]
        self.diagonal = self.diagonal[perm]
        self.diagonalizer = self.diagonalizer[:, perm]
        self.m_mat = self.m_mat[np.ix_(perm, perm)]


def stssa_sir(observations, segments):
    full_range = observations.shape[1]

    m_mat = np.zeros((observations.shape[0], observations.shape[0]))
    for segment in segments:
        if len(segment) == 0:
            continue
        mean_vec = sample_mean(observations, segment)
        mean_mat = np.outer(mean_vec, mean_vec)
        m_mat += (len(segment) / full_range) * mean_mat

    eigvals, eigvecs = np.linalg.eig(m_mat)
    perm = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, perm]
    eigvals = eigvals[perm]

    return STSSAResultsObject(m_mat=m_mat, diagonalizer=eigvecs, diagonal=eigvals)


def stssa_save(observations, segments):
    full_range = observations.shape[1]

    m_mat = np.zeros((observations.shape[0], observations.shape[0]))
    for segment in segments:
        if len(segment) == 0:
            continue
        cov_mat = np.eye(observations.shape[0]) - sample_covariance(observations, segment=segment)
        summand = cov_mat @ cov_mat
        m_mat += (len(segment) / full_range) * summand

    eigvals, eigvecs = np.linalg.eig(m_mat)
    perm = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, perm]
    eigvals = eigvals[perm]

    return STSSAResultsObject(m_mat=m_mat, diagonalizer=eigvecs, diagonal=eigvals)


def st_ball_kernel_local_sample_covariance(data, coords, radius, lag, segment=None, seg_mean=None, scale=True):
    """
    Local covariance in spatio-temporal coordinates (x, y, t)
    using a ball kernel.
    """
    if segment is None:
        segment = np.arange(data.shape[1])
    segment = np.asarray(segment)

    X = data[:, segment]
    C = coords[segment]
    N = X.shape[1]
    p = X.shape[0]

    """
    diffs = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)

    mask = (sq_dists <= radius ** 2).astype(float)
    np.fill_diagonal(mask, 0.0)

    l_cov = (X_centered @ mask @ X_centered.T) / N
    """
    if seg_mean is None:
        seg_mean = np.mean(X, axis=1, keepdims=True)
    else:
        seg_mean = seg_mean[:, np.newaxis]

    X_centered = X - seg_mean

    C_loc = get_unique_spatial_locations(C)
    num_locations = len(C_loc)
    l_cov = np.zeros((p, p))
    for i, coord1 in enumerate(C_loc):
        temp = np.zeros((p, p))
        counter = 0
        for j, coord2 in enumerate(C_loc):
            if i == j:
                continue
            if np.linalg.norm(coord1[0:1] - coord2[0:1]) <= radius:
                X_i = X_centered[:, i::num_locations]
                X_j = X_centered[:, j::num_locations]
                #temp += np.outer(X_centered[:, i], X_centered[:, j])
                temp += X_i[:, :-lag] @ X_j[:, lag:].T
                counter += 1

        if scale and counter > 0:
            temp /= counter

        l_cov += temp

    l_cov /= N

    return l_cov


def stssa_lcor(observations, coords, segments, kernel=("b", 2.2), lag=1):
    full_range = observations.shape[1]
    m_mat = np.zeros((observations.shape[0], observations.shape[0]))

    if kernel[0] == "b":
        full_auto_cov = st_ball_kernel_local_sample_covariance(
            observations, coords, radius=kernel[1], lag=lag
        )
        func = partial(
            st_ball_kernel_local_sample_covariance,
            data=observations,
            coords=coords,
            radius=kernel[1],
            lag=lag
        )
    else:
        raise ValueError("For the first version, use kernel ('b', 2.2)")

    for segment in segments:
        if len(segment) == 0:
            continue
        cov_mat = func(segment=segment)
        diff = full_auto_cov - cov_mat
        sq_diff = diff @ diff
        m_mat += (len(segment) / full_range) * sq_diff

    eigvals, eigvecs = np.linalg.eig(m_mat)
    perm = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, perm]
    eigvals = eigvals[perm]

    return STSSAResultsObject(m_mat=m_mat, diagonalizer=eigvecs, diagonal=eigvals)


def stssa_comb(observations, coords, segments, kernel=("b", 2.2), debug=False):
    M1 = stssa_sir(observations, segments)
    M2 = stssa_save(observations, segments)
    M3 = stssa_lcor(observations, coords, segments, kernel=kernel)

    objs = [M1, M2, M3]
    matrices = [M1.m_mat, M2.m_mat, M3.m_mat]

    if debug:
        print("norms of matrices")
        for m in matrices:
            print(np.linalg.norm(m))

    X = np.concatenate(matrices, axis=0)

    result = STSSAResultsObject(m_mat=None, diagonalizer=None, diagonal=None)
    result.aux["stsir"] = objs[0]
    result.aux["stsave"] = objs[1]
    result.aux["stlcor"] = objs[2]

    V, D, it = joint_diagonalization(X, maxiter=1000, eps=1e-6)

    abs_D = np.abs(D)
    diagonal_of_sum_matrix = np.diagonal(sum(abs_D))
    perm = np.argsort(diagonal_of_sum_matrix)[::-1]
    V = V[:, perm]

    result.diagonalizer = V
    result.diagonal = diagonal_of_sum_matrix[perm]
    result.m_mat = V

    return result


