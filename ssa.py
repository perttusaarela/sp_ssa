import numpy as np
from rjdc import joint_diagonalization
from utils import sample_mean, sample_covariance, scaled_local_sample_covariance, ball_kernel_local_sample_covariance, \
    ring_kernel_local_sample_covariance, gaussian_kernel_local_sample_covariance, standardize_data
from functools import partial


def to_projector(mat):
    AtA = (mat.transpose() @ mat)
    if np.linalg.matrix_rank(AtA) != mat.shape[0]:
        inv = np.linalg.pinv(AtA)
    else:
        inv = np.linalg.inv(AtA)
    return mat @ inv @ mat.transpose()


def compare_as_projectors(mat1, mat2):
    assert mat1.shape == mat2.shape
    p1 = to_projector(mat1)
    p2 = to_projector(mat2)
    sub = p1 - p2
    return 0.5 * np.linalg.norm(sub, ord='fro')**2


# This class will be the top level object for all SSA procedures
class SSA:
    def __init__(self, data, num_non_stationary=0):
        white_signals, whitener = standardize_data(data)
        self.data = white_signals
        self.whitener = whitener
        self.nonstationary_dim = num_non_stationary
        self.aux = None

    def set_nonstationary_dim(self, num_non_stationary):
        self.nonstationary_dim = num_non_stationary

    def sir(self, segments):
        res = ssa_sir(self.data, segments)
        ss, ns = res.get_subspaces(whitener=self.whitener, num_non_stationary=self.nonstationary_dim)
        return ss, ns

    def save(self, segments):
        res = ssa_save(self.data, segments)
        ss, ns = res.get_subspaces(whitener=self.whitener, num_non_stationary=self.nonstationary_dim)
        return ss, ns

    def lcor(self, coords, segments, kernel):
        res = ssa_lcor(self.data, coords, segments, kernel)
        ss, ns = res.get_subspaces(whitener=self.whitener, num_non_stationary=self.nonstationary_dim)
        return ss, ns

    def comb(self, coords, segments, kernel):
        res = sp_ssa_comb(self.data, coords, segments, kernel)
        ss, ns = res.get_subspaces(whitener=self.whitener, num_non_stationary=self.nonstationary_dim)
        self.aux = res.aux
        return ss, ns


#  This class is used to store results from SSA algorithms
class SSAResultsObject:
    def __init__(self, m_mat=None, diagonalizer = None, diagonal = None):
        self.m_mat = m_mat  # scatter matrix
        self.diagonalizer = diagonalizer  # matrix of (pseudo-)eigenvectors
        self.diagonal = diagonal  # list of (pseudo-)eigenvalues
        self.aux = {}  # this will be used to store auxiliary results in the SSA_COMB algorithm

    def get_subspaces(self, whitener: np.ndarray, num_non_stationary: int):
        """
        :param whitener: Cov_U(x^(st))^{-1/2}
        :param num_non_stationary: number of non-stationary signals
        :return: (stationary part, non-stationary part) of the estimated unmixing matrix
        """
        V = self.diagonalizer
        ns_eigs = V[:, :num_non_stationary].transpose()
        ss_eigs = V[:, num_non_stationary:].transpose()
        non_stationary_projector = ns_eigs @ whitener
        stationary_projector = ss_eigs @ whitener
        return stationary_projector, non_stationary_projector

    def sort_by_magnitude(self):
        abs_diagonal = np.abs(self.diagonal)
        perm = np.argsort(abs_diagonal)[::-1]
        self.diagonal = self.diagonal[perm]
        self.diagonalizer = self.diagonalizer[:, perm]
        self.m_mat = self.m_mat[np.ix_(perm, perm)]


def ssa_sir(observations, segments):
    full_range = observations.shape[1]

    m_mat = np.zeros((observations.shape[0], observations.shape[0]))  # initialised result matrix
    for segment in segments:  # looping over U_k
        mean_vec = sample_mean(observations, segment)  # m_{U_k}
        mean_mat = np.outer(mean_vec, mean_vec) # m_{U_k}  m_{U_k}^T
        m_mat += (len(segment) / full_range) * mean_mat

    eigvals, eigvecs = np.linalg.eig(m_mat)  # get eigen-stuff
    perm = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, perm]  # sort by eigenvalues, descending order
    eigvals = eigvals[perm]
    result = SSAResultsObject(m_mat=m_mat, diagonalizer=eigvecs, diagonal=eigvals)
    return result


def ssa_save(observations, segments):
    full_range = observations.shape[1]

    m_mat = np.zeros([observations.shape[0], observations.shape[0]])
    for segment in segments:
        #  I_p - Cov_{U_k}(x^{st})
        cov_mat = np.identity(observations.shape[0], like=m_mat) - sample_covariance(observations, segment=segment)
        summand = cov_mat @ cov_mat
        m_mat += (len(segment) / full_range) * summand

    eigvals, eigvecs = np.linalg.eig(m_mat)
    perm = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, perm]
    eigvals = eigvals[perm]
    result = SSAResultsObject(m_mat=m_mat, diagonalizer=eigvecs, diagonal=eigvals)
    return result


def ssa_cor(observations, segments, seg_sizes, lag=1):
    full_range = observations.shape[1]

    m_mat = np.zeros([observations.shape[0], observations.shape[0]])
    full_auto_cov = sample_covariance(observations, lag=lag)
    for idx, segment in enumerate(segments):
        cov_mat1 = sample_covariance(observations, segment=segment, lag=lag)
        summand = full_auto_cov - cov_mat1
        summand = summand @ summand.transpose()
        m_mat += (seg_sizes[idx] / full_range) * summand

    eigvals, eigvecs = np.linalg.eig(m_mat)
    perm = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, perm]
    eigvals = eigvals[perm]

    result = SSAResultsObject(m_mat=m_mat[np.ix_(perm, perm)], diagonalizer=eigvecs, diagonal=eigvals)
    return result


# Note: The auxiliary functions "local_sample_covariance" have been optimized for specific kernels
def ssa_lcor(observations, coords, segments, kernel):
    full_range = observations.shape[1]

    m_mat = np.zeros([observations.shape[0], observations.shape[0]])
    if kernel[0] == "b":
        full_auto_cov = ball_kernel_local_sample_covariance(data=observations, radius=kernel[1], coords=coords)
        func = partial(ball_kernel_local_sample_covariance, data=observations, coords=coords, radius=kernel[1])
    elif kernel[0] == "sb":
        full_auto_cov = scaled_local_sample_covariance(observations, kernel[1], coords)
        func = partial(scaled_local_sample_covariance, data=observations, coords=coords, radius=kernel[1])
    elif kernel[0] == "r":
        full_auto_cov = ring_kernel_local_sample_covariance(observations, coords, kernel[1][0], kernel[1][1])
        func = partial(ring_kernel_local_sample_covariance, data=observations, coords=coords, inner_radius=kernel[1][0],
                       outer_radius=kernel[1][1])
    elif kernel[0] == "g":
        full_auto_cov = gaussian_kernel_local_sample_covariance(observations, coords, kernel[1])
        func = partial(gaussian_kernel_local_sample_covariance, data=observations, coords=coords, radius=kernel[1])
    else:
        full_auto_cov = np.zeros_like(m_mat)
        func = None
        ValueError("kernel must be either 'b', 'sb, 'r', or 'g'")

    for segment in segments:
        cov_mat = func(segment=segment)
        diff = (full_auto_cov - cov_mat)
        sq_diff = diff @ diff
        m_mat += (len(segment) / full_range) * sq_diff

    eigvals, eigvecs = np.linalg.eig(m_mat)

    perm = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, perm]
    eigvals = eigvals[perm]
    result = SSAResultsObject(m_mat=m_mat, diagonalizer=eigvecs, diagonal=eigvals)
    return result


def multi_ssa_lcor(observations, coords, segments, kernels):
    matrices = []
    for kernel in kernels:
        matrices.append(ssa_lcor(observations, coords, segments, kernel).m_mat)

    # Prep matrices from joint diagonalization
    X = np.concatenate(matrices, axis=0)
    # Jointly diagonalize, V is the diagonalizer, and D is a list of diagonal matrices
    V, D, it = joint_diagonalization(X)

    # Permute V such that its eigenvalues are in decreasing order
    diagonal_of_sum_matrix = np.diagonal(sum(D))
    perm = np.argsort(diagonal_of_sum_matrix)[::-1]
    V = V[:, perm]  # V <- V P^T <=> M = V D V^T = V P^T P D P^T P V

    result = SSAResultsObject(m_mat=sum(D)[np.ix_(perm, perm)], diagonalizer=V, diagonal=diagonal_of_sum_matrix[perm])

    return result


def ssa_comb(observations, segments, seg_sizes, lag: int | list[int] =1):
    M1 = ssa_sir(observations, segments).m_mat
    M2 = ssa_save(observations, segments).m_mat
    matrices = [M1, M2]
    if isinstance(lag, list):
        for tau in lag:
            matrices.append(ssa_cor(observations, segments, seg_sizes, lag=tau).m_mat)
    else:
        matrices.append(ssa_cor(observations, segments, seg_sizes, lag=lag).m_mat)


    # Prep matrices from joint diagonalization
    X = np.concatenate(matrices, axis=0)
    # Jointly diagonalize, V is the diagonalizer, and D is a list of diagonal matrices
    V, D, it = joint_diagonalization(X)

    # Permute V such that its eigenvalues are in decreasing order
    diagonal_of_sum_matrix = np.diagonal(sum(D))
    perm = np.argsort(diagonal_of_sum_matrix)[::-1]
    V = V[:, perm]  # V <- V P^T <=> M = V D V^T = V P^T P D P^T P V

    result = SSAResultsObject(m_mat=sum(D)[np.ix_(perm, perm)], diagonalizer=V, diagonal=diagonal_of_sum_matrix)
    return result


def sp_ssa_comb(observations, coords, segments, kernel, debug=False):
    M1 = ssa_sir(observations, segments)
    M2 = ssa_save(observations, segments)
    objs = [M1, M2]
    matrices = [M1.m_mat, M2.m_mat]
    if isinstance(kernel, list):
        for f in kernel:
            objs.append(ssa_lcor(observations, coords, segments, kernel=f))
            matrices.append(objs[-1].m_mat)
    else:
        objs.append(ssa_lcor(observations, coords, segments, kernel=kernel))
        matrices.append(objs[-1].m_mat)

    if debug:
        print("norms of matrices")
        for m in matrices:
            print(np.linalg.norm(m))


    # Prep matrices from joint diagonalization
    X = np.concatenate(matrices, axis=0)

    # initialize result object
    result = SSAResultsObject(m_mat=None, diagonalizer=None, diagonal=None)
    result.aux["spsir"] = objs[0]
    result.aux["spsave"] = objs[1]
    result.aux["splcor"] = objs[2]

    # Jointly diagonalize, V is the diagonalizer, and D is a list of diagonal matrices
    V, D, it = joint_diagonalization(X)
    abs_D = np.abs(D)
    diagonal_of_sum_matrix = np.diagonal(sum(abs_D))
    perm = np.argsort(diagonal_of_sum_matrix)[::-1]
    V = V[:, perm]  # V <- V P^T <=> M = V D V^T = V P^T P D P^T P V
    result.diagonalizer = V
    result.diagonal = diagonal_of_sum_matrix[perm]
    result.m_mat = V
    return result

