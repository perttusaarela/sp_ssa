import numpy as np
from rjdc import joint_diagonalization
from utils import sample_mean, sample_covariance, local_sample_covariance, ball_kernel_local_sample_covariance, \
    ring_kernel_local_sample_covariance, gaussian_kernel_local_sample_covariance
from functools import partial


def to_projector(mat):
    AtA = (mat.transpose() @ mat)
    if np.linalg.matrix_rank(AtA) != mat.shape[0]:
        inv = np.linalg.pinv(AtA)
    else:
        inv = np.linalg.inv(AtA)
    return mat @ inv @ mat.transpose()


def compare_projectors(real_proj, estimated_proj):
    return 0.5 * np.linalg.norm(real_proj - estimated_proj, ord='fro')**2


def compare_as_projectors(mat1, mat2):
    assert mat1.shape == mat2.shape
    p1 = to_projector(mat1)
    p2 = to_projector(mat2)
    sub = p1 - p2
    return 0.5 * np.linalg.norm(sub, ord='fro')**2


# This class will be the top level object for all SSA procedures
class SSA:
    def __init__(self):
        pass


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
        self.m_mat = self.m_mat[np.ix_(perm, perm)],


def uniform_segments(full_range, num_segments):
    seg_size = int(full_range / num_segments)
    segments = []
    seg_sizes = []
    for idx in range(num_segments):
        segments.append(range(idx * seg_size, min((idx + 1) * seg_size, full_range)))
        seg_sizes.append(len(segments[-1]))

    return segments, seg_sizes


def ssa_sir(observations, segments, seg_sizes):

    full_range = observations.shape[1]

    m_mat = np.zeros((observations.shape[0], observations.shape[0]))  # initialised result matrix
    for idx, segment in enumerate(segments):  # looping over U_k
        mean_vec = sample_mean(observations, segment)  # m_{U_k}
        mean_mat = np.outer(mean_vec, mean_vec) # m_{U_k}  m_{U_k}^T
        m_mat += (seg_sizes[idx] / full_range) * mean_mat

    eigvals, eigvecs = np.linalg.eig(m_mat)  # get eigen-stuff
    perm = np.argsort(eigvals)[::-1]

    eigvecs = eigvecs[:, perm]  # sort by eigenvalues, descending order
    eigvals = eigvals[perm]
    result = SSAResultsObject(m_mat=m_mat, diagonalizer=eigvecs, diagonal=eigvals)
    return result


def ssa_save(observations, segments, seg_sizes):
    full_range = observations.shape[1]

    m_mat = np.zeros([observations.shape[0], observations.shape[0]])
    for idx, segment in enumerate(segments):
        #  I_p - Cov_{U_k}(x^{st})
        cov_mat = np.identity(observations.shape[0], like=m_mat) - sample_covariance(observations, segment=segment)
        summand = cov_mat @ cov_mat.transpose()
        m_mat += (seg_sizes[idx] / full_range) * summand

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
def ssa_lcor(observations, coords, segments, seg_sizes, kernel):
    full_range = observations.shape[1]

    m_mat = np.zeros([observations.shape[0], observations.shape[0]])
    if kernel[0] == "b":
        full_auto_cov = ball_kernel_local_sample_covariance(observations, coords, kernel[1])
        func = partial(ball_kernel_local_sample_covariance, data=observations, coords=coords, radius=kernel[1])
    elif kernel[0] == "r":
        full_auto_cov = ring_kernel_local_sample_covariance(observations, coords, kernel[1][1], kernel[1][2])
        func = partial(ring_kernel_local_sample_covariance, data=observations, coords=coords, inner_radius=kernel[1][1],
                       outer_radius=kernel[1][2])
    elif kernel[0] == "g":
        full_auto_cov = gaussian_kernel_local_sample_covariance(observations, coords, kernel[1])
        func = partial(gaussian_kernel_local_sample_covariance, data=observations, coords=coords, radius=kernel[1])
    else:
        full_auto_cov = np.zeros_like(m_mat)
        func = None
        ValueError("kernel must be either 'b', 'r', or 'g'")

    for idx, segment in enumerate(segments):
        cov_mat = func(segment=segment)
        diff = full_auto_cov - cov_mat
        m_mat += (seg_sizes[idx] / full_range) * diff @ diff


    eigvals, eigvecs = np.linalg.eig(m_mat)

    perm = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, perm]
    eigvals = eigvals[perm]
    result = SSAResultsObject(m_mat=m_mat, diagonalizer=eigvecs, diagonal=eigvals)
    return result


def multi_ssa_lcor(observations, coords, segments, seg_sizes, kernels):
    matrices = []
    for kernel in kernels:
        matrices.append(ssa_lcor(observations, coords, segments, seg_sizes, kernel).m_mat)

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
    M1 = ssa_sir(observations, segments, seg_sizes).m_mat
    M2 = ssa_save(observations, segments, seg_sizes).m_mat
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


def fix_column_signs(B):
    """
    Ensures each column of matrix B has the entry with largest absolute value positive.

    Parameters:
        B (np.ndarray): A 2D NumPy array (e.g., from eigenvector output)

    Returns:
        B_fixed (np.ndarray): Matrix with sign-corrected columns
    """
    B_fixed = B.copy()
    for j in range(B.shape[1]):
        col = B_fixed[:, j]
        max_idx = np.argmax(np.abs(col))
        if col[max_idx] < 0:
            B_fixed[:, j] *= -1  # Flip sign of entire column
    return B_fixed


def sp_ssa_comb(observations, coords, segments, seg_sizes, kernel, scale=True):
    M1 = ssa_sir(observations, segments, seg_sizes)
    M2 = ssa_save(observations, segments, seg_sizes)
    objs = [M1, M2]
    matrices = [M1.m_mat, M2.m_mat]
    if isinstance(kernel, list):
        for f in kernel:
            objs.append(ssa_lcor(observations, coords, segments, seg_sizes, kernel=f))
            matrices.append(objs[-1].m_mat)
    else:
        objs.append(ssa_lcor(observations, coords, segments, seg_sizes, kernel=kernel))
        matrices.append(objs[-1].m_mat)

    if scale:
        for i in range(len(matrices)):
            matrices[i] = matrices[i] / np.max(matrices[i])


    # Prep matrices from joint diagonalization
    X = np.concatenate(matrices, axis=0)
    # Jointly diagonalize, V is the diagonalizer, and D is a list of diagonal matrices

    result = SSAResultsObject(m_mat=None, diagonalizer=None, diagonal=None)
    result.aux["spsir"] = objs[0]
    result.aux["spsave"] = objs[1]
    result.aux["splcor"] = objs[2]

    V, D, it = joint_diagonalization(X)
    abs_D = np.abs(D)
    diagonal_of_sum_matrix = np.diagonal(sum(abs_D))
    perm = np.argsort(diagonal_of_sum_matrix)[::-1]
    V = V[:, perm]  # V <- V P^T <=> M = V D V^T = V P^T P D P^T P V
    result.diagonalizer = V
    result.diagonal = diagonal_of_sum_matrix[perm]
    result.m_mat = V
    return result



def sp_ssa_sum(observations, coords, segments, seg_sizes, kernel, scale=True):
    M1 = ssa_sir(observations, segments, seg_sizes)
    M2 = ssa_save(observations, segments, seg_sizes)
    objs = [M2, M1]
    matrices = [M2.m_mat, M1.m_mat]
    if isinstance(kernel, list):
        for f in kernel:
            objs.append(ssa_lcor(observations, coords, segments, seg_sizes, kernel=f))
            matrices.append(objs[-1].m_mat)
    else:
        objs.append(ssa_lcor(observations, coords, segments, seg_sizes, kernel=kernel))
        matrices.append(objs[-1].m_mat)

    if scale:
        for i in range(len(matrices)):
            matrices[i] = matrices[i] / np.max(matrices[i])

    # Prep matrices from joint diagonalization
    X = sum(matrices)
    eigvals, eigvecs = np.linalg.eig(X)
    perm = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, perm]
    eigvals = eigvals[perm]

    result = SSAResultsObject(m_mat=X[np.ix_(perm, perm)], diagonalizer=eigvecs, diagonal=eigvals)
    result.aux["spsir"] = objs[0]
    result.aux["spsave"] = objs[1]
    result.aux["splcor"] = objs[2]
    return result

    result = SSAResultsObject(m_mat=sum(D)[np.ix_(perm, perm)], diagonalizer=V, diagonal=diagonal_of_sum_matrix[perm])

    return result
