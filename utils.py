import numpy as np
from scipy.stats import ortho_group


def generate_random_orthogonal_matrix(n):
    return ortho_group.rvs(n)


def matrix_square_root(matrix):
    #sq = sqrtm(matrix)
    eig_vals, eig_vecs = np.linalg.eigh(matrix)
    ret = eig_vecs.T @ np.diag(1 / np.sqrt(eig_vals)) @ eig_vecs
    return ret


def generate_random_invertible_matrix(p):
    candidate = np.random.rand(p, p)
    while np.linalg.matrix_rank(candidate) != p:
        candidate = np.random.rand(p, p)

    return candidate


def sample_mean(data, segment=None):
    if segment is None:
        return np.mean(data, axis=1)

    return np.mean(data[:, segment], axis=1)


def sample_covariance(data, segment=None, seg_mean=None):
    if segment is None:
        segment = range(data.shape[1])
    X = data[:, segment]

    if seg_mean is None:
        seg_mean = X.mean(axis=1, keepdims=True)
    else:
        seg_mean = seg_mean[:, np.newaxis]

    centered = X - seg_mean
    cov = (centered @ centered.T) / X.shape[1]
    return cov


def standardize_data(data, mean=None, cov=None):
    if mean is None and cov is None:
        cov, mean = sample_covariance(data)
    if mean is None:
        mean = sample_mean(data)
    if cov is None:
        cov = sample_covariance(data, seg_mean=mean)

    assert all(np.linalg.eigvals(cov) >= 0)  # pos-def

    for i in range(data.shape[1]):
        data[:, i] -= mean

    sqrt_cov = matrix_square_root(cov)

    return sqrt_cov @ data, sqrt_cov


def numpy_standardize_data(data):
    data -= data.mean(axis=1, keepdims=True)
    cov = np.cov(data, bias=True, rowvar=True)
    eigvals, eigvecs = np.linalg.eig(cov)

    # Compute A^{-1/2}
    sqrt_cov = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

    return sqrt_cov @ data, sqrt_cov


def ball_kernel(vec, radius):
    return 1 if np.linalg.norm(vec) <= radius else 0


def ring_kernel(vec, inner_radius, outer_radius):
    return 1 if inner_radius < np.linalg.norm(vec) <= outer_radius else 0


gauss_const = 1.6448536269514722  # \Psi^{-1}(0.95)


def gauss_kernel(vec, radius):
    return np.exp(-0.5 * (gauss_const * np.linalg.norm(vec) / radius)**2)


def scaled_local_sample_covariance(data, radius, coords, segment=None, seg_mean=None):
    if segment is None:
        segment = np.arange(data.shape[1])
    segment = np.asarray(segment)

    X = data[:, segment]  # (D, N)
    C = coords[segment]  # (N, 2)
    N = X.shape[1]
    D = X.shape[0]

    # Mean
    if seg_mean is None:
        seg_mean = X.mean(axis=1, keepdims=True)

    Xc = X - seg_mean  # (D, N)

    # Pairwise distance mask
    diff = C[:, None, :] - C[None, :, :]
    dist2 = np.sum(diff ** 2, axis=2)
    mask = (dist2 <= radius ** 2)

    # Only consider j > i
    mask = np.triu(mask, k=1)

    l_cov = np.zeros((D, D))

    for i in range(N):
        js = np.where(mask[i])[0]
        counter = js.size
        if counter == 0:
            continue

        Xi = Xc[:, i:i + 1]  # (D, 1)
        Xj = Xc[:, js]  # (D, K)

        Sj = np.sum(Xj, axis=1, keepdims=True)  # (D, 1)

        # Sum_j (Xi*Xjᵀ) + Sum_j (Xj*Xiᵀ)
        tmp = Xi @ Sj.T + Sj @ Xi.T  # (D, D)

        l_cov += tmp / counter  # per-i normalization

    return l_cov / N  # final normalization



def ball_kernel_local_sample_covariance(data, coords, radius, segment=None, seg_mean=None):
    if segment is None:
        segment = range(data.shape[1])
    segment = np.array(segment)

    X = data[:, segment]                    # Shape: (D, N)
    C = coords[segment]                     # Shape: (N, d)
    N = X.shape[1]
    D = X.shape[0]

    # Compute or reuse mean
    if seg_mean is None:
        seg_mean = np.mean(X, axis=1, keepdims=True)  # Shape: (D, 1)
    else:
        seg_mean = seg_mean[:, np.newaxis]            # Ensure shape (D, 1)

    X_centered = X - seg_mean                         # Shape: (D, N)

    # Step 1: Compute pairwise distances (squared)
    diffs = C[:, np.newaxis, :] - C[np.newaxis, :, :]   # (N, N, d)
    sq_dists = np.sum(diffs ** 2, axis=2)               # (N, N)

    # Step 2: Create weights matrix using ball kernel
    mask = (sq_dists <= radius ** 2).astype(float)      # binary weights
    np.fill_diagonal(mask, 0.0)                     # exclude self-pairs

    # Step 3: Compute weighted covariance using matrix multiplication
    l_cov = (X_centered @ mask @ X_centered.T) / (N)

    return l_cov


def ring_kernel_local_sample_covariance(data, coords, inner_radius, outer_radius, segment=None, seg_mean=None):
    if segment is None:
        segment = range(data.shape[1])
    segment = np.array(segment)

    X = data[:, segment]                    # (D, N)
    C = coords[segment]                     # (N, d)
    N = X.shape[1]
    D = X.shape[0]

    # Compute or reuse mean
    if seg_mean is None:
        seg_mean = np.mean(X, axis=1, keepdims=True)  # (D, 1)
    else:
        seg_mean = seg_mean[:, np.newaxis]

    X_centered = X - seg_mean               # (D, N)

    # Step 1: Pairwise squared distances
    diffs = C[:, np.newaxis, :] - C[np.newaxis, :, :]  # (N, N, d)
    sq_dists = np.sum(diffs ** 2, axis=2)              # (N, N)

    # Step 2: Create binary mask for ring kernel
    r2_inner = inner_radius ** 2
    r2_outer = outer_radius ** 2
    mask = ((sq_dists > r2_inner) & (sq_dists <= r2_outer)).astype(float)

    # Step 3: Remove diagonal (u == u')
    np.fill_diagonal(mask, 0.0)

    # Step 4: Weighted covariance calculation
    l_cov = (X_centered @ mask @ X_centered.T) / N

    return l_cov


def gaussian_kernel_local_sample_covariance(data, coords, radius, segment=None, seg_mean=None):
    if segment is None:
        segment = range(data.shape[1])
    segment = np.array(segment)

    X = data[:, segment]  # (D, N)
    C = coords[segment]  # (N, d)
    N = X.shape[1]
    D = X.shape[0]

    # Compute or reuse mean
    if seg_mean is None:
        seg_mean = np.mean(X, axis=1, keepdims=True)  # (D, 1)
    else:
        seg_mean = seg_mean[:, np.newaxis]

    X_centered = X - seg_mean  # (D, N)

    # Step 1: Pairwise squared distances
    diffs = C[:, np.newaxis, :] - C[np.newaxis, :, :]  # (N, N, d)
    sq_dists = np.sum(diffs ** 2, axis=2)  # (N, N)

    # Step 2: Gaussian kernel weights
    scale = gauss_const ** 2 / (2 * radius ** 2)
    weights = np.exp(-scale * sq_dists)

    # Step 3: Zero diagonal (u == u')
    np.fill_diagonal(weights, 0.0)

    # Step 4: Weighted covariance
    l_cov = (X_centered @ weights @ X_centered.T) / N

    return l_cov
