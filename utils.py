import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import ortho_group


def generate_random_orthogonal_matrix(n):
    return ortho_group.rvs(n)


def matrix_square_root(matrix):
    return sqrtm(matrix)


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
    mean = np.mean(data, axis=1)
    cov = np.cov(data, bias=True)
    assert all(np.linalg.eigvals(cov) >= 0)
    for i in range(data.shape[1]):
        data[:, i] -= mean

    sqrt_cov = matrix_square_root(cov)
    return sqrt_cov @ data, sqrt_cov



def ball_kernel(vec, radius):
    return 1 if np.linalg.norm(vec) <= radius else 0


def ring_kernel(vec, inner_radius, outer_radius):
    return 1 if inner_radius < np.linalg.norm(vec) <= outer_radius else 0


gauss_const = 1.6448536269514722  # \Psi^{-1}(0.95)


def gauss_kernel(vec, radius):
    return np.exp(-0.5 * (gauss_const * np.linalg.norm(vec) / radius)**2)


def local_sample_covariance(data, func, coords, segment=None, seg_mean=None):
    if segment is None:
        segment = range(data.shape[1])
    segment = np.array(segment)

    X = data[:, segment]
    C = coords[segment]

    # Compute or reuse mean
    if seg_mean is None:
        seg_mean = np.mean(X, axis=1, keepdims=True)

    X_centered = X - seg_mean  # Shape: (D, N)
    N = X_centered.shape[1]

    # Precompute weights matrix W[u, u'] = func(coords[u] - coords[u'])
    # Use broadcasting to compute all pairwise differences
    diffs = C[:, np.newaxis, :] - C[np.newaxis, :, :]  # shape (N, N, dim)
    weights = np.vectorize(func, signature='(d)->()')(diffs)  # shape (N, N)

    # Remove diagonal (u == u') if needed
    np.fill_diagonal(weights, 0.0)

    # Compute covariance: L = (1/N) * sum_{u ≠ u'} w_{u,u'} * (x_u - μ)(x_{u'} - μ)^T
    # This is: X_centered @ W @ X_centered.T
    l_cov = (X_centered @ weights @ X_centered.T) / N

    return l_cov



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
    np.fill_diagonal(mask, 0.0)                         # exclude self-pairs

    # Step 3: Compute weighted covariance using matrix multiplication
    l_cov = (X_centered @ mask @ X_centered.T) / N

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
