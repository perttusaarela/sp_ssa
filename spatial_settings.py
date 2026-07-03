import numpy as np
from pyssaBSS.spatial import (generate_coordinates, generate_spatial_data, partition_coordinates,
                               ssa_matern_covariance, params_to_block_vector)
from ps_kernel import get_nonstationary_signals
NUM_STATIONARY = 5

MEANS = [
    [1.5, -1.5],
    [1.0, -0.5, 2.0],
    [-1.5, -0.5, 0.5, 1.0]
]

def rotate_array(arr):
    return [arr[-1]] + arr[:-1]

# Non-stationarity in mean
def spatial_setting_1(num_points, side_length=1, seed=None, mean_params=MEANS, debug=False):
    if seed is not None:
        np.random.seed(seed)
    coordinates = generate_coordinates(num_points, side_length)  # uniform coordinates on [0, side_length]^2
    cov_mat = ssa_matern_covariance(coordinates)
    cholesky = np.linalg.cholesky(cov_mat)
    num_signals = NUM_STATIONARY + 3
    Z = np.random.randn(num_points, num_signals)  # all Gaussian vectors at once
    signals = (cholesky @ Z).T

    # Center each signal
    signals -= signals.mean(axis=1, keepdims=True)

    for num_stripes in range(2, 5):
        param_idx = num_stripes - 2
        partition = partition_coordinates(coordinates, num_stripes, num_stripes, side_length)

        mean_vector = params_to_block_vector(mean_params[param_idx] + rotate_array(mean_params[param_idx]), partition)
        signals[NUM_STATIONARY + param_idx] += mean_vector

    signals = np.vstack(signals)
    signals -= signals.mean(axis=1, keepdims=True)
    signals /= signals.std(axis=1, keepdims=True)
    if debug:
        print(np.mean(signals, axis=1))
        print(np.std(signals, axis=1))

    return np.vstack(signals), coordinates


VARS = [
    [0.4, 1.4],
    [3.0, 0.5, 1.5],
    [0.4, 0.8, 1.5, 1.2]
]
# non-stationarity in variance
def spatial_setting_2(num_points, side_length=1, seed=None, var_params=VARS, debug=False):
    if seed is not None:
        np.random.seed(seed)
    coordinates = generate_coordinates(num_points, side_length)
    cov_mat = ssa_matern_covariance(coordinates)
    cholesky = np.linalg.cholesky(cov_mat)
    # Step 3: Preallocate signals array
    num_signals = NUM_STATIONARY + 3

    # Step 4: Generate stationary signals in batch
    Z = np.random.randn(num_points, num_signals)  # all Gaussian vectors at once
    signals = (cholesky @ Z).T

    for num_stripes in range(2, 5):
        param_idx = num_stripes - 2
        partition = partition_coordinates(coordinates, num_stripes, num_stripes, side_length)

        params = var_params[param_idx] + rotate_array(var_params[param_idx])
        variance = params_to_block_vector(params, partition)
        std_vector = np.sqrt(variance)

        signals[NUM_STATIONARY + param_idx, :] *= std_vector

        if debug:
            for seg in partition:
                print(param_idx, len(seg))
                print(np.var(signals[NUM_STATIONARY:, seg], axis=1))

    # Optional debug prints
    if debug:
        print("Means:", np.round(np.mean(signals, axis=1), 6))
        print("Stds:", np.round(np.std(signals, axis=1), 6))

    signals -= signals.mean(axis=1, keepdims=True)
    signals /= signals.std(axis=1, keepdims=True)

    return signals, coordinates


CORRS = [
    ([(0.3, 0.5), (1.5, 1.3)], [(1.0, 2.0), (0.5, 2.0)]),
    ([(1.0, 1.5), (0.5, 0.8), (2.0, 1.7)], [(0.5, 2.0), (1.0, 2.0), (0.5, 2.0)]),
    ([(1.6, 1.6), (0.3, 0.3), (2.5, 3.0), (0.8, 3.0)], [(0.5, 1.8), (1.0, 3.0), (0.5, 1.2), (0.3, 2.5)])
]


# Non-stationarity in autocovariance
def spatial_setting_3(num_points, side_length=1, seed=None, params_list=CORRS, debug=False):
    if seed is not None:
        np.random.seed(seed)

    # Step 1: Generate coordinates
    coordinates = generate_coordinates(num_points, side_length)

    # Step 2: Covariance and Cholesky for stationary signals
    cov_mat = ssa_matern_covariance(coordinates)
    cholesky = np.linalg.cholesky(cov_mat)

    # Step 3: Preallocate signals array
    num_signals = NUM_STATIONARY + 3
    signals = np.empty((num_signals, num_points))

    # Step 4: Generate stationary signals in batch
    Z = np.random.randn(num_points, NUM_STATIONARY)
    stationary_signals = (cholesky @ Z).T
    signals[:NUM_STATIONARY, :] = stationary_signals

    for num_stripes in range(2, 5):
        partition = partition_coordinates(coordinates, num_stripes, num_stripes, side_length)
        ns_signal = np.empty(num_points)
        param_flag = False
        for stripe_idx, indices in enumerate(partition):
            coords_subset = coordinates[indices]
            if stripe_idx % num_stripes == 0:
                param_flag = not param_flag

            if param_flag:
                nu, phi = params_list[num_stripes - 2][0][stripe_idx % num_stripes]
            else:
                nu, phi = params_list[num_stripes - 2][0][stripe_idx % num_stripes]

            cm =ssa_matern_covariance(coords_subset, nu=nu, phi=phi)
            data = generate_spatial_data(cm)

            # Center & normalize
            data -= data.mean()
            data /= data.std()
            ns_signal[indices] = data

        signals[NUM_STATIONARY + (num_stripes - 2), :] = ns_signal

    # Optional debug
    if debug:
        print("Means:", np.round(signals.mean(axis=1), 6))
        print("Vars:", np.round(signals.var(axis=1), 6))

    signals -= signals.mean(axis=1, keepdims=True)
    signals /= signals.std(axis=1, keepdims=True)

    return signals, coordinates



MEAN_IDX = 0
VAR_IDX = 1
CORR_IDX = 2

PARAMS = [
    MEANS[MEAN_IDX],
    VARS[VAR_IDX],
    CORRS[CORR_IDX],
]

def spatial_setting_4(num_points, side_length=1, seed=None, params_list=PARAMS, debug=False):
    if seed is not None:
        np.random.seed(seed)

    # Step 1: Generate coordinates
    coordinates = generate_coordinates(num_points, side_length)

    # Step 2: Compute covariance and Cholesky
    cov_mat = ssa_matern_covariance(coordinates)
    cholesky = np.linalg.cholesky(cov_mat)

    # Step 3: Preallocate signals array
    num_signals = NUM_STATIONARY + 3  # stationary + mean + variance + autocov
    signals = np.empty((num_signals, num_points))

    # Step 4: Stationary signals (batch)
    Z = np.random.randn(num_points, NUM_STATIONARY + 2)
    stationary_signals = (cholesky @ Z).T
    signals[:NUM_STATIONARY + 2, :] = stationary_signals

    # Step 5: Mean signal (2 stripes)
    partition = partition_coordinates(coordinates, 2 + MEAN_IDX, 2 + MEAN_IDX, side_length)

    mean_params = params_list[0] + rotate_array(params_list[0])
    mean_vector = params_to_block_vector(mean_params, partition)
    signals[NUM_STATIONARY, :] += mean_vector

    # Step 6: Variance signal (3 stripes)
    partition = partition_coordinates(coordinates, 2 + VAR_IDX, 1, side_length)

    var_params = params_list[1] + rotate_array(params_list[1])
    variance = params_to_block_vector(var_params, partition)
    std_vector = np.sqrt(variance)
    for seg in partition:
        signals[NUM_STATIONARY + 1, seg] /= signals[NUM_STATIONARY + 1, seg].std()
    signals[NUM_STATIONARY + 1, :] *= std_vector


    # Step 7: Autocov signal (4 stripes)
    partition = partition_coordinates(coordinates, 2 + CORR_IDX, 2 + CORR_IDX, side_length)

    ns_signal = np.empty(num_points)
    num_stripes = 2 + CORR_IDX
    param_flag = False
    for stripe_idx, indices in enumerate(partition):
        coords_subset = coordinates[indices]
        if stripe_idx % num_stripes == 0:
            param_flag = not param_flag

        if param_flag:
            nu, phi = params_list[num_stripes - 2][0][stripe_idx % num_stripes]
        else:
            nu, phi = params_list[num_stripes - 2][0][stripe_idx % num_stripes]

        cm = ssa_matern_covariance(coords_subset, nu=nu, phi=phi)
        data = generate_spatial_data(cm)

        # Center & normalize
        data -= data.mean()
        data /= data.std()
        ns_signal[indices] = data

    signals[-1] = ns_signal

    # Optional debug
    if debug:
        print("Means:", np.round(signals.mean(axis=1), 6))
        print("Vars:", np.round(signals.var(axis=1), 6))

    signals -= signals.mean(axis=1, keepdims=True)
    signals /= signals.std(axis=1, keepdims=True)

    return signals, coordinates


def ps_kernel_matrix(coords, Sigma, sigma, jitter=1e-6):
    """
    Vectorized Paciorek-Schervish kernel.

    coords: (n, 2)
    Sigma: (n, 2, 2)  positive definite matrices per point
    sigma: (n,)       local standard deviations

    returns:
        K: (n, n) covariance matrix
    """

    n = coords.shape[0]

    # --- pairwise coordinate differences ---
    diff = coords[:, None, :] - coords[None, :, :]   # (n, n, 2)

    # --- pairwise averaged covariance matrices ---
    S_bar = 0.5 * (Sigma[:, None, :, :] + Sigma[None, :, :, :])  # (n, n, 2, 2)

    # --- determinants ---
    det_Si = np.linalg.det(Sigma)                  # (n,)
    det_Sbar = np.linalg.det(S_bar)               # (n, n)

    # --- inverse of S_bar (batched) ---
    S_bar_inv = np.linalg.inv(S_bar)              # (n, n, 2, 2)

    # --- quadratic form: diff^T S^{-1} diff ---
    # shape trick: (n,n,1,2) @ (n,n,2,2) @ (n,n,2,1)
    diff_expanded = diff[..., None]              # (n, n, 2, 1)

    quad = np.matmul(
        np.matmul(diff_expanded.transpose(0, 1, 3, 2), S_bar_inv),
        diff_expanded
    ).squeeze(-1).squeeze(-1)  # (n, n)

    # --- normalization term ---
    norm = (
        (det_Si[:, None] ** 0.25) *
        (det_Si[None, :] ** 0.25) /
        np.sqrt(det_Sbar + jitter)
    )

    # --- final kernel ---
    K = sigma[:, None] * sigma[None, :] * norm * np.exp(-0.5 * quad)

    # enforce symmetry (numerical stability)
    K = 0.5 * (K + K.T)

    # jitter for numerical stability
    K += jitter * np.eye(n)

    return K


def ps_cov(coords):
    n = coords.shape[0]

    # Example spatially varying parameters
    sigma = 1.0 + 0.3 * np.sin(coords[:, 0]) + 0.3 * np.cos(coords[:, 1])

    ell = 0.2 + 0.1 * np.sin(coords[:, 0])

    Sigma = np.zeros((n, 2, 2))
    Sigma[:, 0, 0] = ell ** 2
    Sigma[:, 1, 1] = ell ** 2

    return ps_kernel_matrix(coords, Sigma, sigma)


def spatial_setting_5(num_points, side_length=1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Step 1: Generate coordinates
    coordinates = generate_coordinates(num_points, side_length)

    # Step 2: Covariance and Cholesky for stationary signals
    cov_mat = ssa_matern_covariance(coordinates)
    cholesky = np.linalg.cholesky(cov_mat)

    # Step 3: Preallocate signals array
    num_signals = NUM_STATIONARY + 3
    signals = np.empty((num_signals, num_points))

    # Step 4: Generate stationary signals in batch
    Z = np.random.randn(num_points, num_signals)
    stationary_signals = (cholesky @ Z[:, :NUM_STATIONARY]).T
    signals[:NUM_STATIONARY, :] = stationary_signals


    signals[NUM_STATIONARY:, :] = get_nonstationary_signals(coordinates)

    signals -= signals.mean(axis=1, keepdims=True)
    signals /= signals.std(axis=1, keepdims=True)

    return signals, coordinates
