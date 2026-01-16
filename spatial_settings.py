import numpy as np
from spatial import (generate_coordinates, generate_spatial_data, partition_coordinates,
                     ssa_matern_covariance, spatial_data_from_cholesky, sort_by_partition)
import matplotlib.pyplot as plt

NUM_STATIONARY = 5


def get_segments(partition):
    return [part[-1] for part in partition]


def params_to_block_vector(params, segments):
    # Find maximum index to size the array
    max_index = max(max(seg) for seg in segments)
    result = np.zeros(max_index + 1, dtype=float)

    # Assign each segment's parameter to its indices
    for param, seg in zip(params, segments):
        result[np.array(seg)] = param

    return result

MEANS = [
    [1.5, -1.5],
    [1.0, -0.5, 2.0],
    [-1.5, -0.5, 0.5, 1.0]
]

def rotate_array(arr):
    return [arr[-1]] + arr[:-1]

# Non-stationarity in mean
def striped_spatial_setting_1(num_points, side_length=1, seed=None, mean_params=MEANS, debug=False):
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
        segs = get_segments(partition)

        mean_vector = params_to_block_vector(mean_params[param_idx] + rotate_array(mean_params[param_idx]), segs)
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
def striped_spatial_setting_2(num_points, side_length=1, seed=None, var_params=VARS, debug=False):
    if seed is not None:
        np.random.seed(seed)
    coordinates = generate_coordinates(num_points, side_length)
    cov_mat = ssa_matern_covariance(coordinates)
    #cov_mat = np.identity(num_points)
    cholesky = np.linalg.cholesky(cov_mat)
    # Step 3: Preallocate signals array
    num_signals = NUM_STATIONARY + 3
    signals = np.zeros((num_signals, num_points))

    # Step 4: Generate stationary signals in batch
    Z = np.random.randn(num_points, num_signals)  # all Gaussian vectors at once
    signals = (cholesky @ Z).T

    for num_stripes in range(2, 5):
        param_idx = num_stripes - 2
        partition = partition_coordinates(coordinates, num_stripes, num_stripes, side_length)
        segs = get_segments(partition)

        params = var_params[param_idx] + rotate_array(var_params[param_idx])
        variance = params_to_block_vector(params, segs)
        std_vector = np.sqrt(variance)

        # Efficient row-wise multiplication instead of full diagonal matrix
        #scaled_cholesky = cholesky * std_vector[:, None]

        #spatial_data = spatial_data_from_cholesky(scaled_cholesky)
        #spatial_data -= spatial_data.mean()
        signals[NUM_STATIONARY + param_idx, :] *= std_vector

        if debug:
            for seg in segs:
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


def check_data(data, coords, split):
    print("Global mean: ", data.mean(axis=1))
    print("Global variance: ", data.var(axis=1))
    print("Global cov: ", np.cov(data, rowvar=True, bias=True))
    print("Local data for split: ", split)
    for seg in get_segments(partition_coordinates(coords, split[0], split[1], int(np.sqrt(coords.shape[0])))):
        print("mean: ", data[:, seg].mean(axis=1))
        print("var: ", data[:, seg].var(axis=1))
        print("cov: ", np.cov(data[:, seg], rowvar=True, bias=True))


# Non-stationarity in autocovariance
def striped_spatial_setting_3(num_points, side_length=1, seed=None, params_list=CORRS, debug=False):
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
        segs= get_segments(partition)
        ns_signal = np.empty(num_points)
        param_flag = False
        for stripe_idx, indices in enumerate(segs):
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

def striped_spatial_setting_4(num_points, side_length=1, seed=None, params_list=PARAMS, debug=False):
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
    #stationary_signals -= stationary_signals.mean(axis=1, keepdims=True)
    #stationary_signals /= stationary_signals.std(axis=1, keepdims=True)
    signals[:NUM_STATIONARY + 2, :] = stationary_signals

    # Step 5: Mean signal (2 stripes)
    partition = partition_coordinates(coordinates, 2 + MEAN_IDX, 2 + MEAN_IDX, side_length)
    segs= get_segments(partition)

    mean_params = params_list[0] + rotate_array(params_list[0])
    mean_vector = params_to_block_vector(mean_params, segs)
    signals[NUM_STATIONARY, :] += mean_vector

    # Step 6: Variance signal (3 stripes)
    partition = partition_coordinates(coordinates, 2 + VAR_IDX, 1, side_length)
    segs= get_segments(partition)

    var_params = params_list[1] + rotate_array(params_list[1])
    variance = params_to_block_vector(var_params, segs)
    std_vector = np.sqrt(variance)
    for seg in segs:
        signals[NUM_STATIONARY + 1, seg] /= signals[NUM_STATIONARY + 1, seg].std()
    signals[NUM_STATIONARY + 1, :] *= std_vector


    # Step 7: Autocov signal (4 stripes)
    partition = partition_coordinates(coordinates, 2 + CORR_IDX, 2 + CORR_IDX, side_length)
    segs = get_segments(partition)

    ns_signal = np.empty(num_points)
    num_stripes = 2 + CORR_IDX
    param_flag = False
    for stripe_idx, indices in enumerate(segs):
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


if __name__ == "__main__":
    sigs, coords = striped_spatial_setting_3(num_points=2500, side_length=50)
    print("global mean: ", sigs.mean(axis=1))
    print("global var: ", sigs.var(axis=1))
    for split in range(2, 5):
        print("split", split)
        partition = partition_coordinates(coords, split, split, 50)
        segs = get_segments(partition)
        for seg in segs:
            print("mean: ", sigs[:, seg].mean())
            print("var: ", sigs[:, seg].var())

