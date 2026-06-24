import numpy as np
from spatio_temporal import (generate_spatiotemporal_coordinates, get_unique_spatial_locations, generate_spatiotemporal_data, 
                             partition_spatiotemporal_coordinates, full_spatiotemporal_covariance,  spatial_matern_covariance, 
                             temporal_exponential_covariance, get_segments, params_to_block_vector)

NUM_STATIONARY = 3
NUM_NONSTATIONARY = 2

MEANS = {
    2: [1.5, -1.5],
    3: [1.0, -0.5, 2.0],
    4: [-1.5, -0.5, 0.5, 1.0],
}

VARS = {
    2: [0.4, 1.4],
    3: [3.0, 0.5, 1.5],
    4: [0.4, 0.8, 1.5, 1.2],
}

CORRS = {
    2: [(0.3, 0.5), (1.5, 1.3)],
    3: [(1.0, 1.5), (0.5, 0.8), (2.0, 1.7)],
    4: [(1.6, 1.6), (0.3, 0.3), (2.5, 3.0), (0.8, 3.0)],
}

def rotate_array(arr):
    return [arr[-1]] + arr[:-1]


# To fix that the number of space-time blocks may be larger than the number of parameters prepared
#def repeat_to_length(arr, target_len):
#            reps = (target_len + len(arr) - 1) // len(arr)
#            return (arr * reps)[:target_len]


# Helper function for each of the settings
def get_partition_by_Type(coordinates, num_stripes, setting_Type, side_length, time_length):
    if setting_Type == "space":
        return partition_spatiotemporal_coordinates(coordinates, num_stripes, num_stripes, 1, side_length, time_length)
    elif setting_Type == "time":
        return partition_spatiotemporal_coordinates(coordinates, 1, 1, num_stripes, side_length, time_length)
    elif setting_Type == "space_time":
        return partition_spatiotemporal_coordinates(coordinates, num_stripes, num_stripes, num_stripes, side_length, time_length)
    else:
        raise ValueError("setting_Type must be 'space', 'time', or 'space_time'")
    
# gets the actual numbers of blocks in each direction
def get_partition_dims(num_stripes, setting_Type):
    if setting_Type == "space":
        return num_stripes, num_stripes, 1
    elif setting_Type == "time":
        return 1, 1, num_stripes
    elif setting_Type == "space_time":
        return num_stripes, num_stripes, num_stripes
    else:
        raise ValueError("setting_Type must be 'space', 'time', or 'space_time'")
    

def partition_index_to_grid(idx, nx, ny, nt):
    """
    Recover (ix, iy, it) from the partition list order used in
    partition_spatiotemporal_coordinates.
    """
    per_t = nx * ny # no. of blocks in one time layer
    it = idx // per_t # time index
    rem = idx % per_t # position inside the current time layer
    iy = rem // nx
    ix = rem % nx
    return ix, iy, it


def assign_palette_to_blocks(partition, palette, num_stripes, setting_Type):
    """
    Assign fixed palette (MEAS, VARS, CORRS) values to non-empty blocks so that neighboring blocks differ.
    """
    nx, ny, nt = get_partition_dims(num_stripes, setting_Type)

    assigned = []
    for idx, part in enumerate(partition):
        seg = part[-1]
        if len(seg) == 0:
            continue

        ix, iy, it = partition_index_to_grid(idx, nx, ny, nt) 
        palette_idx = (ix + iy + it) % len(palette)
        assigned.append(palette[palette_idx])

    return assigned


def get_mean_block_params(partition, num_stripes, setting_Type):
    if num_stripes not in MEANS:
        raise ValueError(f"No mean palette defined for num_stripes={num_stripes}")
    return assign_palette_to_blocks(partition, MEANS[num_stripes], num_stripes, setting_Type)


def get_variance_block_params(partition, num_stripes, setting_Type):
    if num_stripes not in VARS:
        raise ValueError(f"No variance palette defined for num_stripes={num_stripes}")
    return assign_palette_to_blocks(partition, VARS[num_stripes], num_stripes, setting_Type)


def get_corr_block_params(partition, num_stripes, setting_Type):
    if num_stripes not in CORRS:
        raise ValueError(f"No covariance palette defined for num_stripes={num_stripes}")
    return assign_palette_to_blocks(partition, CORRS[num_stripes], num_stripes, setting_Type)


# For the simplicity of each of the settings, used these helper functions
def get_stationary_background(num_locations, num_times, side_length, temporal_theta, seed=None):
    if seed is not None:
        np.random.seed(seed)

    coordinates = generate_spatiotemporal_coordinates(num_locations, num_times, side_length)
    num_points = num_locations * num_times

    spatial_points = get_unique_spatial_locations(coordinates)
    cov_mat, spatial_cov, temporal_cov = full_spatiotemporal_covariance(spatial_points, num_times, nu=0.5, phi=1.0, theta=temporal_theta)
    cov_mat += 1e-6 * np.eye(cov_mat.shape[0])
    cholesky = np.linalg.cholesky(cov_mat)

    return coordinates, num_points, cholesky


def get_normalize_signals(signals):
    signals = signals - signals.mean(axis=1, keepdims=True)
    std = signals.std(axis=1, keepdims=True)
    std[std < 1e-10] = 1.0
    return signals / std


def build_subset_covariance(coords_subset, nu, phi, theta):
    coords_subset = np.asarray(coords_subset, dtype=float)

    unique_spatial = np.unique(coords_subset[:, :2], axis=0)
    unique_times = np.unique(coords_subset[:, 2])

    spatial_cov_block = spatial_matern_covariance(unique_spatial, nu=nu, phi=phi)
    temporal_cov_block = temporal_exponential_covariance(unique_times, theta=theta)
    full_cov_block = np.kron(temporal_cov_block, spatial_cov_block)

    spatial_index = {tuple(pt): i for i, pt in enumerate(unique_spatial)}
    time_index = {t: i for i, t in enumerate(unique_times)}

    canonical_indices = []
    num_spatial = len(unique_spatial)

    for row in coords_subset:
        s_idx = spatial_index[(row[0], row[1])]
        t_idx = time_index[row[2]]
        canonical_indices.append(t_idx * num_spatial + s_idx)

    canonical_indices = np.array(canonical_indices, dtype=int)
    cm = full_cov_block[np.ix_(canonical_indices, canonical_indices)]
    cm += 1e-6 * np.eye(cm.shape[0])

    return cm


def simulate_covariance_nonstationary_signal(coordinates, num_points, num_stripes, setting_Type, side_length, time_length, temporal_theta, seed=None):
    partition = get_partition_by_Type(coordinates, num_stripes, setting_Type, side_length, time_length)
    segs = get_segments(partition)
    nonempty_segs = [seg for seg in segs if len(seg) > 0]

    corr_params = get_corr_block_params(partition, num_stripes, setting_Type)

    if len(corr_params) != len(nonempty_segs):
        raise ValueError(
            f"Covariance setting: number of covariance parameters ({len(corr_params)}) "
            f"does not match number of non-empty segments ({len(nonempty_segs)})"
        )

    ns_signal = np.empty(num_points, dtype=float)

    for block_idx, indices in enumerate(nonempty_segs):
        coords_subset = coordinates[indices]
        nu, phi = corr_params[block_idx]

        cm = build_subset_covariance(coords_subset, nu, phi, temporal_theta)
        data = generate_spatiotemporal_data(cm)

        data -= data.mean()
        std = data.std()
        if std > 1e-10:
            data /= std

        ns_signal[indices] = data

    return ns_signal


# Setting 1: Non-stationarity in mean
def spatio_temporal_setting_1(num_locations, num_times, side_length=1, time_length=1, seed=None, temporal_theta=0.5, setting_Type="space", debug=False,):
    coordinates, num_points, cholesky = get_stationary_background(num_locations, num_times, side_length, temporal_theta, seed=seed)

    num_signals = NUM_STATIONARY + NUM_NONSTATIONARY
    Z = np.random.randn(num_points, num_signals)
    signals = (cholesky @ Z).T

    for signal_offset, num_stripes in enumerate(range(2, 4)):
        partition = get_partition_by_Type(coordinates, num_stripes, setting_Type, side_length, time_length)
        segs = get_segments(partition)
        nonempty_segs = [seg for seg in segs if len(seg) > 0]

        block_params = get_mean_block_params(partition, num_stripes, setting_Type)
        #print(f"\nSetting 1, signal {signal_offset}, stripes={num_stripes}")
        #print("Assigned mean block parameters:")
        #print(block_params)


        if len(block_params) != len(nonempty_segs):
            raise ValueError(
                f"Setting 1: number of mean parameters ({len(block_params)}) "
                f"does not match number of non-empty segments ({len(nonempty_segs)})"
            )

        mean_vector = params_to_block_vector(block_params, nonempty_segs)
        signals[NUM_STATIONARY + signal_offset] += mean_vector

    signals = get_normalize_signals(signals)

    if debug:
        print("Means:", np.round(signals.mean(axis=1), 6))
        print("Stds:", np.round(signals.std(axis=1), 6))

    return signals, coordinates


# Setting 2: non-stationarity in variance
def spatio_temporal_setting_2(num_locations, num_times, side_length=1, time_length=1, seed=None, temporal_theta=0.5, setting_Type="space", debug=False,):
    coordinates, num_points, cholesky = get_stationary_background(num_locations, num_times, side_length, temporal_theta, seed=seed)

    num_signals = NUM_STATIONARY + NUM_NONSTATIONARY
    Z = np.random.randn(num_points, num_signals)
    signals = (cholesky @ Z).T

    for signal_offset, num_stripes in enumerate(range(2, 4)):
        partition = get_partition_by_Type(coordinates, num_stripes, setting_Type, side_length, time_length)
        segs = get_segments(partition)
        nonempty_segs = [seg for seg in segs if len(seg) > 0]

        block_params = get_variance_block_params(partition, num_stripes, setting_Type)

        if len(block_params) != len(nonempty_segs):
            raise ValueError(
                f"Setting 2: number of variance parameters ({len(block_params)}) "
                f"does not match number of non-empty segments ({len(nonempty_segs)})"
            )

        variance_vector = params_to_block_vector(block_params, nonempty_segs)
        std_vector = np.sqrt(variance_vector)

        signals[NUM_STATIONARY + signal_offset] *= std_vector

    signals = get_normalize_signals(signals)

    if debug:
        print("Means:", np.round(signals.mean(axis=1), 6))
        print("Stds:", np.round(signals.std(axis=1), 6))

    return signals, coordinates




# Setting 3: Non-stationarity in autocovariance
def spatio_temporal_setting_3(num_locations, num_times, side_length=1, time_length=1, seed=None, temporal_theta=0.5, setting_Type="space_time", debug=False,):
    coordinates, num_points, cholesky = get_stationary_background(num_locations, num_times, side_length, temporal_theta, seed=seed)

    num_signals = NUM_STATIONARY + NUM_NONSTATIONARY
    signals = np.empty((num_signals, num_points), dtype=float)

    # stationary part
    Z = np.random.randn(num_points, NUM_STATIONARY)
    stationary_signals = (cholesky @ Z).T
    signals[:NUM_STATIONARY] = stationary_signals

    # two covariance-nonstationary signals
    for signal_offset, num_stripes in enumerate(range(2, 4)):
        ns_signal = simulate_covariance_nonstationary_signal(
            coordinates=coordinates,
            num_points=num_points,
            num_stripes=num_stripes,
            setting_Type=setting_Type,
            side_length=side_length,
            time_length=time_length,
            temporal_theta=temporal_theta,
        )
        signals[NUM_STATIONARY + signal_offset] = ns_signal

    signals = get_normalize_signals(signals)

    if debug:
        print("Means:", np.round(signals.mean(axis=1), 6))
        print("Vars:", np.round(signals.var(axis=1), 6))

    return signals, coordinates



# Setting 4: Combined mean + variance + covariance
def spatio_temporal_setting_4(num_locations, num_times, side_length=1, time_length=1, seed=None, temporal_theta=0.5, setting_Type="space_time", debug=False,):
    """
    Combined setting contains:
    - 3 stationary signals
    - 1 mean-nonstationary signal from setting 1
    - 1 covariance-nonstationary signal from setting 3
    """
    coordinates, num_points, cholesky = get_stationary_background(
        num_locations, num_times, side_length, temporal_theta, seed=seed
    )

    num_signals = NUM_STATIONARY + NUM_NONSTATIONARY
    signals = np.empty((num_signals, num_points), dtype=float)

    Z = np.random.randn(num_points, NUM_STATIONARY + 1)
    base_signals = (cholesky @ Z).T

    signals[:NUM_STATIONARY] = base_signals[:NUM_STATIONARY]

    # Nonstationary signal 1: mean nonstationary
    mean_signal = base_signals[NUM_STATIONARY].copy()

    mean_num_stripes = 2
    mean_partition = get_partition_by_Type(
        coordinates, mean_num_stripes, setting_Type, side_length, time_length
    )
    mean_segs = get_segments(mean_partition)
    mean_nonempty_segs = [seg for seg in mean_segs if len(seg) > 0]

    mean_params = get_mean_block_params(mean_partition, mean_num_stripes, setting_Type)

    if len(mean_params) != len(mean_nonempty_segs):
        raise ValueError(
            f"Setting 4 mean: number of mean parameters ({len(mean_params)}) "
            f"does not match number of non-empty segments ({len(mean_nonempty_segs)})"
        )

    mean_vector = params_to_block_vector(mean_params, mean_nonempty_segs)
    mean_signal = mean_signal + mean_vector
    signals[NUM_STATIONARY] = mean_signal

    # Nonstationary signal 2: covariance nonstationary
    corr_num_stripes = 3
    cov_signal = simulate_covariance_nonstationary_signal(
        coordinates=coordinates,
        num_points=num_points,
        num_stripes=corr_num_stripes,
        setting_Type=setting_Type,
        side_length=side_length,
        time_length=time_length,
        temporal_theta=temporal_theta,
    )
    signals[NUM_STATIONARY + 1] = cov_signal

    signals = get_normalize_signals(signals)

    if debug:
        print("Means:", np.round(signals.mean(axis=1), 6))
        print("Vars:", np.round(signals.var(axis=1), 6))

    return signals, coordinates

if __name__ == "__main__":
    sigs, coords = spatio_temporal_setting_1(num_locations=250, num_times=10, side_length=50, time_length=10, setting_Type="space_time")
    print("global mean: ", sigs.mean(axis=1))
    print("global var: ", sigs.var(axis=1))
    for split in range(2, 4):
        print("split", split)
        partition = partition_spatiotemporal_coordinates(coords, split, split, split, 50, 10)
        segs = get_segments(partition)
        for seg in segs:
            print("mean: ", np.round(sigs[:, seg].mean(axis=1), 3))
            print("var: ", np.round(sigs[:, seg].var(axis=1), 3))