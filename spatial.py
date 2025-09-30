import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from sklearn.gaussian_process.kernels import Matern
from scipy.spatial.distance import cdist
from collections import defaultdict
from functools import partial
from timeit import default_timer as timer


def generate_coordinates(num_data_points, hi=1):
    return hi * np.random.rand(num_data_points, 2)


def is_in_rectangle_mask(points, corner, height, width):
    """Vectorized check: returns boolean mask of points inside rectangle."""
    x, y = corner
    mask_x = (points[:, 0] >= x) & (points[:, 0] < x + width)
    mask_y = (points[:, 1] >= y) & (points[:, 1] < y + height)
    return mask_x & mask_y

def partition_coordinates(coordinates, num_x_segments, num_y_segments, side_length=1):
    coordinates = np.asarray(coordinates)
    unif_height = side_length / num_y_segments
    unif_width = side_length / num_x_segments

    partition = []
    for iy in range(num_y_segments):
        for ix in range(num_x_segments):
            x0 = ix * unif_width
            y0 = iy * unif_height
            corner = [x0, y0]

            mask = is_in_rectangle_mask(coordinates, corner, unif_height, unif_width)
            indices = np.nonzero(mask)[0]

            partition.append((corner, unif_height, unif_width, indices.tolist()))

    return partition


def sort_by_partition(points, partition):
    permutation = []
    for part in partition:
        permutation.extend(part[-1])

    return points[permutation, :]


def matern_covariance(points, nu=1.5, phi=1.0):
    matern = Matern(length_scale=phi, nu=nu)
    mat = matern(points)
    return mat


def ssa_matern_covariance(points, nu=0.5, phi=1.0, sigma=1.0):
    return sigma * matern_covariance(points, nu=nu, phi=phi * np.sqrt(2 * nu))


def anisotropic_ssa_matern_covariance(points, beta=0.0, r=1.0, nu=0.5, phi=1.0, sigma=1.0):
    assert not np.isclose(0, r)
    B_mat = np.asarray([[np.cos(beta), -np.sin(beta)], [1 / r * np.sin(beta), 1 / r * np.cos(beta)]]).transpose()
    return ssa_matern_covariance(points @ B_mat, nu=nu, phi=phi, sigma=sigma)


def spatial_data_from_cholesky(cholesky):
    n = cholesky.shape[0]
    gaussian_data = np.random.multivariate_normal(np.zeros(n), np.eye(n))
    spatial_data = cholesky @ gaussian_data

    return spatial_data


def generate_spatial_data(covariance_matrix, mean=None):
    if mean is None:
        mean = np.zeros(covariance_matrix.shape[1])

    cholesky = np.linalg.cholesky(covariance_matrix)
    spatial_data = spatial_data_from_cholesky(cholesky) + mean

    return spatial_data


def get_segments(partition):
    segments = []
    seg_sizes = []
    for idx, part in enumerate(partition):
        segment = part[-1]
        seg_sizes.append(len(segment))
        segments.append(segment)
    return segments, seg_sizes

NONSTATIONARY_BLOCKS = [
    [[0,1,2], [3, 6], [4, 7], [5, 8]],
    [[2,5,8], [0,3], [1,4], [6, 7]],
    [[0,1,3,4], [2,5], [6,7], [8]],
    [[0, 1, 3, 4], [2, 5, 8], [6, 7]]
]

def center_of_box(idx):
    height = np.floor(idx / 3)
    width = idx % 3
    return (1 + width * 2) / 6, (1 + height * 2) / 6


def block_centers():
    result = []
    for blocks in NONSTATIONARY_BLOCKS:
        centers = []
        for block in blocks:
            block_centers_w = [center_of_box(idx)[0] for idx in block]
            block_centers_h = [center_of_box(idx)[1] for idx in block]
            centers.append((np.mean(block_centers_w), np.mean(block_centers_h)))
        result.append(np.array(centers))
    return result


CENTERS = block_centers()



def params_to_block_vector(params, block_sizes, block_idx=0):
    vec = np.zeros(sum(block_sizes))
    for idx, param in enumerate(params):
        part = NONSTATIONARY_BLOCKS[block_idx][idx]
        for block in part:
            lower_bound = sum(block_sizes[:block])
            vec[lower_bound:lower_bound+block_sizes[block]] = [param] * block_sizes[block]

    return vec


def segs_by_block(segs, block_idx=0):
    result = []
    for block in NONSTATIONARY_BLOCKS[block_idx]:
        tmp = []
        for idx in block:
            tmp.extend(segs[idx])
        result.append(tmp)

    return result




# Non-stationarity in mean
def spatial_setting_1(num_points, side_length=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    coordinates = generate_coordinates(num_points, side_length)  # uniform coordinates on [0, side_length]^2
    partition = partition_coordinates(coordinates, 3, 3, side_length)
    # coordinates = sort_by_partition(coordinates, partition)
    segs, seg_size = get_segments(partition)
    cov_mat = matern_covariance(coordinates)
    signals = []
    cholesky = np.linalg.cholesky(cov_mat)
    for _ in range(8):
        spatial_data = spatial_data_from_cholesky(cholesky)
        signals.append(spatial_data)

    non_stationary_mean1 = params_to_block_vector([0.25, -0.25, 0.5, -0.5], seg_size, 0)
    non_stationary_mean2 = params_to_block_vector([0.75, -0.75, 1.25, -1.25], seg_size, 1)
    non_stationary_mean3 = params_to_block_vector([-1.5, -2.0, 2.0, 1.5], seg_size, 2)

    #partition = clustering_partition(coordinates, 3, side_length)
    #segs, seg_size = get_segments(partition)
    #non_stationary_mean1 = cluster_param_vectors(num_points, partition, [-1.5, -2.0, 3.5])
    #non_stationary_mean2 = cluster_param_vectors(num_points, partition, [-1.25, 0.75, 0.5])
    #non_stationary_mean3 = cluster_param_vectors(num_points, partition, [-1, -0.5, 1.5])

    signals[5] += non_stationary_mean1
    signals[6] += non_stationary_mean2
    signals[7] += non_stationary_mean3

    return np.vstack(signals), coordinates


# non-stationarity in variance
def spatial_setting_2(num_points, side_length=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    coordinates = generate_coordinates(num_points, side_length)
    partition = partition_coordinates(coordinates, 3, 3, side_length)
    coordinates = sort_by_partition(coordinates, partition)
    segs, seg_size = get_segments(partition)
    cov_mat = matern_covariance(coordinates)
    cholesky = np.linalg.cholesky(cov_mat)
    signals = []
    for _ in range(5):
        spatial_data = spatial_data_from_cholesky(cholesky)
        signals.append(spatial_data)

    variance1 = params_to_block_vector([0.25, 0.5, 0.75, 1.0], seg_size, 0)
    variance2 = params_to_block_vector([0.75, 0.55, 1.25, 1.5], seg_size, 1)
    variance3 = params_to_block_vector([0.5, 1.0, 2.0, 1.5], seg_size, 2)
    vars = [variance1, variance2, variance3]

    for var in vars:
        spatial_data = generate_spatial_data(cov_mat + np.diag(var))
        signals.append(spatial_data)

    return np.vstack(signals), coordinates


# Non-stationarity in autocovariance
def spatial_setting_3(num_points, side_length=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    coordinates = generate_coordinates(num_points, side_length)
    # partition = partition_coordinates(coordinates, 3, 3, side_length)
    partition = clustering_partition(coordinates, 3, side_length)
    # coordinates = sort_by_partition(coordinates, partition)
    segs, seg_size = get_segments(partition)
    cov_mat = ssa_matern_covariance(coordinates, nu=0.5, phi=0.5)
    cholesky = np.linalg.cholesky(cov_mat)
    signals = []
    for _ in range(5):
        spatial_data = spatial_data_from_cholesky(cholesky)
        signals.append(spatial_data)


    segs = [list(seg) for seg in segs]

    params_list = [[(1.2, 1.4), (0.5, 3.0), (0.7, 0.7)],
                   [(1.5, 2.7), (0.7, 1.0), (1.2, 1.9)],
                   [(0.5, 0.5), (1.0, 1.0), (1.0, 2.0)]]
    for params in params_list:
        ns_signal = np.zeros(num_points)
        for idx, seg in enumerate(segs):
            cm = ssa_matern_covariance(coordinates[seg], nu=params[idx][0], phi=params[idx][1])
            data = generate_spatial_data(cm)
            ns_signal[seg] = data

        signals.append(ns_signal)

    return np.vstack(signals), coordinates


def cluster_param_vectors(vec_len, parts, params):
    vec = np.zeros(vec_len)
    for idx, part in enumerate(parts):
        vec[part[-1]] = np.array(params[idx])

    return vec



def spatial_setting_4(num_points, side_length=1, seed=None, clusters=True):
    if seed is not None:
        np.random.seed(seed)
    coordinates = generate_coordinates(num_points, side_length)
    cov_mat = matern_covariance(coordinates)
    cholesky = np.linalg.cholesky(cov_mat)
    signals = []
    for _ in range(6):
        spatial_data = spatial_data_from_cholesky(cholesky)
        signals.append(spatial_data)

    if clusters:
        partition = clustering_partition(coordinates, 3, side_length)
        segs, seg_size = get_segments(partition)
        non_stationary_mean = cluster_param_vectors(num_points, partition, [-1.5, -2.0, 3.5])
        variance = cluster_param_vectors(num_points, partition, [1.2, 0.55, 2.0])
    else:
        partition = partition_coordinates(coordinates, 3, 3, side_length)
        segs, seg_size = get_segments(partition)
        non_stationary_mean = params_to_block_vector([-1.5, -2.0, 2.0, 1.5], seg_size, 0)
        variance = params_to_block_vector([0.2, 0.55, 2.0, 3.0], seg_size, 1)
        partition = clustering_partition(coordinates, 3, side_length)
        segs, seg_size = get_segments(partition)
        segs = [list(seg) for seg in segs]

    signals[5] += non_stationary_mean
    signals.append(generate_spatial_data(cov_mat + np.diag(variance)))

    ns_signal = np.zeros(num_points)
    params_list = [[(1.2, 1.4), (0.5, 3.0), (0.7, 0.7)]]
                   # [(1.5, 2.7), (0.7, 1.0), (1.2, 1.9)]]
                   # [(0.5, 0.5), (1.0, 1.0), (1.0, 2.0)]]
    for params in params_list:
        for idx, seg in enumerate(segs):
            cm = ssa_matern_covariance(coordinates[seg], nu=params[idx][0], phi=params[idx][1])
            data = generate_spatial_data(cm)
            ns_signal[seg] = data

        signals.append(ns_signal)

    signals = np.vstack(signals)

    return signals, coordinates



def clustering_partition(coordinates, num_clusters, side_length):
    cluster_centers = generate_coordinates(num_clusters, side_length)

    # Compute distances between all coordinates and cluster centers
    distances = cdist(coordinates, cluster_centers)  # shape: (N_coords, N_clusters)

    # Find the closest cluster index for each coordinate
    closest_clusters = np.argmin(distances, axis=1)

    # Initialize clusters
    partition_dict = defaultdict(list)
    for idx, cluster_idx in enumerate(closest_clusters):
        partition_dict[cluster_idx].append(idx)

    # Build final partition format: [[center, [indices]], ...]
    partition = [[cluster_centers[i], partition_dict[i]] for i in range(num_clusters)]

    return partition


if __name__ == '__main__':
    num_points = 2000
    sl = int(np.sqrt(num_points))
    np.random.seed(0)
    start = timer()
    part1 = spatial_setting_2(num_points, sl)
    end = timer()
    print(end - start)
