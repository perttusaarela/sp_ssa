import numpy as np
from sklearn.gaussian_process.kernels import Matern
from scipy.special import kv, gamma

NUM_STATIONARY = 5


def generate_spatiotemporal_coordinates(num_locations, num_times, hi=1):
    """
    Generates uniformly random 2-D coordinates, and a time coordinates
    :param num_locations: Number of spatial locations
    :param num_times: Number of time points
    :param hi: bounds of the spatial domain. Sample ares from [0, hi] x [0, hi]
    :return: (num_locations * num_times, 3) numpy array of spatial-temporal coordinates
    """

    spatial = hi * np.random.rand(num_locations, 2)
    
    coordinates = []

    for t in range(num_times):

        for i in range(num_locations):

            coordinates.append([
                spatial[i,0],
                spatial[i,1],
                t
            ])

    return np.array(coordinates)

def get_unique_spatial_locations(coordinates):
    """
    Get the unique spatial locations (x, y), by ignoring time.
    """
    spatial_coords = coordinates[:, :2]
    return np.unique(spatial_coords, axis=0)


def is_in_spatiotemporal_mask(points, corner, height, width, start_t, end_t):
    """Vectorized check: returns boolean mask of points inside a spatial rectangle and time interval.
    :points: array of (N * 3)
    :corner: Bottom-left corner of the rectangle (x,y)
    :height: rectangle height
    :width: rectangle width
    :start_t: starting time
    :end_t: ending time
    """
    x, y = corner
    mask_x = (points[:, 0] >= x) & (points[:, 0] < x + width)
    mask_y = (points[:, 1] >= y) & (points[:, 1] < y + height)
    mask_t = (points[:, 2] >= start_t) & (points[:, 2] < end_t)
    return mask_x & mask_y & mask_t

def partition_spatiotemporal_coordinates(coordinates, num_x_segments, num_y_segments, num_t_segments, side_length=1, time_length=1):
    """
    Partitions spatio-temporal coordinates into num_x_segments, num_y_segments and time segments.
    :param coordinates: a list of spatial–temporal coordinates
    :param num_x_segments: number of splits in x direction
    :param num_y_segments: number of splits in y direction
    :param num_t_segments: number of splits in time direction
    :param side_length: The side length of the spatial domain
    :return: A list where each block contains the indices of
         observations belonging to a spatial–temporal region
    """
    coordinates = np.asarray(coordinates)
    unif_height = side_length / num_y_segments  # so far only uniform partitioning is possible but this could be extended
    unif_width = side_length / num_x_segments
    unif_time = time_length / num_t_segments

    partition = []
    for it in range(num_t_segments):
        start_t = it* unif_time
        end_t = (it + 1)* unif_time

        for iy in range(num_y_segments):
            for ix in range(num_x_segments):
                x0 = ix * unif_width
                y0 = iy * unif_height
                corner = [x0, y0]

                mask = is_in_spatiotemporal_mask(coordinates, corner, unif_height, unif_width, start_t, end_t)
                indices = np.nonzero(mask)[0]

                partition.append((corner, unif_height, unif_width, start_t,end_t, indices.tolist()))

    return partition


def sort_by_partition(points, partition):
    """
    Sorts the list of points so that points in the same partition occupy a range of indices without gaps
    :param points: list of coordinates
    :param partition: a partition given by partition_coordinates
    :return: sorted list
    """
    permutation = []
    new_partition = []
    pointer = 0
    for part in partition:
        indices = part[-1]
        permutation.extend(indices)
        new_partition.append((part[0], part[1], part[2], list(range(pointer, pointer + len(indices)))))
        pointer += len(indices)


    return points[permutation, :], new_partition


def spatial_matern_covariance(spatial_points, nu=0.5, phi=1.0):
    """
    Spatial Matérn covariance:
        C(h) = c_{nu,phi} (||h||/phi)^nu K_nu(||h||/phi)
    :param spatial_points: a list of spatial coordinates
    :param nu: A parameter of Matern kernel (smoothness)
    :param phi: A parameter of Matern kernel (scale)
    :return: Matern covariance matrix for spatial data
    """
    spatial_points = np.asarray(spatial_points, dtype=float)
    diff = spatial_points[:, None, :] - spatial_points[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))

    r = dists / phi
    cov = np.zeros_like(r, dtype=float)

    # for r > 0
    mask = r > 0
    const = 1.0 / (2.0**(nu - 1.0) * gamma(nu))
    cov[mask] = const * (r[mask] ** nu) * kv(nu, r[mask])

    # define the limit at zero as 1
    cov[~mask] = 1.0
    return cov

def temporal_exponential_covariance(num_times, theta=1.0):
    """
    Temporal exponential covariance:
        C(t1, t2) = exp( -|t2 - t1| / theta )
    """
    times = np.arange(num_times)
    diff = np.abs(times[:, None] - times[None, :])
    return np.exp(-diff / theta)

def full_spatiotemporal_covariance(spatial_points, num_times, nu=0.5, phi=1.0, theta=1.0):
    """
    Final covariance:
        full_cov_ST = Cov_T kron Cov_S
    """
    spatial_cov = spatial_matern_covariance(spatial_points, nu=nu, phi=phi)
    temporal_cov = temporal_exponential_covariance(num_times, theta=theta)
    full_cov = np.kron(temporal_cov, spatial_cov)
    return full_cov, spatial_cov, temporal_cov


def spatial_data_from_cholesky(cholesky):
    """
    Spatial data via the Cholesky method
    :param cholesky: A cholesky matrix L
    :return: a list of n data points with covariance L^TL
    """
    n = cholesky.shape[0]
    gaussian_data = np.random.multivariate_normal(np.zeros(n), np.eye(n))   # zero mean vector, Cov = I_n
    spatial_data = cholesky @ gaussian_data

    return spatial_data


def generate_spatiotemporal_data(covariance_matrix, mean=None, jitter=1e-6):
    """
    Generate spatio-temporal data from covariance matrix
    :param covariance_matrix: Positive-definite matrix
    :param mean: a mean vector, must be same size as covariance_matrix
    :return: spatio-temporal data
    """
    if mean is None:
        mean = np.zeros(covariance_matrix.shape[0])  # if mean is not specified, it is assumed to be zero

    cov = np.asarray(covariance_matrix, dtype=float)
    cov = 0.5 * (cov + cov.T)  # enforce symmetry
    cov = cov + jitter * np.eye(cov.shape[0])

    cholesky = np.linalg.cholesky(cov)  # compute the cholseky decomp. of Cov
    n = cov.shape[0]
    z = np.random.randn(n)
    return cholesky @ z + mean


def get_segments(partition):
    """
    Extracts only the indices of each (spatio-temporal) block from partition given by partition_coordinates
    :param partition: a partition given by partition_coordinates
    :return: (list of indices in each block, a list of number of elements in each block)
    """
    return [part[-1] for part in partition]


def params_to_block_vector(params, segments):
    segments = [seg for seg in segments if len(seg) > 0]
    
    if len(segments) == 0:
        raise ValueError("All segments are empty. Check your partitioning arguments and time/space bounds.")

    if len(params) != len(segments):
        raise ValueError(
            f"Number of parameters ({len(params)}) does not match number of non-empty segments ({len(segments)})."
        )

    # Find maximum index to size the array
    max_index = max(max(seg) for seg in segments)
    result = np.zeros(max_index + 1, dtype=float)

    # Assign each segment's parameter to its indices
    for param, seg in zip(params, segments):
        result[np.array(seg)] = param

    return result


def points_in_polygon(points, polygon):
    """
    Determines whether spatial locations lie inside a polygon.
    Here, only spatial coordinates are used.
    """
    points = np.asarray(points)

    if points.size == 0:
        print("Warning: points array is empty!")
        return np.array([], dtype=bool)
    
    polygon = np.asarray(polygon)

    x = points[:, 0]
    y = points[:, 1]

    x1 = polygon[:, 0]
    y1 = polygon[:, 1]
    x2 = np.roll(x1, -1)
    y2 = np.roll(y1, -1)

    dy = y2 - y1
    non_horizontal = dy != 0

    # Only consider non-horizontal edges
    x1 = x1[non_horizontal]
    y1 = y1[non_horizontal]
    x2 = x2[non_horizontal]
    y2 = y2[non_horizontal]
    dy = dy[non_horizontal]

    # Ray casting condition
    cond = ((y1[:, None] > y) != (y2[:, None] > y))

    xinters = (x2[:, None] - x1[:, None]) * (y - y1[:, None]) / dy[:, None] + x1[:, None]

    crossings = cond & (x < xinters)
    return np.sum(crossings, axis=0) % 2 == 1


def partition_points_by_polygons(points, polygons):
    points = np.asarray(points)
    print("Points in partition_points_by_polygons:", points.shape)
    if points.size == 0:
        print("Warning: points array is empty at start!")
        return [], np.array([])
    
    partitions = []
    assigned = np.zeros(len(points), dtype=bool)

    for poly in polygons:
        mask = points_in_polygon(points, poly) & (~assigned)
        idx = np.nonzero(mask)[0]
        partitions.append(idx)
        assigned |= mask

    unassigned = points[~assigned]
    return partitions, unassigned


if __name__ == "__main__":
    coords = generate_spatiotemporal_coordinates(400, 10)
    print(type(coords))
    print(coords.shape)
    print(coords[:5])
    poly = np.asarray([[5, 0], [0, 5], [5, 10], [10, 5]])
    a, b = partition_points_by_polygons(coords, [poly])
    print(a[0].shape)
    print(b.shape)