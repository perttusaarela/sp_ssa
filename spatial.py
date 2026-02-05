import numpy as np
from sklearn.gaussian_process.kernels import Matern

NUM_STATIONARY = 5


def generate_coordinates(num_data_points, hi=1):
    """
    Generates uniformly random 2-D coordinates
    :param num_data_points: Number of data points to generate
    :param hi: bounds for the box. Sample ares from [0, hi] x [0, hi]
    :return: (2 x num_data_points) numpy array of coordinates
    """
    return hi * np.random.rand(num_data_points, 2)


def is_in_rectangle_mask(points, corner, height, width):
    """Vectorized check: returns boolean mask of points inside rectangle."""
    x, y = corner
    mask_x = (points[:, 0] >= x) & (points[:, 0] < x + width)
    mask_y = (points[:, 1] >= y) & (points[:, 1] < y + height)
    return mask_x & mask_y

def partition_coordinates(coordinates, num_x_segments, num_y_segments, side_length=1):
    """
    Partitions coordinates into num_x_segments and num_y_segments segments.
    :param coordinates: a list of coordinates
    :param num_x_segments: number of splits in x direction
    :param num_y_segments: number of splits in y direction
    :param side_length: The side length of the whole area
    :return: A list where each element 4-tuple of the form (lower left corner,
        height of box, width of box, indices of coordinates within box)
        Note that this partitioning style is convenient for some plots
    """
    coordinates = np.asarray(coordinates)
    unif_height = side_length / num_y_segments  # so far only uniform partitioning is possible but this could be extended
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


def matern_covariance(points, nu=1.5, phi=1.0):
    """
    Computes the covariance matrix of a set of points using sklearn Matern kernel. This differs from the usual Matern
    Kernel by a constant
    :param points: a list of coordinates
    :param nu: parameter of Matern kernel
    :param phi: parameter of Matern kernel
    :return: Matern covariance matrix
    """
    matern = Matern(length_scale=phi, nu=nu)
    mat = matern(points)
    return mat


def ssa_matern_covariance(points, nu=0.5, phi=1.0, sigma=1.0):
    """
    Computes the usual Matern covariance matrix of a set of points using sklearn Matern kernel.
    :param points: a list of coordinates
    :param nu: parameter of Matern kernel
    :param phi: parameter of Matern kernel
    :param sigma: variance
    :return: Matern covariance matrix
    """
    return sigma * matern_covariance(points, nu=nu, phi=phi * np.sqrt(2 * nu))


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


def generate_spatial_data(covariance_matrix, mean=None):
    """
    Generate spatial data from covariance matrix
    :param covariance_matrix: Pos-def matrix
    :param mean: a mean vector, must be same size as covariance_matrix
    :return: spatial data
    """
    if mean is None:
        mean = np.zeros(covariance_matrix.shape[1])  # if mean is not specified, it is assumed to be zero

    cholesky = np.linalg.cholesky(covariance_matrix)  # compute the cholseky decomp. of Cov
    spatial_data = spatial_data_from_cholesky(cholesky) + mean

    return spatial_data


def get_segments(partition):
    """
    Extracts only the indices of each block from partition given by partition_coordinates
    :param partition: a partition given by partition_coordinates
    :return: (list of indices in each block, a list of number of elements in each block)
    """
    return [part[-1] for part in partition]


def params_to_block_vector(params, segments):
    # Find maximum index to size the array
    max_index = max(max(seg) for seg in segments)
    result = np.zeros(max_index + 1, dtype=float)

    # Assign each segment's parameter to its indices
    for param, seg in zip(params, segments):
        result[np.array(seg)] = param

    return result


def points_in_polygon(points, polygon):
    points = np.asarray(points)
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
    coords = generate_coordinates(400, 10)
    poly = np.asarray([[5, 0], [0, 5], [5, 10], [10, 5]])
    a, b = partition_points_by_polygons(coords, [poly])
    print(a[0].shape)
    print(b.shape)




