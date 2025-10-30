import os
import pickle
from utils import *
from spatial import partition_coordinates, get_segments
from ssa import sp_ssa_comb
import numpy as np
from timeit import default_timer as timer
import sys


def to_projector(mat):
    return mat @ np.linalg.pinv(mat.T @ mat) @ mat.T


def compare_as_projectors(mat1, mat2):
    assert mat1.shape == mat2.shape
    p1 = to_projector(mat1)
    p2 = to_projector(mat2)
    sub = p1 - p2
    return 0.5 * np.linalg.norm(sub, ord='fro')**2


def create_result_dict(n_entries, splits, max_count):
    return {
        method: [{split: np.zeros((2, max_count)) for split in splits} for _ in range(n_entries)]
        for method in ["spsir", "spsave", "splcor", "spcomb", "random"]
    }


def empty_result():
    empty_result = {method: {} for method in ["spsir", "spsave", "splcor", "spcomb", "random"]}
    splits = [(2, 2), (3, 3), (4, 4)]
    for method in empty_result.keys():
        for split in splits:
            empty_result[method][split] = np.array([0.0, 0.0])
    return empty_result


def process_observation(observation):
    splits = [(2, 2), (3, 3), (4, 4)]
    results = {method: {} for method in ["spsir", "spsave", "splcor", "spcomb", "random"]}

    signals = observation[0]
    coordinates = observation[1]
    side_length = int(np.sqrt(coordinates.shape[0]))

    num_non_stationary = 3
    num_stationary = 3
    p = num_stationary + num_non_stationary

    mixing_matrix = generate_random_orthogonal_matrix(p)  # we try to estimate this
    mixed_signals = np.dot(mixing_matrix, signals)  # mix the spatial data

    unmixing_mat = mixing_matrix.transpose()
    proj1 = unmixing_mat[:num_stationary, :]
    proj2 = unmixing_mat[num_stationary:, :]
    p1_T = proj1.T
    p2_T = proj2.T

    r_mat = generate_random_orthogonal_matrix(p)  # random reference matrix

    # standardized data, i.e., Cov^{-1/2}(data - \mu)
    whitened_signals = numpy_standardize_data(mixed_signals)

    ss_base = r_mat[:, :num_stationary].transpose() @ whitened_signals[1]  # random separator for ss
    ns_base = r_mat[:, num_stationary:].transpose() @ whitened_signals[1]

    random_res = np.array(
        (
            compare_as_projectors(p1_T, ss_base.T),  # stationary proj.
            compare_as_projectors(p2_T, ns_base.T)  # non-stat. proj
        )
    )

    for split in splits:
        part = partition_coordinates(coordinates, split[0], split[1], side_length)
        segments, seg_sizes = get_segments(part)
        try:
            res = sp_ssa_comb(whitened_signals[0], coordinates, segments, seg_sizes, ('b', 2))
        except:
            print("max_iter failure")
            return empty_result()

        results["random"][split] = random_res
        for method, obj in res.aux.items():
            ss_mat, ns_mat = obj.get_subspaces(whitened_signals[1], 3)
            try:
                results[method][split] = np.array(
                    (
                        compare_as_projectors(p1_T, ss_mat.transpose()),  # stationary proj.
                        compare_as_projectors(p2_T, ns_mat.transpose())  # non-stat. proj
                    )
                )
            except np.linalg.LinAlgError:
                print("skipping one iteration of {}".format(method))

        ss_mat, ns_mat = res.get_subspaces(whitened_signals[1], 3)
        try:
            results['spcomb'][split] = np.array(
                (
                    compare_as_projectors(p1_T, ss_mat.transpose()),  # stationary proj.
                    compare_as_projectors(p2_T, ns_mat.transpose())  # non-stat. proj
                )
            )
        except np.linalg.LinAlgError:
            print("skipping one iteration of {}".format('spcomb'))

    return results

MAX_COUNT = 20

def pool_process_data(filename):
    splits = [(2, 2), (3, 3), (4, 4)]

    collected_data = {
        "spsir": [{s: np.zeros((2, MAX_COUNT)) for s in splits} for _ in range(6)],
        "spsave": [{s: np.zeros((2, MAX_COUNT)) for s in splits} for _ in range(6)],
        "splcor": [{s: np.zeros((2, MAX_COUNT)) for s in splits} for _ in range(6)],
        "spcomb": [{s: np.zeros((2, MAX_COUNT)) for s in splits} for _ in range(6)],
        "random": [{s: np.zeros((2, MAX_COUNT)) for s in splits} for _ in range(6)]
    }
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        idx = 0
        for num_points, list_of_observations in data.items():
            with Pool(len(os.sched_getaffinity(0))) as pool:
                results = pool.map(process_observation, list_of_observations[:MAX_COUNT])

            for i, sub_result in enumerate(results):
                for method, datum in sub_result.items():
                    for split in splits:
                        collected_data[method][idx][split][:, i] = datum[split]

            idx += 1

    final_result = {
        "spsir": [{s: np.zeros((2, 1)) for s in splits} for _ in range(6)],
        "spsave": [{s: np.zeros((2, 1)) for s in splits} for _ in range(6)],
        "splcor": [{s: np.zeros((2, 1)) for s in splits} for _ in range(6)],
        "spcomb": [{s: np.zeros((2, 1)) for s in splits} for _ in range(6)],
        "random": [{s: np.zeros((2, 1)) for s in splits} for _ in range(6)],
    }

    #eps_vec = np.array([1e-8] * 2)
    for method, data in final_result.items():
        for data_point in range(6):
            for split in splits:
                data_at_hand = collected_data[method][data_point][split]
                non_zero_cols = np.any(data_at_hand > 1e-8, axis=0)
                filtered = data_at_hand[:, non_zero_cols]
                final_result[method][data_point][split] = np.mean(filtered, axis=1)

    return final_result


if __name__ == '__main__':
    idx_str = sys.argv[1]
    #idx_str = "1"
    from multiprocessing import Pool
    cwd = os.getcwd()
    filepath = cwd + "/data/full_data/sim" + idx_str + "_short_alt.pkl"
    start = timer()
    results = pool_process_data(filepath)
    end = timer()
    print(end - start)
    with open(cwd + "/data/full_data/sim" + idx_str + "_short_res.pkl", 'wb') as f:
        pickle.dump(results, f)
