import pickle

import numpy as np
from spatial import (partition_coordinates, get_segments, non_cluster_spatial_setting_3, spatial_setting_2,
                     spatial_setting_3, non_cluster_spatial_setting_4)
from ssa import ssa_sir, ssa_save, ssa_lcor, sp_ssa_comb, SSAResultsObject
from utils import generate_random_orthogonal_matrix
from functools import partial
from itertools import product
from timeit import default_timer as timer


def augment_data(data, noise_dim):
    dim, num_points = data.shape
    aug_data = np.random.randn(noise_dim, num_points)
    return np.vstack((data, aug_data))


def normalized_scree_plot(eigenvalues):
    abs_eigs = np.abs(eigenvalues)
    normalizer = np.cumsum(abs_eigs)
    phi_vec = np.zeros_like(abs_eigs)
    phi_vec[0] = 1
    phi_vec[1:] = abs_eigs[1:] / normalizer[1:]
    return phi_vec


def augmented_eigenvector_estimator(data, procedure, noise_dim, num_trials):
    dim, num_points = data.shape
    results = np.empty((num_trials, dim))
    for trial in range(num_trials):
        aug_data = augment_data(data, noise_dim)
        eigenvectors, _ = procedure(aug_data)
        v_aug_j = eigenvectors[dim:, :dim]
        norms = np.linalg.norm(v_aug_j, axis=0)**2
        results[trial] = norms

    f_vec = np.sum(results, axis=0)
    return f_vec * (1 / num_trials)


def augmentation_estimator(data, procedure, noise_dim, num_trials):
    f_vec = augmented_eigenvector_estimator(data, procedure, noise_dim, num_trials)
    phi_vec = normalized_scree_plot(procedure(data)[1])
    cum_f_vec = np.cumsum(f_vec)
    g_vec = cum_f_vec + phi_vec

    return g_vec  # , cum_f_vec, phi_vec


def estimate_rank(data, procedure, noise_dim, num_trials):
    g_vec = augmentation_estimator(data, procedure, noise_dim, num_trials)

    return np.argmin(g_vec)


def ssa_procedure(data, coords, sl, ssa_method=sp_ssa_comb, split=(3,3), kernel=('g', 1)):
    part = partition_coordinates(coords, split[0], split[1], sl)
    segs, seg_size = get_segments(part)
    if kernel is None:
        ssa_res = ssa_method(observations=data, segments=segs, seg_sizes=seg_size)
    else:
        ssa_res = ssa_method(observations=data, coords=coords, segments=segs, seg_sizes=seg_size, kernel=kernel)
    ssa_res.sort_by_magnitude()
    return ssa_res.diagonalizer, ssa_res.diagonal


def all_ssa_procedures(data, coords, sl, split=(3,3), kernel=('b', 2)):
    part = partition_coordinates(coords, split[0], split[1], sl)
    segs, seg_size = get_segments(part)
    res = sp_ssa_comb(data, coords, segs, seg_size, kernel=kernel, scale=False)
    res.sort_by_magnitude()
    for method, obj in res.aux.items():
        obj.sort_by_magnitude()

    return res

methods = {
        "spsir": (ssa_sir, None),
        "spsave": (ssa_save, None),
        "splcor": (ssa_lcor, ('b', 2)),
        "spcomb": (sp_ssa_comb, ('b', 2)),
    }

def multi_augmented_eigenvector_estimator(data, proc, noise_dim, num_trials):
    dim, num_points = data.shape
    results = {
        m: np.empty((num_trials, dim)) for m in methods.keys()
    }
    for trial in range(num_trials):
        aug_data = augment_data(data, noise_dim)
        res_obj: SSAResultsObject = proc(aug_data)
        v_aug_j = res_obj.diagonalizer[dim:, :dim]
        norms = np.einsum('ij,ij->j', v_aug_j, v_aug_j)
        results["spcomb"][trial] = norms
        for method, obj in res_obj.aux.items():
            v_aug_j = obj.diagonalizer[dim:, :dim]
            norms = np.einsum('ij,ij->j', v_aug_j, v_aug_j)
            results[method][trial] = norms

    f_vecs = {
        m: (1 / num_trials) * np.sum(res, axis=0) for m, res in results.items()
    }
    return f_vecs


def multi_augmentation_estimator(data, procedure, noise_dim, num_trials):
    # Get f_vecs for all methods
    f_vecs = multi_augmented_eigenvector_estimator(data, procedure, noise_dim, num_trials)

    # Compute the procedure for plain data
    res_obj: SSAResultsObject = procedure(data)
    phi_vecs = {"spcomb": normalized_scree_plot(res_obj.diagonal)}
    for method, obj in res_obj.aux.items():
        phi_vecs[method] = normalized_scree_plot(obj.diagonal)

    g_vecs = {
        m: np.cumsum(f_vec) + phi_vecs[m] for m, f_vec in f_vecs.items()
    }

    return g_vecs  # , cum_f_vec, phi_vec


def multi_estimate_rank(data, procedure, noise_dim, num_trials):
    g_vecs = multi_augmentation_estimator(data, procedure, noise_dim, num_trials)

    return {m: np.argmin(g_vec) for m, g_vec in g_vecs.items()}



test_points = [1600]
splits = [(2, 2), (3, 3), (4, 4)]
kernel_radius = 2
num_trials = [10, 20]
num_noise_signals = [1, 2] #, 3, 4, 5] #  10, 15] # , 20]
max_count = 10

def process_file(filename):
    ret = []
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        for num_points in test_points:

            sl = int(np.sqrt(num_points))
            data_points = data[num_points][:max_count]
            for obs, coords in data_points:
                mat = generate_random_orthogonal_matrix(6)
                mixed_signals = mat @ obs

                for split in splits:
                    procedure_funcs = {
                        method: partial(ssa_procedure, coords=coords, sl=sl, ssa_method=ssa_fn, split=split,
                                        kernel=kernel)
                        for method, (ssa_fn, kernel) in methods.items()
                    }

                    # Iterate over all combinations of method, trials, and noise dimensions
                    for (method, s), d in product(product(methods.keys(), num_trials), num_noise_signals):
                        func = procedure_funcs[method]
                        res = estimate_rank(mixed_signals, func, d, s)
                        ret.append(res)
                        # Here you should collect or save `res` as needed
    return ret


def fast_process_file(filename):
    ret = []
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        for num_points in test_points:

            sl = int(np.sqrt(num_points))
            data_points = data[num_points][:max_count]
            for obs, coords in data_points:
                mat = generate_random_orthogonal_matrix(6)
                mixed_signals = mat @ obs

                for split in splits:
                    func = partial(all_ssa_procedures, coords=coords, sl=sl, split=split, kernel=('b', 2))
                    for s, d in product(num_trials, num_noise_signals):
                        res = multi_estimate_rank(mixed_signals, func, s, d)
                        ret.append({
                            "num_points": num_points,
                            "split": split,
                            "num_trials": s,
                            "noise_dim": d,
                            "result": res,
                        })

    return ret


from concurrent.futures import ProcessPoolExecutor, as_completed

def process_one_pair(obs, coords, num_points, sl, splits, methods, num_trials_list, num_noise_signals):
    results = []
    mat = generate_random_orthogonal_matrix(6)
    mixed_signal = mat @ obs

    for split in splits:
        procedure_funcs = {
            method: partial(ssa_procedure, coords=coords, sl=sl, ssa_method=ssa_fn, split=split, kernel=kernel)
            for method, (ssa_fn, kernel) in methods.items()
        }

        for method, func in procedure_funcs.items():
            for s in num_trials_list:
                for d in num_noise_signals:
                    try:
                        rank, cum_f, phi, g = estimate_rank(mixed_signal, func, d, s)
                        results.append({
                            'rank': rank,
                            'method': method,
                            'split': split,
                            'trials': s,
                            'noise_dim': d
                        })
                    except Exception as e:
                        results.append({
                            'error': str(e),
                            'method': method,
                            'split': split,
                            'trials': s,
                            'noise_dim': d
                        })

    return results



def process_file_obs_parallel(filename, max_workers=8):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    test_points = [900]
    splits = [(2, 2)] # , (3, 3), (4, 4)]
    kernel_radius = 2
    methods = {
        "spsir": (ssa_sir, None),
        "spsave": (ssa_save, None),
        "splcor": (ssa_lcor, ('b', kernel_radius)),
        "spcomb": (sp_ssa_comb, ('b', kernel_radius)),
    }

    num_trials_list = [10]
    num_noise_signals = [1, 2] # , 10, 15, 20]
    max_examples = 24

    all_tasks = []

    for num_points in test_points:
        sl = int(np.sqrt(num_points))
        data_points = data[num_points][:max_examples]

        for obs, coords in data_points:
            all_tasks.append((obs, coords, num_points, sl, splits, methods, num_trials_list, num_noise_signals))

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_one_pair, *task) for task in all_tasks]

        for future in as_completed(futures):
            results.extend(future.result())

    return results


if __name__ == "__main__":

    for _ in range(5):
        data, coords = non_cluster_spatial_setting_4(1300, 30)
        func = partial(all_ssa_procedures, coords=coords, sl=30, split=(4,4), kernel=('b', 2))
        print(multi_estimate_rank(data, func, 3, 10))

    exit()
    file = "data/full_data/sim2_short.pkl"
    np.random.seed(1)
    #start = timer()
    #process_file(file)
    #end = timer()
    #print(end - start)
    start = timer()
    res = fast_process_file(file)
    print(res)
    end = timer()
    print(end - start)


    exit()

