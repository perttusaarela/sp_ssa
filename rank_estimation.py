import pickle
import numpy as np
from spatial_settings import (striped_spatial_setting_1, striped_spatial_setting_2, striped_spatial_setting_3,
                              striped_spatial_setting_4, partition_coordinates, get_segments)
from ssa import ssa_sir, ssa_save, ssa_lcor, sp_ssa_comb, SSAResultsObject
from utils import generate_random_orthogonal_matrix, numpy_standardize_data
from functools import partial
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
    return g_vec


def estimate_rank(data, procedure, noise_dim, num_trials):
    g_vec = augmentation_estimator(data, procedure, noise_dim, num_trials)
    return np.argmin(g_vec)


def ssa_procedure(data, coords, sl, ssa_method=sp_ssa_comb, split=(3,3), kernel=('g', 1)):
    part = partition_coordinates(coords, split[0], split[1], sl)
    segs, seg_size = get_segments(part)
    if kernel is None:
        ssa_res = ssa_method(observations=data, segments=segs)
    else:
        ssa_res = ssa_method(observations=data, coords=coords, segments=segs, kernel=kernel)
    ssa_res.sort_by_magnitude()
    return ssa_res.diagonalizer, ssa_res.diagonal


def all_ssa_procedures(data, coords, sl, split=(3,3), kernel=('sb', 3.4)):
    part = partition_coordinates(coords, split[0], split[1], sl)
    segs = get_segments(part)
    res = sp_ssa_comb(data, coords, segs, kernel=kernel)
    res.sort_by_magnitude()
    for method, obj in res.aux.items():
        obj.sort_by_magnitude()

    return res

methods = {
        "spsir": (ssa_sir, None),
        "spsave": (ssa_save, None),
        "splcor": (ssa_lcor, ('sb', 3.4)),
        "spcomb": (sp_ssa_comb, ('sb', 3.4)),
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
        # numpy trickery for computing the norms faster
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

def multi_augmentation_estimator(data, procedure, noise_dim, num_trials, debug=False):
    method = "spsir"
    # Get f_vecs for all methods
    f_vecs = multi_augmented_eigenvector_estimator(data, procedure, noise_dim, num_trials)

    # Compute the procedure for data
    res_obj: SSAResultsObject = procedure(data)

    phi_vecs = {"spcomb": normalized_scree_plot(res_obj.diagonal)}
    if debug:
        print("diagonal for phi_vec for comb: ", res_obj.diagonal)
    for method, obj in res_obj.aux.items():
        if debug:
            print(f"Diagonal for {method}: ", obj.diagonal)
        phi_vecs[method] = normalized_scree_plot(obj.diagonal)

    g_vecs = {
        m: np.cumsum(f_vec) + phi_vecs[m] for m, f_vec in f_vecs.items()
    }

    return g_vecs


def multi_estimate_rank(data, procedure, noise_dim, num_trials, debug=False):
    g_vecs = multi_augmentation_estimator(data, procedure, noise_dim, num_trials, debug=debug)

    return {m: np.argmin(g_vec) for m, g_vec in g_vecs.items()}


class RankStats:
    def __init__(self, rank_stat=None):
        self.counts = {
            m: np.zeros(10, dtype=int) for m in methods.keys()
        }
        if rank_stat is not None:
            for method, rank in rank_stat.items():
                self.counts[method][rank] += 1


    def __add__(self, other):
        for method in self.counts.keys():
            self.counts[method] += other.counts[method]
        return self

    def update(self, rank_stat):
        for method, rank in rank_stat.items():
            self.counts[method][rank] += 1

    def avg(self):
        return {
            m: (np.arange(len(c)) * c).sum() / c.sum() for m, c in self.counts.items()
        }

    def most_common(self):
        return {
            m: np.argmax(c) for m, c in self.counts.items()
        }

    def dump(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.counts, f)

    def __repr__(self):
        return "RankStats()"

    def __str__(self):
        ret = f""
        for methods in self.counts.keys():
            ret += f"{methods}: {self.counts[methods]}\n"

        ret += f"avg: {self.avg()}\n"
        ret += f"choice: {self.most_common()}\n"
        return ret

