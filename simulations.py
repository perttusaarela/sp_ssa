import pickle
import numpy as np
from spatial_settings import (spatial_setting_1, spatial_setting_2, spatial_setting_3,
                              spatial_setting_4, partition_coordinates)
from spatial import get_segments
from ssa import SSA, compare_as_projectors
from functools import partial
from timeit import default_timer as timer
from utils import generate_random_orthogonal_matrix, standardize_data
from rank_estimation import RankStats, all_ssa_procedures, multi_estimate_rank

METHODS = ["spsir", "spsave", "splcor", "spcomb", "random"]
SPLITS  = [(2, 2), (3, 3), (4, 4)]

def test_func(setup):

    if setup["split"] is None:
        splits = SPLITS
    else:
        splits = [setup["split"]]
    num_stationary = setup["num_stationary"]
    num_non_stationary = setup["num_non_stationary"]
    p = num_stationary + num_non_stationary
    num_tests = setup["num_tests"]
    T = setup["T"]
    seed = setup["seed"]
    side_length = int(np.sqrt(T))
    res_aux = {
        "spsir": {s: np.zeros((2, num_tests)) for s in splits},
        "spsave": {s: np.zeros((2, num_tests)) for s in splits},
        "splcor": {s: np.zeros((2, num_tests)) for s in splits},
        "spcomb": {s: np.zeros((2, num_tests)) for s in splits},
        "random": {s: np.zeros((2, num_tests)) for s in splits},
    }
    results = {
        method: {s: np.zeros((2, num_tests)) for s in splits}
        for method in METHODS
    }
    start = timer()
    func = setup["setting"]
    for i in range(num_tests):
        obs, coords = func(T, side_length, seed=seed)
        mixing_matrix = generate_random_orthogonal_matrix(p)  # rand. invertible p-by-p matrix
        unmixing_mat = mixing_matrix.T
        mixed_signals = mixing_matrix @ obs  # mix signals
        proj1 = unmixing_mat[:num_stationary, :]  # True projectors
        proj2 = unmixing_mat[num_stationary:, :]
        r_mat = generate_random_orthogonal_matrix(p)  # random guess

        for split in splits:
            partition = partition_coordinates(coords, split[0], split[1], side_length=side_length)
            segments = get_segments(partition)

            ssa_obj = SSA(data=mixed_signals, num_non_stationary=num_non_stationary)

            ss_mat, ns_mat = ssa_obj.comb(coords=coords, segments=segments, kernel=("sb", 3.4))
            ss_base = r_mat[:, :num_stationary].T @ ssa_obj.whitener  # baseline guess for ss
            ns_base = r_mat[:, num_stationary:].T @ ssa_obj.whitener  # baseline guess for ns

            if num_stationary > 0:
                res_rand = compare_as_projectors(proj1.T, ss_base.T)  # result of random guess
                res = np.array(compare_as_projectors(proj1.T, ss_mat.T))  # result of algorithm
                results["spcomb"][split][0, i] = res
                results["random"][split][0, i] = res_rand

            if num_non_stationary > 0:
                res = np.array(compare_as_projectors(proj2.T, ns_mat.T))
                res_rand = compare_as_projectors(proj2.T, ns_base.T)

                results["spcomb"][split][1, i] = res
                results["random"][split][1, i] = res_rand

            if ssa_obj.aux is not None:
                for method, obj in ssa_obj.aux.items():
                    ss_mat, ns_mat = obj.get_subspaces(ssa_obj.whitener, num_non_stationary)
                    res = np.array(compare_as_projectors(proj1.T, ss_mat.T))
                    results[method][split][0, i] = res
                    res = np.array(compare_as_projectors(proj2.T, ns_mat.T))
                    results[method][split][1, i] = res

    for method, arr in results.items():
        for s, data_arr in arr.items():
            print("{}: {}: {}".format(method, s, np.mean(data_arr, axis=1)))

    end = timer()
    print("time: ", end - start)
    if setup["file"] is not None:
        with open(setup["file"] + f"_{T}_{setup["idx"]}.pkl", "wb") as f:
            pickle.dump(results, f)


def subspace_simulation(setting: int):
    """
    produces the data for Simulation 1
    """
    setup = {
        "num_stationary": 5,
        "num_non_stationary": 3,
        "num_tests": 200,
        "T": 4900,
        "split" : None,
        "seed": None,
        "setting": spatial_setting_4,
        "file": "data/final/setting4/data",
        "idx": 0
    }
    if setting == 1:
        setup["setting"] = spatial_setting_1
        setup["file"] = "data/subspace/setting1/data"
    elif setting == 2:
        setup["setting"] = spatial_setting_2
        setup["file"] = "data/subspace/setting2/data"
    elif setting == 3:
        setup["setting"] = spatial_setting_3
        setup["file"] = "data/subspace/setting3/data"
    elif setting == 4:
        setup["setting"] = spatial_setting_4
        setup["file"] = "data/subspace/setting4/data"
    else:
        print("invalid setting")
        exit()

    print("Starting full test for setting: ", setup["setting"])
    T_arr = [x**2 for x in range(20, 80, 10)]
    sim_start = timer()
    range_idx = 10
    for idx in range(range_idx):
        setup["idx"] = idx
        print("Starting tests for idx: ", idx)
        idx_start = timer()
        for T in T_arr:
            print(f"Starting {setup["num_tests"]} tests for num_points: {T}")
            setup["T"] = T
            test_func(setup)
        print("Finished tests for idx: ", idx)
        print("Time spent: ", timer() - idx_start)

    print("Finished tests for setting: ", setup["setting"])
    print("Time spent: ", timer() - sim_start)

    print("Combining sub-results to file: ", f"data/subspace/results/setting{setting}.pkl")
    total_tests = range_idx * setup["num_tests"]
    full_data = {
        m: {
            d: {
                s: np.zeros((2, total_tests)) for s in SPLITS
            } for d in T_arr
        } for m in METHODS
    }
    chunk = setup["num_tests"]
    path = f"data/subspace/setting{setting}"
    for num_points in T_arr:
        for idx in range(range_idx):
            file = f"{path}/data_{num_points}_{idx}.pkl"
            with open(file, "rb") as f:
                data = pickle.load(f)
                for m, data_by_split in data.items():
                    for split, data_array in data_by_split.items():
                        full_data[m][num_points][split][:, idx * chunk: (idx + 1) * chunk] = data_array

    # final result will be the means of the full data set
    final_result = {
        m: {
            d: {
                s: full_data[m][d][s].mean(axis=1)
                for s in SPLITS
            }
            for d in T_arr
        }
        for m in METHODS
    }

    with open(f"data/subspace/results/setting{setting}.pkl", "wb") as f:
        pickle.dump(final_result, f)


def rank_simulation():
    """
    Produces the data for simulation 2
    """
    sl = 60
    num_points = sl ** 2
    num_tests = 1
    test_params = [
        {
            "num_tests": num_tests,
            "noise_dim": 1,
            "file": "data/rank/r1",
            "func": spatial_setting_4
        },
        {
            "num_tests": num_tests,
            "noise_dim": 5,
            "file": "data/rank/r5",
            "func": spatial_setting_4
        },
        {
            "num_tests": num_tests,
            "noise_dim": 10,
            "file": "data/rank/r10",
            "func": spatial_setting_4
        },
        {
            "num_tests": num_tests,
            "noise_dim": 15,
            "file": "data/rank/r15",
            "func": spatial_setting_4
        }
    ]
    num_iter = 1
    total_start_time = timer()

    for params in test_params:
        print("Starting test for params: ", params)
        param_start_time = timer()
        for idx in range(num_iter):
            start = timer()
            counts = RankStats()
            noise_dim = params["noise_dim"]
            print("idx: ", idx)
            print("noise_dim: ", noise_dim)
            for _ in range(params["num_tests"]):
                data, coords = params["func"](num_points, sl)

                data = generate_random_orthogonal_matrix(8) @ data

                data, _ = standardize_data(data)

                func = partial(all_ssa_procedures, coords=coords, sl=sl, split=(4, 4), kernel=('sb', 3.4))
                try:
                    res = multi_estimate_rank(data, func, noise_dim, 10, debug=False)
                    counts.update(res)
                except Exception as e:
                    print("skipping one iteration", e)

            print(counts)
            counts.dump(params["file"] + "_" + str(idx) + ".pkl")
            print("time: ", timer() - start)

        print(f"Total time for params '{params["file"]}' : {timer() - param_start_time}")

    print("All done!")
    print("Total execution time: ", timer() - total_start_time)

    print("Combining all sub-results to file: ", "data/rank/results/rank_sim4.pkl")
    methods = METHODS[:-1]

    data_dict = {
        1: {
            m: np.zeros(10, dtype=int) for m in methods
        },
        5: {
            m: np.zeros(10, dtype=int) for m in methods
        },
        10: {
            m: np.zeros(10, dtype=int) for m in methods
        },
        15: {
            m: np.zeros(10, dtype=int) for m in methods
        }
    }
    path = "data/rank"
    for dim in data_dict.keys():
        for idx in range(num_iter):
            file = f"{path}/r{dim}_{idx}.pkl"
            with open(file, "rb") as f:
                data = pickle.load(f)
                for m, data_arr in data.items():
                    data_dict[dim][m] += data_arr

    with open("data/rank/results/rank_sim4.pkl", "wb") as f:
        pickle.dump(data_dict, f)


import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Run rank estimation or source separation simulations."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-r",
        "--rank",
        action="store_true",
        help="Run the rank estimation simulation."
    )
    group.add_argument(
        "-s",
        "--subspace",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run the source separation simulation for setting i ∈ {1, 2, 3, 4}."
    )

    args = parser.parse_args()

    if args.rank:
        rank_simulation()
    elif args.subspace is not None:
        subspace_simulation(args.subspace)


if __name__ == "__main__":
    main()
