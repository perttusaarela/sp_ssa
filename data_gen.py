from functools import partial
import pickle
import os
import sys
import numpy as np
from timeit import default_timer as timer
from spatial import spatial_setting_1, spatial_setting_2, spatial_setting_3, spatial_setting_4

NUM_SIMULATIONS = 200


def sim1(params):
    print(params)
    num_points = params[0]
    seed_idx = params[1]
    side_length = int(np.sqrt(num_points))
    s = np.random.randint(10000)
    seed = (s * NUM_SIMULATIONS * num_points + seed_idx) % (2 ** 32)
    return spatial_setting_1(num_points, side_length, seed=seed)


def sim2(params):
    print(params)
    num_points = params[0]
    seed_idx = params[1]
    side_length = int(np.sqrt(num_points))
    s = np.random.randint(1000000)
    seed = (s * NUM_SIMULATIONS * num_points + seed_idx) % (2 ** 32)
    return spatial_setting_2(num_points, side_length, seed=seed)


def sim3(params):
    print(params)
    num_points = params[0]
    seed_idx = params[1]
    side_length = int(np.sqrt(num_points))
    s = np.random.randint(10000)
    seed = (s * NUM_SIMULATIONS * num_points + seed_idx) % (2 ** 32)
    return spatial_setting_3(num_points, side_length, seed=seed)


def sim4(params):
    print(params)
    num_points = params[0]
    seed_idx = params[1]
    side_length = int(np.sqrt(num_points))
    s = np.random.randint(10000)
    seed = (s * NUM_SIMULATIONS * num_points + seed_idx) % (2 ** 32)
    return spatial_setting_4(num_points, side_length, seed=seed)


def pool_func(sim):
    n_range = range(70, 80, 10)
    sim_range = range(NUM_SIMULATIONS)

    parallel_processes = len(os.sched_getaffinity(0))
    print("num parallel: ", parallel_processes)
    input_params = [(x ** 2, iter_i) for x in n_range for iter_i in sim_range]
    print("inputs: ", input_params)
    pool = Pool(parallel_processes)
    results = pool.map(sim, input_params)

    salt = np.random.randint(1000000)

    return results, salt


def dask_func(sim):
    n_range = range(20, 80, 10)
    sim_range = range(NUM_SIMULATIONS)

    input_params = [(x ** 2, iter_i) for x in n_range for iter_i in sim_range]

    delayed_funcs = []
    for params in input_params:
        delayed_funcs.append(delayed(sim)(params))

    results = compute(delayed_funcs)

    salt = np.random.randint(1000000)

    return results, salt


if __name__ == '__main__':
    from timeit import default_timer as timer

    filename = sys.argv[1]
    idx = int(sys.argv[2])
    np.random.seed(np.random.randint(1000000000))
    import multiprocessing

    multiprocessing.freeze_support()

    print(len(sys.argv))

    if len(sys.argv) > 4:
        project_name = sys.argv[3]
    else:
        project_name = ""

    if idx == 1:
        sim = sim1
        print("Starting data generation for spatial setting 1")
    elif idx == 2:
        sim = sim2
        print("Starting data generation for spatial setting 2")
    elif idx == 3:
        sim = sim3
        print("Starting data generation for spatial setting 3")
    elif idx == 4:
        sim = sim4
        print("Starting data generation for spatial setting 4")
    else:
        raise ValueError

    start = timer()
    res, salt = dask_func(sim)
    end = timer()

    if filename is not None:
        with open(os.path.join(os.getcwd(), filename[:-4] + str(salt) + ".pkl"), "wb") as f:
            pickle.dump(res, f)

    print(end - start)
