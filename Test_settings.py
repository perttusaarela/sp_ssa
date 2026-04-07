import numpy as np

from spatio_temporal_settings import (
    spatio_temporal_setting_1,
    spatio_temporal_setting_2,
    spatio_temporal_setting_3,
    spatio_temporal_setting_4,
)

from spatio_temporal import partition_spatiotemporal_coordinates, get_segments
from st_ssa import ST_SSA


def get_test_partition(coords, setting_Type):
    if setting_Type == "space":
        return partition_spatiotemporal_coordinates(
            coords, 2, 2, 1, side_length=1, time_length=20
        )
    elif setting_Type == "time":
        return partition_spatiotemporal_coordinates(
            coords, 1, 1, 2, side_length=1, time_length=20
        )
    else:
        return partition_spatiotemporal_coordinates(
            coords, 2, 2, 2, side_length=1, time_length=20
        )


def check_global_standardization(signals):
    means = signals.mean(axis=1)
    vars_ = signals.var(axis=1)

    print("global means:", np.round(means, 4))
    print("global vars :", np.round(vars_, 4))

    assert np.all(np.abs(means) < 1e-6), "Some signals are not centered properly."
    assert np.all(np.abs(vars_ - 1.0) < 1e-3), "Some signals are not normalized properly."


def check_local_means(signals, nonempty_segments, signal_indices):
    print("Checking local means")
    for idx in signal_indices:
        local_means = np.array([signals[idx, seg].mean() for seg in nonempty_segments])
        print(f"signal {idx} local means:", np.round(local_means, 4))
        print(f"signal {idx} local mean spread:", np.round(local_means.max() - local_means.min(), 4))


def check_local_vars(signals, nonempty_segments, signal_indices):
    print("Checking local variances")
    for idx in signal_indices:
        local_vars = np.array([signals[idx, seg].var() for seg in nonempty_segments])
        print(f"signal {idx} local vars :", np.round(local_vars, 4))
        print(f"signal {idx} local var spread:", np.round(local_vars.max() - local_vars.min(), 4))


def test_one_setting(setting_func, setting_name, setting_Type):
    print(f"\n--- Testing {setting_name}, Type={setting_Type} ---")

    signals, coords = setting_func(
        num_locations=100,
        num_times=20,
        side_length=1,
        time_length=20,
        seed=123,
        Type=setting_Type,
        debug=False,
    )

    print("signals shape:", signals.shape)
    print("coords shape:", coords.shape)

    # shape checks
    assert signals.shape == (5, 2000), f"Unexpected signals shape: {signals.shape}"
    assert coords.shape == (2000, 3), f"Unexpected coords shape: {coords.shape}"

    # global normalization checks
    check_global_standardization(signals)

    # partition checks
    partition = get_test_partition(coords, setting_Type)
    segments = get_segments(partition)
    nonempty_segments = [seg for seg in segments if len(seg) > 0]

    print("total segments:", len(segments))
    print("nonempty segments:", len(nonempty_segments))
    print("segment sizes (first 5):", [len(seg) for seg in nonempty_segments[:5]])

    assert len(nonempty_segments) > 0, "All segments are empty."


    # local nonstationarity checks
    # signal indices 3 and 4 are the nonstationary signals
    if setting_name == "setting_1_mean":
        check_local_means(signals, nonempty_segments, signal_indices=[3, 4])

    elif setting_name == "setting_2_variance":
        check_local_vars(signals, nonempty_segments, signal_indices=[3, 4])

    elif setting_name == "setting_3_covariance":
        print("Setting 3 is covariance-based, so local means/vars may not be the main signal.")
        check_local_means(signals, nonempty_segments, signal_indices=[3, 4])
        check_local_vars(signals, nonempty_segments, signal_indices=[3, 4])

    elif setting_name == "setting_4_combined":
        check_local_means(signals, nonempty_segments, signal_indices=[3, 4])
        check_local_vars(signals, nonempty_segments, signal_indices=[3, 4])

    
    # SSA method checks
    obj = ST_SSA(signals, num_non_stationary=2)

    ss_sir, ns_sir = obj.sir(nonempty_segments)
    print("SIR stationary shape:", ss_sir.shape)
    print("SIR nonstationary shape:", ns_sir.shape)

    ss_save, ns_save = obj.save(nonempty_segments)
    print("SAVE stationary shape:", ss_save.shape)
    print("SAVE nonstationary shape:", ns_save.shape)

    ss_lcor, ns_lcor = obj.lcor(coords, nonempty_segments, kernel=("b", 2.2))
    print("LCOR stationary shape:", ss_lcor.shape)
    print("LCOR nonstationary shape:", ns_lcor.shape)

    ss_comb, ns_comb = obj.comb(coords, nonempty_segments, kernel=("b", 2.2))
    print("COMB stationary shape:", ss_comb.shape)
    print("COMB nonstationary shape:", ns_comb.shape)

    # shape checks for SSA outputs
    assert ss_sir.shape == (3, 5)
    assert ns_sir.shape == (2, 5)

    assert ss_save.shape == (3, 5)
    assert ns_save.shape == (2, 5)

    assert ss_lcor.shape == (3, 5)
    assert ns_lcor.shape == (2, 5)

    assert ss_comb.shape == (3, 5)
    assert ns_comb.shape == (2, 5)

    print("All checks passed!")


if __name__ == "__main__":
    tests = [
        (spatio_temporal_setting_1, "setting_1_mean"),
        (spatio_temporal_setting_2, "setting_2_variance"),
        (spatio_temporal_setting_3, "setting_3_covariance"),
        (spatio_temporal_setting_4, "setting_4_combined"),
    ]

    setting_Types = ["space", "time", "space_time"]

    for func, name in tests:
        for setting_Type in setting_Types:
            test_one_setting(func, name, setting_Type)