import os
import numpy as np
import matplotlib.pyplot as plt

from spatio_temporal_settings import spatio_temporal_setting_1


def plot_maps_for_time_points(signal, coords, times_to_plot, save=False, filename=None):
    fig, axes = plt.subplots(1, len(times_to_plot), figsize=(4 * len(times_to_plot), 4), squeeze=False)

    vmin = np.min(signal)
    vmax = np.max(signal)

    for j, t in enumerate(times_to_plot):
        ax = axes[0, j]
        mask = coords[:, 2] == t

        sc = ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=signal[mask],
            s=25,
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(f"Time t={int(t)}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, label="Value")
    plt.tight_layout()

    if save:
        os.makedirs("plots", exist_ok=True)
        if filename is None:
            filename = "plots/maps_for_time_points.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def plot_time_series_for_locations(signal, coords, location_indices, save=False, filename=None):
    plt.figure(figsize=(7, 4))

    for idx in location_indices:
        x0, y0 = coords[idx, 0], coords[idx, 1]

        mask = (coords[:, 0] == x0) & (coords[:, 1] == y0)
        loc_coords = coords[mask]
        loc_signal = signal[mask]

        order = np.argsort(loc_coords[:, 2])
        times = loc_coords[order, 2]
        values = loc_signal[order]

        plt.plot(times, values, marker='o', label=f"({x0:.1f}, {y0:.1f})")

    plt.xlabel("time")
    plt.ylabel("value")
    plt.title("Time series at selected locations")
    plt.legend()
    plt.tight_layout()

    if save:
        os.makedirs("plots", exist_ok=True)
        if filename is None:
            filename = "plots/time_series_for_locations.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def plot_spatial_mean_and_variance_over_time(signal, coords, save=False, filename=None):
    times = np.unique(coords[:, 2])
    means = []
    variances = []

    for t in times:
        mask = coords[:, 2] == t
        vals = signal[mask]
        means.append(np.mean(vals))
        variances.append(np.var(vals))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(times, means, marker='o')
    axes[0].set_title("Spatial mean over time")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("mean")

    axes[1].plot(times, variances, marker='o')
    axes[1].set_title("Spatial variance over time")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("variance")

    plt.tight_layout()

    if save:
        os.makedirs("plots", exist_ok=True)
        if filename is None:
            filename = "plots/mean_variance_over_time.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    print("starting simulation", flush=True)

   
    signals, coords = spatio_temporal_setting_1(
        num_locations=1000,
        num_times=20,
        side_length=50,
        time_length=20,
        seed=123
    )

    print("simulation done", flush=True)

  
    signal_idx = 5
    signal = signals[signal_idx]

    print("plotting maps", flush=True)
    plot_maps_for_time_points(signal, coords, times_to_plot=[0, 10, 15], save=True)

    print("plotting time series", flush=True)
    first_time_idx = np.where(coords[:, 2] == 0)[0]
    chosen_locations = first_time_idx[[0, 300, 600, 900]]
    plot_time_series_for_locations(signal, coords, chosen_locations, save=True)

    print("plotting mean/variance", flush=True)
    plot_spatial_mean_and_variance_over_time(signal, coords, save=True)

    print("all done", flush=True)