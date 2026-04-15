import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle
import seaborn as sns
from matplotlib.patches import Rectangle
from spatio_temporal import (generate_spatiotemporal_coordinates, generate_spatiotemporal_data, get_unique_spatial_locations,
                            full_spatiotemporal_covariance,)
sns.set_palette("colorblind")  # This updates matplotlib too
#from sklearn.gaussian_process.kernels import Matern
# plot settings
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': False,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.7
})

SPLITS = [(2, 2, 2), (3, 3, 3), (4, 4, 4)]
NOISE_DIMS = [1, 5, 10, 15]
METHODS = ["stsir", "stsave", "stlcor", "stcomb", "random"]
MARKERS ={
    "stsir": 'o',
    "stsave": '^',
    "stlcor": 'v',
    "stcomb": '*',
    "random": 'x'
}

def get_method_name(method):
    if method == "stsir":
        return "stSSA-SIR"
    elif method == "stsave":
        return "stSSA-SAVE"
    elif method == "stlcor":
        return "stSSA-LCOR"
    elif method == "stcomb":
        return "stSSA-COMB"
    elif method == "random":
        return "Random"
    return method

def plot_folder(folder, show=True, save=False):
    """
    Iterates over the given folder and plots each file with ssa_plot
    :param folder: A directory with only files of data for ssa_plot
    :param show: Boolean for whether to show the plots
    :param save: Boolean for whether to save the plots
    """
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return
    
    for file in os.listdir(folder):  # for each setting
        filepath = os.path.join(folder, file)
        ssa_plot(filepath, show=show, save=save)


def ssa_plot(file, show=True, save=False):
    """
    Creates ssa plots
    :param file: Pickle file of data given by the format:
        dict(method (str): dict(num_points (int): dict(split (tuple): result (tuple))))
    :param show: Boolean for whether to show the plots
    :param save: Boolean for whether to save the plots
    """
    loc_arr = [100, 400, 900, 1600]
    time_arr = [5, 10, 20]
    sizes = [(l, t) for l in loc_arr for t in time_arr]
    x_axis = [l * t for l, t in sizes]

    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True,
                             figsize=(5.5, 3.5))  # we make three plots, one for each split
    #separated_data_ss = [{x: [] for x in METHODS} for _ in range(len(SPLITS))]
    #separated_data_ns = [{x: [] for x in METHODS} for _ in range(len(SPLITS))]
    with open(file, "rb") as f:
        data = pickle.load(f)

    legend_lines = []
    legend_labels = []

    for col_idx, split in enumerate(SPLITS):
        ax_top = axes[0, col_idx]
        ax_bottom = axes[1, col_idx]

        for method in METHODS:
            if method not in data:
                continue

            stationary_vals = []
            nonstationary_vals = []

            for size in sizes:
                if size not in data[method]:
                    stationary_vals.append(np.nan)
                    nonstationary_vals.append(np.nan)
                    continue

                if split not in data[method][size]:
                    stationary_vals.append(np.nan)
                    nonstationary_vals.append(np.nan)
                    continue

                res = data[method][size][split]
                stationary_vals.append(res[0])
                nonstationary_vals.append(res[1])

            label = get_method_name(method)

            line_top, = ax_top.plot(
                x_axis,
                nonstationary_vals,
                marker=MARKERS[method],
                linestyle="-",
                label=label,
            )

            ax_bottom.plot(
                x_axis,
                stationary_vals,
                marker=MARKERS[method],
                linestyle="-",
                label=label,
            )

            if col_idx == 0:
                legend_lines.append(line_top)
                legend_labels.append(label)

        ax_top.set_title(f"Split = {split}")
        ax_top.set_ylabel(r"$n_{\mathrm{perf}}$")
        ax_bottom.set_ylabel(r"$s_{\mathrm{perf}}$")
        ax_bottom.set_xlabel("Total observations = locations × times")

        ax_top.set_ylim(-0.1, 3.0)
        ax_bottom.set_ylim(-0.1, 3.0)

    fig.legend(
        legend_lines,
        legend_labels,
        loc="lower center",
        ncol=len(legend_labels),
        bbox_to_anchor=(0.5, -0.03),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save:
        base = os.path.splitext(os.path.basename(file))[0]
        save_dir = r"C:\Users\Käyttäjä\Desktop\Masters_degree_in_Mathematics_and_Statistics\Research\sp_ssa\plots"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{base}.pdf"), format="pdf")

    if show:
        plt.show()
    else:
        plt.close()

def compute_cell_stats(x, y, values, x_edges, y_edges, stat="mean"):
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    out = np.full((nx, ny), np.nan)

    for i in range(nx):
        for j in range(ny):
            mask = (
                (x >= x_edges[i]) & (x < x_edges[i + 1]) &
                (y >= y_edges[j]) & (y < y_edges[j + 1])
            )
            if np.any(mask):
                out[i, j] = values[mask].mean() if stat == "mean" else values[mask].var()

    return out


from matplotlib.patches import Rectangle

def spatio_temporal_nonstationarity_example(
    seed=None,
    type_sel="m",
    num_locations=200,
    num_times=6,
    side_length=1,
    time_length=6,
    selected_time=0,
    nx=3,
    ny=3,
    num_buckets=7,
    plot=True,
    save=False,
    save_file=None,
):
    """
    Draw a spatio-temporal illustration plot at one fixed time slice.

    type_sel = "m"  -> mean nonstationarity
    type_sel = "v"  -> variance nonstationarity

    Background box color:
        darker = larger box statistic
        - mean if type_sel == "m"
        - variance if type_sel == "v"

    Point symbols:
        grouped into observed-value ranges
        each range gets its own marker and color
    """
    if seed is not None:
        np.random.seed(seed)

    # --------------------------------------------------
    # 1. Generate spatio-temporal coordinates and one signal
    # --------------------------------------------------
    coords = generate_spatiotemporal_coordinates(
        num_locations, num_times, side_length
    )

    spatial_points = get_unique_spatial_locations(coords)
    full_cov, _, _ = full_spatiotemporal_covariance(
        spatial_points,
        num_times,
        nu=0.5,
        phi=1.0,
        theta=0.5,
    )

    values = generate_spatiotemporal_data(full_cov)

    # --------------------------------------------------
    # 2. Define spatial block edges
    # --------------------------------------------------
    x_edges = np.linspace(0, side_length, nx + 1)
    y_edges = np.linspace(0, side_length, ny + 1)

    # --------------------------------------------------
    # 3. Select one time slice for visualization
    # --------------------------------------------------
    mask_t = coords[:, 2] == selected_time
    coords_t = coords[mask_t]
    values_t = values[mask_t].copy()

    x = coords_t[:, 0]
    y = coords_t[:, 1]

    # --------------------------------------------------
    # 4. Induce blockwise nonstationarity
    # --------------------------------------------------
    if type_sel == "m":
        # blockwise means
        block_vals = np.linspace(-1.5, 1.5, nx * ny)
        stat_type = "mean"
        label_stat = "mean"

    elif type_sel == "v":
        # blockwise variances
        block_vals = np.linspace(0.5, 2.0, nx * ny)
        stat_type = "var"
        label_stat = "variance"

    else:
        raise ValueError("type_sel must be 'm' or 'v'")

    for j in range(ny):
        for i in range(nx):
            block_id = j * nx + i

            mask = (
                (x >= x_edges[i]) & (x < x_edges[i + 1]) &
                (y >= y_edges[j]) & (y < y_edges[j + 1])
            )

            if not np.any(mask):
                continue

            # center inside each block first
            values_t[mask] -= np.mean(values_t[mask])

            if type_sel == "m":
                values_t[mask] += block_vals[block_id]

            else:  # variance case
                std = np.std(values_t[mask])
                if std > 1e-10:
                    values_t[mask] /= std
                values_t[mask] *= np.sqrt(block_vals[block_id])

    # --------------------------------------------------
    # 5. Standardize globally for display
    # --------------------------------------------------
    global_std = np.std(values_t)
    if global_std > 1e-10:
        values_t = (values_t - np.mean(values_t)) / global_std
    else:
        values_t = values_t - np.mean(values_t)

    # --------------------------------------------------
    # 6. Compute box statistics for background shading
    # --------------------------------------------------
    cell_stats = compute_cell_stats(
        x, y, values_t, x_edges, y_edges, stat=stat_type
    )

    stat_min = np.nanmin(cell_stats)
    stat_max = np.nanmax(cell_stats)

    def gray_from_stat(val):
        if np.isnan(val):
            return 0.95
        denom = stat_max - stat_min
        if denom < 1e-12:
            return 0.6
        t = (val - stat_min) / denom
        # larger statistic -> darker box
        return 0.92 - 0.65 * t

    # --------------------------------------------------
    # 7. Bucket observed values into ranges
    # --------------------------------------------------
    bins = np.linspace(values_t.min(), values_t.max(), num_buckets + 1)
    bucket_idx = np.digitize(values_t, bins) - 1
    bucket_idx = np.clip(bucket_idx, 0, num_buckets - 1)

    markers = ['o', 's', '^', 'D', 'P', '*', 'X'][:num_buckets]

    # Use visible colors for the symbol groups
    colors = sns.color_palette("coolwarm", num_buckets)

    # --------------------------------------------------
    # 8. Draw figure
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # background rectangles
    for i in range(nx):
        for j in range(ny):
            rect = Rectangle(
                (x_edges[i], y_edges[j]),
                x_edges[i + 1] - x_edges[i],
                y_edges[j + 1] - y_edges[j],
                facecolor=str(gray_from_stat(cell_stats[i, j])),
                edgecolor="black",
                linewidth=0.6,
            )
            ax.add_patch(rect)

    # scatter points by bucket
    for b, marker in enumerate(markers):
        mask = bucket_idx == b
        if not np.any(mask):
            continue

        label = f"{bins[b]:.1f} – {bins[b + 1]:.1f}"

        ax.scatter(
            x[mask],
            y[mask],
            marker=marker,
            s=40,
            color=colors[b],
            edgecolors="black",
            linewidths=0.4,
            label=label,
        )

    # --------------------------------------------------
    # 9. Axes / legend / title
    # --------------------------------------------------
    ax.set_title(
        f"Spatio-temporal example ({label_stat} nonstationarity), time={selected_time}"
    )
    ax.set_xlim(0, side_length)
    ax.set_ylim(0, side_length)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=True,
        fontsize=9,
        title="Observed value range",
    )

    plt.tight_layout()

    # --------------------------------------------------
    # 10. Save / show
    # --------------------------------------------------
    if save:
        save_dir = r"C:\Users\Käyttäjä\Desktop\Masters_degree_in_Mathematics_and_Statistics\Research\sp_ssa\plots"
        os.makedirs(save_dir, exist_ok=True)

        if save_file is None:
            out_name = f"spatiotemporal_{label_stat}_t{selected_time}.pdf"
            plt.savefig(os.path.join(save_dir, out_name), format="pdf")
        else:
            plt.savefig(save_file, format="pdf")

    if plot:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    spatio_temporal_nonstationarity_example(seed=2343, type_sel="m", nx=2, ny=2, selected_time=0, save=True, plot=False)
    spatio_temporal_nonstationarity_example(seed=2343, type_sel="m", nx=3, ny=3, selected_time=0, save=True, plot=False)
    spatio_temporal_nonstationarity_example(seed=2343, type_sel="v", nx=2, ny=2, selected_time=0, save=True, plot=False)
    spatio_temporal_nonstationarity_example(seed=2343, type_sel="v", nx=3, ny=3, selected_time=0, save=True, plot=False)

    subspace_folder = "data/subspace/results"

    if os.path.exists(subspace_folder):
        plot_folder(subspace_folder, show=False, save=True)
    else:
        print(f"Folder not found: {subspace_folder}")
        print("Run the subspace simulations first, or check where the .pkl result files were saved.")

   
