import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle
import seaborn as sns
import matplotlib.colors as mcolors
from pyssaBSS.spatial import generate_coordinates, generate_spatial_data, ssa_matern_covariance
sns.set_palette("colorblind")  # This updates matplotlib too
# plot settings
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 800,
    'savefig.dpi': 800,
    'text.usetex': True,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.7
})

SPLITS = [(2, 2), (3, 3), (4, 4)]
NOISE_DIMS = [1, 5, 10, 15]
METHODS = ["spsir", "spsave", "splcor", "spcomb", "random"]
MARKERS ={
    "spsir": 'o',
    "spsave": '^',
    "splcor": 'v',
    "spcomb": '*',
    "random": 'x'
}

palette = sns.color_palette("colorblind", n_colors=len(METHODS))

COLORS = dict(zip(METHODS, palette))

def lighten(color, amount=0.45):
    r, g, b = mcolors.to_rgb(color)
    return (
        r + (1 - r) * amount,
        g + (1 - g) * amount,
        b + (1 - b) * amount,
    )

def plot_folder(folder, show=True, save=False):
    """
    Iterates over the given folder and plots each file with ssa_plot
    :param folder: A directory with only files of data for ssa_plot
    :param show: Boolean for whether to show the plots
    :param save: Boolean for whether to save the plots
    """
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
    x_axis = [x ** 2 for x in range(20, 80, 10)]

    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True,
                             figsize=(5.5, 3.5))  # we make three plots, one for each split
    separated_data_ss = [{x: [] for x in METHODS} for _ in range(len(SPLITS))]
    separated_data_ns = [{x: [] for x in METHODS} for _ in range(len(SPLITS))]
    with open(file, 'rb') as f:
        data = pickle.load(f)
        for met, method_data in data.items():
            for data_by_num_points in method_data.values():
                for idx, data_by_split in enumerate(data_by_num_points.values()):
                    separated_data_ss[idx][met].append(data_by_split[0])
                    separated_data_ns[idx][met].append(data_by_split[1])

    plot_data = [separated_data_ss, separated_data_ns]
    # Store line objects only once (for legend)
    method_lines = []
    method_labels = []
    print(file)
    for row_idx, row in enumerate(axes):
        for col_idx, ax in enumerate(row):
            for method in METHODS:
                data_to_plot = plot_data[row_idx][col_idx].get(method, None)
                if data_to_plot is None or not data_to_plot:
                    continue
                if 'sp' in method:
                    if 'lcor' in method:
                        method_label = method[:2] + "SSA" + method[3:]
                    else:
                        method_label = method[:2] + "SSA" + method[2:]
                else:
                    method_label = method
                line, = ax.plot(x_axis, data_to_plot, label=method_label,
                                marker=MARKERS[method], linestyle='-')
                # Only collect legend handles from the first subplot
                if row_idx == 0 and col_idx == 0:
                    method_lines.append(line)
                    method_labels.append(method_label)

                ax.set_ylim(-0.2, 3.0)

                if row_idx == 0:
                    ax.set_title(f"{SPLITS[col_idx]}")
                if row_idx == 1 and col_idx == 0:
                    ax.set_ylabel(r"$\mathbf{s}_{\mathrm{perf}}$")
                if col_idx == 0 and row_idx == 0:
                    ax.set_ylabel(r"$\mathbf{n}_{\mathrm{perf}}$")
                if row_idx == 0 and col_idx == 2:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel("Nonstationary", rotation=270, labelpad=15)
                if row_idx == 1 and col_idx == 2:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel("Stationary", rotation=270, labelpad=15)
                if row_idx == 1 and col_idx == 1:
                    ax.set_xlabel("Number of points")

    # Add a single legend to the entire figure
    fig.legend(method_lines, method_labels, loc='lower center', ncol=len(METHODS), bbox_to_anchor=(0.5, -0.025))
    plt.tight_layout(rect=[0.01, 0.01, 1, 0.95])  # Make room for the legend and title
    if save:
        plt.savefig(f"plots/new/sim{file[-13]}.pdf", dpi=800, format='pdf')

    if show:
        plt.show()


def rank_plot(data, save=False, plot=True, save_file=None):
    """
    Creates bar plots for the rank simulations
    :param data: Resulting data from the rank estimation simulations. Is assumed to be of the form
        dict(rank: dict(method: np.array))
    :param save: Boolean indicating whether to save the figure
    :param plot: Boolean indicating whether to show the figure
    :param save_file: A file name to save the figure. Only used if save=True
    """
    methods = METHODS[:-1]             # same as before
    noise_dims = [1, 5, 10, 15]        # same as before

    max_score = 8      # maximum dimension
    center_score = 3   # correct dimension

    score_levels = list(range(max_score + 1))  # 0..<=8

    # Color scheme: neutral center, darker edges
    center_color = np.array([0.85, 0.85, 0.85])  # neutral correct score

    colors = []
    for score in score_levels:

        if score < center_score:  # undershoot -> blue
            # normalize
            d = (center_score - score) / center_score
            # dark blue -> light gray
            blue = np.array([0.2, 0.4, 1.0])  # dark-ish blue
            color = blue * d + center_color * (1 - d)

        elif score > center_score:  # overshoot -> red
            d = (score - center_score) / (max_score - center_score)
            red = np.array([1.0, 0.3, 0.3])  # red tones
            color = red * d + center_color * (1 - d)

        else:  # Correct estimate
            color = center_color

        colors.append(color)

    colors = np.array(colors)


    fig, ax = plt.subplots(figsize=(10, 6))

    # find the proportions for each bar that will be plotted
    proportions = []
    y_labels = []
    real_row_indices = []
    group_midpoints = {}
    y_index = 0
    group_gap = 1

    for r in noise_dims:
        group_start = y_index

        for method in methods:
            scores = data[r][method]
            # Changes computation names to the ones actually used in the paper
            if 'sp' in method:
                if 'lcor' in method:
                    method_label = method[:2] + "SSA" + method[3:]
                else:
                    method_label = method[:2] + "SSA" + method[2:]
            else:
                method_label = method

            if len(scores) == 0:
                continue
            # distribution of scores (0–8)
            counts = np.asarray(scores)  # scores is actually histogram counts
            N = counts.sum()
            prop = counts / N
            proportions.append(prop)
            y_labels.append(method_label)
            real_row_indices.append(y_index)
            y_index += 1

        group_end = y_index
        group_midpoints[r] = (group_start + group_end - 1) / 2

        # spacer row
        proportions.append([0.0] * len(score_levels))
        y_labels.append("")
        y_index += group_gap

    # create the actual plots
    left = np.zeros(len(proportions))
    used_scores = set()

    for i, score_val in enumerate(score_levels):
        widths = np.array([p[score_val] for p in proportions])

        # Check if this score actually appears and only use appeared values in the legend
        if np.any(widths > 0):
            used_scores.add(score_val)
            label = str(score_val)
        else:
            label = None  # do NOT register legend entry

        ax.barh(
            np.arange(len(proportions)),
            widths,
            left=left,
            color=colors[i],
            label=label
        )

        left += widths

    # Y-ticks for real rows only
    ax.set_yticks(real_row_indices)
    ax.set_yticklabels(
        [y_labels[i] for i in real_row_indices],
        rotation=25,
        ha='right',
        va='center',
        fontsize=10
    )

    ax.set_xlabel("Proportion")

    # add noise dim labels on the right
    xlim = ax.get_xlim()
    x_offset = xlim[1] * 1.005
    for r, y in group_midpoints.items():
        ax.text(
            x_offset, y,
            f"r={r}",
            va='center',
            ha='left',
            fontsize=12,
            rotation=-90
        )

    # Collect legend handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Convert labels to ints and filter used scores
    labels_int = np.array([int(l) for l in labels])
    sort_idx = np.argsort(labels_int)

    handles = [handles[i] for i in sort_idx]
    labels = [labels[i] for i in sort_idx]

    ax.legend(
        handles,
        labels,
        title=r"$\hat{q}$",
        bbox_to_anchor=(1.10, 1),
        fontsize=12,
        loc='upper left'
    )

    plt.subplots_adjust(right=0.8)

    if save:
        if save_file is None:
            plt.savefig("plots/plot.pdf", dpi=800, format='pdf')
        else:
            plt.savefig(save_file, dpi=800, format='pdf')

    if plot:
        plt.show()


def generate_means(low=-3, high=3):
    m = np.zeros((3, 3))

    m[0, 0] = np.random.uniform(low, high)
    leftover = -4 * m[0, 0]

    m[1, 1] = np.random.uniform(-abs(leftover), abs(leftover))
    leftover -= m[1, 1]

    m[0, 1] = np.random.uniform(-abs(leftover), abs(leftover)) / 2
    m[1, 0] = (leftover - 2 * m[0, 1]) / 2

    leftover = -m[1, 1] - 2 * m[0, 1]
    m[0, 2] = np.random.uniform(-abs(leftover), abs(leftover)) / 4
    m[1, 2] = (leftover - 4 * m[0, 2]) / 2

    leftover = -m[1, 1] - 2 * m[1, 2]
    m[2, 2] = np.random.uniform(-abs(leftover), abs(leftover)) / 4
    m[2, 1] = (leftover - 4 * m[2, 2]) / 2

    m[2, 0] = -0.5 * (m[2, 1] + m[1, 0]) - 0.25 * m[1, 1]

    return m


def compute_cell_stats(x, y, values, x_edges, y_edges, stat="mean"):
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
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
def nonstationarity_example(seed=None, type_sel="m", nx=3, ny=3, plot=True, save=False, save_file=None):
    if seed is not None:
        np.random.seed(seed)

    points_per_unit = 50
    num_units = 9
    side_length = int(np.sqrt(points_per_unit * num_units))

    coords = np.empty((points_per_unit * num_units, 2))

    # generate coordinates
    for i in range(num_units):
        box = generate_coordinates(points_per_unit, hi=side_length / 3)
        offset = np.array([(i // 3) * side_length / 3,
                           (i % 3) * side_length / 3])
        coords[i * points_per_unit:(i + 1) * points_per_unit] = box + offset

    x, y = coords[:, 0], coords[:, 1]

    # generate values
    cov = ssa_matern_covariance(coords)
    values = generate_spatial_data(cov)

    if type_sel == "m":
        means = generate_means()
        for i in range(num_units):
            sl = slice(i * points_per_unit, (i + 1) * points_per_unit)
            values[sl] -= values[sl].mean()
            values[sl] += means[i // 3, i % 3]
        label_stat = "mean"

    elif type_sel == "v":
        var = np.array([
            [0.25, 1.0, 0.25],
            [1.0, 2.0, 1.0],
            [0.25, 1.0, 0.25]
        ])
        for i in range(num_units):
            sl = slice(i * points_per_unit, (i + 1) * points_per_unit)
            values[sl] -= values[sl].mean()
            values[sl] /= values[sl].std()
            values[sl] *= np.sqrt(var[i // 3, i % 3])
        label_stat = "variance"

    values = (values - values.mean()) / values.std()

    # bucket values
    num_buckets = 7
    bins = np.linspace(values.min(), values.max(), num_buckets + 1)
    bucket_idx = np.digitize(values, bins) - 1

    markers = ['o', 's', '^', 'D', 'P', '*', 'x'][:num_buckets]

    # grid setup
    x_edges = np.linspace(x.min(), x.max(), nx + 1)
    y_edges = np.linspace(y.min(), y.max(), ny + 1)

    stat_type = "mean" if type_sel == "m" else "var"
    cell_stats = compute_cell_stats(x, y, values, x_edges, y_edges, stat=stat_type)

    vmin, vmax = values.min(), values.max()

    def gray(val):
        if not np.isfinite(val):
            return 0.95
        t = np.clip((val - vmin) / (vmax - vmin), 0, 1)
        return 0.95 - 0.85 * t

    # Create actual plots
    fig, ax = plt.subplots(figsize=(7, 6))

    for i in range(nx):
        for j in range(ny):
            ax.add_patch(Rectangle(
                (x_edges[i], y_edges[j]),
                x_edges[i + 1] - x_edges[i],
                y_edges[j + 1] - y_edges[j],
                facecolor=str(gray(cell_stats[i, j])),
                edgecolor='black',
                linewidth=0.5
            ))

    for b, marker in enumerate(markers):
        mask = bucket_idx == b
        if not np.any(mask):
            continue
        edgecolors = 'black' if marker not in ['x', '+', '|', '_'] else None
        ax.scatter(
            x[mask], y[mask],
            marker=marker,
            s=30,
            edgecolors=edgecolors,
            linewidths=0.5,
            label=f'{bins[b]:.1f}–{bins[b + 1]:.1f}'
        )

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 0),
        ncol=4,
        frameon=True,
        fontsize=12,
        markerscale=2,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])

    plt.tight_layout()

    if save:
        if save_file is None:
            plt.savefig(f"plots/partition_plot_{label_stat}_{nx}.pdf", dpi=800, format='pdf')
        else:
            plt.savefig(save_file, dpi=800, format='pdf')

    if plot:
        plt.show()


def new_nonstationarity_example(seed=None, type_sel="m", nx=3, ny=3, plot=True, save=False, save_file=None):
    if seed is not None:
        np.random.seed(seed)

    points_per_unit = 50
    num_units = 9
    side_length = int(np.sqrt(points_per_unit * num_units))

    coords = np.empty((points_per_unit * num_units, 2))

    # generate coordinates
    for i in range(num_units):
        box = generate_coordinates(points_per_unit, hi=side_length / 3)
        offset = np.array([(i // 3) * side_length / 3,
                           (i % 3) * side_length / 3])
        coords[i * points_per_unit:(i + 1) * points_per_unit] = box + offset

    x, y = coords[:, 0], coords[:, 1]

    # generate values
    cov = ssa_matern_covariance(coords)
    values = generate_spatial_data(cov)
    values = (values - values.mean()) / values.std()
    if type_sel == "m":
        means = generate_means()
        for i in range(num_units):
            sl = slice(i * points_per_unit, (i + 1) * points_per_unit)
            values[sl] -= values[sl].mean()
            values[sl] += means[i // 3, i % 3]
        label_stat = "mean"

    elif type_sel == "v":
        var = np.array([
            [0.25, 1.0, 0.25],
            [1.0, 2.0, 1.0],
            [0.25, 1.0, 0.25]
        ])
        for i in range(num_units):
            sl = slice(i * points_per_unit, (i + 1) * points_per_unit)
            values[sl] -= values[sl].mean()
            values[sl] /= values[sl].std()
            values[sl] *= np.sqrt(var[i // 3, i % 3])
        label_stat = "variance"

    values = (values - values.mean()) / values.std()



    # grid setup
    x_edges = np.linspace(x.min(), x.max(), nx + 1)
    y_edges = np.linspace(y.min(), y.max(), ny + 1)

    stat_type = "mean" if type_sel == "m" else "var"
    cell_stats = compute_cell_stats(x, y, values, x_edges, y_edges, stat=stat_type)
    print(stat_type, cell_stats)
    vmin, vmax = values.min(), values.max()

    def gray(val):
        if not np.isfinite(val):
            return 0.95
        t = np.clip((val - vmin) / (vmax - vmin), 0, 1)
        return 0.85 - 0.85 * t

    # colormap and normalizer for points
    cmap = plt.cm.coolwarm
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Create actual plots
    fig, ax = plt.subplots(figsize=(7, 6))

    for i in range(nx):
        for j in range(ny):
            ax.add_patch(Rectangle(
                (x_edges[i], y_edges[j]),
                x_edges[i + 1] - x_edges[i],
                y_edges[j + 1] - y_edges[j],
                facecolor=str(gray(cell_stats[i, j])),
                edgecolor='black',
                linewidth=0.6
            ))

    sc = ax.scatter(
        x, y,
        c=values,
        cmap=cmap,
        norm=norm,
        marker='o',
        s=55,
        edgecolors='black',
        linewidths=0.3,
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=-0.01)
    cbar.outline.set_linewidth(0.3)  # border thickness
    cbar.ax.tick_params(pad=1.5, labelsize=15, width=0.3, length=3.4)
    # cbar.set_label('Value', fontsize=2)

    ax.axis('off')
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.set_xlim(x_edges[0] - 0.5, x_edges[-1] + 0.5)
    ax.set_ylim(y_edges[0] - 0.5, y_edges[-1] + 0.5)

    plt.tight_layout()

    if save:
        if save_file is None:
            plt.savefig(f"plots/partition/partition_plot_{label_stat}_{nx}.pdf", dpi=800, format='pdf')
        else:
            plt.savefig(save_file, dpi=800, format='pdf')

    if plot:
        plt.show()


from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D



class VerticalHandler(HandlerTuple):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height,
                       fontsize, trans):

        top, bottom = orig_handle

        y_top = ydescent + 0.85 * height
        y_bottom = ydescent + 0.07 * height

        artists = []

        for artist, y in zip((top, bottom), (y_top, y_bottom)):
            line = Line2D(
                [xdescent, xdescent + width],
                [y, y],
                color=artist.get_color(),
                linestyle=artist.get_linestyle(),
                marker=artist.get_marker(),
                linewidth=artist.get_linewidth(),
                markersize=artist.get_markersize(),
            )
            line.set_transform(trans)
            artists.append(line)

        return artists


def new_plot_folder(folder, show=True, save=False):
    """
    Creates one (4,3) figure:
        rows    = settings (files)
        columns = partitions
    Each subplot contains
        color     -> method
        solid     -> stationary
        dashed    -> nonstationary
    """

    files = sorted(os.listdir(folder))

    x_axis = [x ** 2 for x in range(20, 80, 10)]

    fig, axes = plt.subplots(
        nrows=len(files),
        ncols=len(SPLITS),
        sharex=True,
        sharey=True,
        figsize=(8, 8)
    )

    # save handles for legend
    method_handles = []

    for row, file in enumerate(files):

        filepath = os.path.join(folder, file)

        separated_data_ss = [{m: [] for m in METHODS} for _ in range(len(SPLITS))]
        separated_data_ns = [{m: [] for m in METHODS} for _ in range(len(SPLITS))]

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        for method, method_data in data.items():
            for point_data in method_data.values():
                for split_idx, split_data in enumerate(point_data.values()):
                    separated_data_ss[split_idx][method].append(split_data[0])
                    separated_data_ns[split_idx][method].append(split_data[1])

        for col, ax in enumerate(axes[row]):

            for method in METHODS:

                if "sp" in method:
                    if "lcor" in method:
                        label = method[:2] + "SSA" + method[3:]
                    else:
                        label = method[:2] + "SSA" + method[2:]
                else:
                    label = method

                base = COLORS[method]

                # Nonstationary
                ax.plot(
                    x_axis,
                    separated_data_ns[col][method],
                    marker=MARKERS[method],
                    linestyle="-",  # change to ":" if you want dotted
                    color=base,
                )

                # Stationary
                line, = ax.plot(
                    x_axis,
                    separated_data_ss[col][method],
                    marker=MARKERS[method],
                    color=lighten(base, 0.3),
                    linestyle="--",
                    label=label
                )

                # Save one handle per method
                if row == 0 and col == 0:
                    method_handles.append(line)

            ax.set_ylim(-0.2, 3.0)

            if row == 0:
                ax.set_title(f"{SPLITS[col]}")

            if col == 0:
                ax.set_ylabel(f"Setting {row+1}")

            if row == len(files) - 1:
                ax.set_xlabel("Number of points")

    handles = []
    labels = []

    for method in METHODS:
        base = COLORS[method]
        light = lighten(base, 0.3)

        ns = Line2D(
            [], [],
            color=base,
            linestyle="-",
            marker=MARKERS[method],
            linewidth=2,
        )

        ss = Line2D(
            [], [],
            color=light,
            linestyle="--",
            marker=MARKERS[method],
            linewidth=2,
        )



        handles.append((ss, ns))
        labels.append(method)

    fig.legend(
        handles,
        labels,
        handler_map={tuple: VerticalHandler()},
        loc="lower center",
        ncol=len(METHODS),
        bbox_to_anchor=(0.5, 0.025),
    )

    # Legend for methods
    #fig.legend(
    #    method_handles,
    #    [h.get_label() for h in method_handles],
    #    loc="lower center",
    #    ncol=len(METHODS),
    #    bbox_to_anchor=(0.5, -0.02)
    #)

    # Legend for line styles
    ss = plt.Line2D([], [], color="black", linestyle="-", label="Stationary")
    ns = plt.Line2D([], [], color="black", linestyle="--", label="Nonstationary")


    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save:
        plt.savefig("plots/new/all_settings.pdf", dpi=800)

    if show:
        plt.show()

if __name__ == '__main__':
    #new_nonstationarity_example(seed=2343, type_sel="v", nx=2, ny=2, save=False, plot=True)
    #new_nonstationarity_example(seed=2343, type_sel="v", nx=3, ny=3, save=False, plot=True)
    #exit()

    folder = "data/final/results/"
    new_plot_folder(folder, show=True, save=True)
    exit()
    subspace_folder = "data/csc/results"
    plot_folder(subspace_folder, show=False, save=True)
    exit()
    rank_folder = 'data/rank/results'
    for file in os.listdir(rank_folder):
        print(file)
        with open(os.path.join(rank_folder, file), "rb") as f:
            data = pickle.load(f)
            rank_plot(data, save=True, plot=False, save_file=f"plots/{file[:-4]}.pdf")
