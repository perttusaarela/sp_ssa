import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle
import seaborn as sns

from rank_estimation import splits
from spatial import (generate_coordinates, clustering_partition, partition_coordinates, generate_spatial_data,
                     sort_by_partition, ssa_matern_covariance)
sns.set_palette("colorblind")  # This updates matplotlib too

# Use a serif font like you'd find in journals
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
    'text.usetex': True,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.7
})


def plot_non_stationary_mean_example():
    data = np.array([
        [1, -1, 1, -1],
        [-1, 1, -1, 1],
        [1, -1, 1, -1],
        [-1, 1, -1, 1]
    ])

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap=plt.get_cmap('jet'))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = ax.text(j, i, '{:.2f}'.format(data[i, j]), ha='center', va='center', color='w')

    fig.tight_layout()

    nums = np.random.uniform(low=-10.0, high=10.0, size=4)
    nums[3] = -1*np.sum(nums[:3])
    nums = nums / np.linalg.norm(nums)
    data = np.zeros((4, 4))
    for i in range(4):
        data[:, i] = nums

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap=plt.get_cmap('jet'))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = ax.text(j, i, r'$\mu = {:.2f}$'.format(data[i, j])+'\n'+r'$\sigma = {:.2f}$'.format(1.0), ha='center', va='center', color='w')

    fig.tight_layout()

    plt.show()


def plot_clustering_example():
    num_points = 900
    sl = int(np.sqrt(num_points))
    data = generate_coordinates(num_points, sl)
    clusters = clustering_partition(data, 3 , sl)
    plt.figure()
    colors = ['m', 'y', 'c']
    cluster_markers = ['o', '^', 'v']
    for idx, cluster in enumerate(clusters):
        cluster_data = data[cluster[1]]
        plt.scatter(cluster_data[:,0], cluster_data[:,1], color=colors[idx], marker=cluster_markers[idx])
        plt.scatter(cluster[0][0], cluster[0][1], marker='X', c='k')

    plt.show()



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


def plot_folder(folder, show=True, save=False):
    x_axis = [x ** 2 for x in range(20, 80, 10)]
    for file in os.listdir(folder):  # for each setting
        fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True,
                                 figsize=(5.5, 3.5))  # we make three plots, one for each split
        separated_data_ss = [{x: [] for x in METHODS} for _ in range(len(SPLITS))]
        separated_data_ns = [{x: [] for x in METHODS} for _ in range(len(SPLITS))]
        with open(os.path.join(folder, file), 'rb') as f:
            data = pickle.load(f)
            for met, method_data in data.items():
                for data_by_num_points in method_data:
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
                    line, = ax.plot(x_axis, data_to_plot, label=method,
                                    marker=MARKERS[method], linestyle='-')
                    # Only collect legend handles from the first subplot
                    if row_idx == 0 and col_idx == 0:
                        method_lines.append(line)
                        method_labels.append(method)

                    ax.set_ylim(-0.2, 2.5)

                    if row_idx == 0:
                        ax.set_title(f"{SPLITS[col_idx]}")
                    if row_idx == 1 and col_idx == 0:
                        ax.set_ylabel("Non-stationary")
                    if col_idx == 0 and row_idx == 0:
                        ax.set_ylabel("Stationary")
                    if row_idx == 1 and col_idx == 1:
                        ax.set_xlabel("Number of points")

        # Add a single legend to the entire figure
        fig.legend(method_lines, method_labels, loc='lower center', ncol=len(METHODS), bbox_to_anchor=(0.5, -0.025))
        plt.tight_layout(rect=[0.01, 0.01, 1, 0.95])  # Make room for the legend and title

        if save:
            plt.savefig(f"plots/new/{file[:-4]}.pdf", dpi=300, format='pdf')

        if show:
            plt.show()


def plot_folder_alt(folder, show=True, save=False):
    x_axis = [x ** 2 for x in range(20, 80, 10)]
    for file in os.listdir(folder):  # for each setting
        fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True,
                                 figsize=(5.5, 3.5))  # we make three plots, one for each split
        separated_data_ss = [{x: [] for x in METHODS} for _ in range(len(SPLITS))]
        separated_data_ns = [{x: [] for x in METHODS} for _ in range(len(SPLITS))]
        with open(os.path.join(folder, file), 'rb') as f:
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
                    line, = ax.plot(x_axis, data_to_plot, label=method,
                                    marker=MARKERS[method], linestyle='-')
                    # Only collect legend handles from the first subplot
                    if row_idx == 0 and col_idx == 0:
                        method_lines.append(line)
                        method_labels.append(method)

                    ax.set_ylim(-0.2, 3.0)

                    if row_idx == 0:
                        ax.set_title(f"{SPLITS[col_idx]}")
                    if row_idx == 1 and col_idx == 0:
                        ax.set_ylabel("Non-stationary")
                    if col_idx == 0 and row_idx == 0:
                        ax.set_ylabel("Stationary")
                    if row_idx == 1 and col_idx == 1:
                        ax.set_xlabel("Number of points")

        # Add a single legend to the entire figure
        fig.legend(method_lines, method_labels, loc='lower center', ncol=len(METHODS), bbox_to_anchor=(0.5, -0.025))
        plt.tight_layout(rect=[0.01, 0.01, 1, 0.95])  # Make room for the legend and title

        if save:
            plt.savefig(f"plots/new/{file[:-4]}.pdf", dpi=300, format='pdf')

        if show:
            plt.show()


def stitch_work():
    with open("data/full_data/csc/sim4_short_res.pkl", "rb") as f:
        data = pickle.load(f)
        with open(f"data/full_data/csc/sim4_short_res(1).pkl", "rb") as fs:
            data_scaled = pickle.load(fs)
            data["spcomb_s"] = data_scaled["spcomb"]

        with open("data/full_data/short/sim4_short_res_fused.pkl", "wb") as ff:
            pickle.dump(data, ff)


def rank_plot(file):
    methods = METHODS[:4]
    noise_dims = NOISE_DIMS[:-2]

    with open(file, 'rb') as f:
        data = pickle.load(f)

    num_splits = len(SPLITS)
    fig, axes = plt.subplots(nrows=1, ncols=num_splits, squeeze=False, sharey=True)
    axes = axes[0]  # flatten since squeeze=False

    score_levels = list(range(6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(score_levels)))  # consistent colors for all subplots

    for idx, split in enumerate(SPLITS):
        method_labels = [f"{m}_{d}" for d in noise_dims for m in methods]
        scores_by_label = {label: [] for label in method_labels}

        for data_point in data:
            if data_point['split'] != split:
                continue
            noise_dim = data_point['noise_dim']
            if noise_dim not in noise_dims:
                continue
            for method, score in data_point['result'].items():
                if method == 'error' or method not in methods:
                    continue
                label = f"{method}_{noise_dim}"
                scores_by_label[label].append(score)

        proportions = {
            label: np.array([
                np.sum(np.asarray(scores) == level) for level in score_levels
            ]) / len(scores) if scores else np.zeros(len(score_levels))
            for label, scores in scores_by_label.items()
        }

        sorted_labels = sorted(proportions.keys(), key=lambda x: (x.split('_')[0], int(x.split('_')[1])))

        ax = axes[idx]
        left = np.zeros(len(sorted_labels))

        for i, level in enumerate(score_levels):
            widths = [proportions[label][level] for label in sorted_labels]
            ax.barh(sorted_labels, widths, left=left, label=f"{level}", color=colors[i])
            left += widths

        ax.set_xlabel("Proportion")
        ax.set_title(f"{split}")
        if idx == num_splits - 1:
            ax.legend(title="Score", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


def rank_plot_box(file, save=False, plot=True):
    methods = METHODS[:4]
    noise_dims = NOISE_DIMS[:-2]

    with open(file, 'rb') as f:
        data = pickle.load(f)

    num_splits = len(SPLITS)
    fig, axes = plt.subplots(nrows=1, ncols=num_splits, squeeze=False, sharey=True, figsize=(12, 6))
    axes = axes[0]  # flatten

    # Score levels 0–5
    score_levels = list(range(8))

    # Center score = 3 (correct)
    center_score = 3

    # Distance from correct score
    distances = np.array([abs(s - center_score) for s in score_levels])

    # Normalize distances to [0,1]
    norm_dist = distances / distances.max()

    # Create a neutral center color (e.g., light gray) and dark edges
    # You can customize these two colors
    edge_color = np.array([0.2, 0.2, 0.2])  # dark gray for wrong answers
    center_color = np.array([0.85, 0.85, 0.85])  # light neutral for correct score

    # Linear interpolation between center_color and edge_color
    colors = np.array([
        center_color * (1 - d) + edge_color * d
        for d in norm_dist
    ])

    group_gap = 1  # space between noise_dim groups

    for idx, split in enumerate(SPLITS):
        ax = axes[idx]
        grouped_scores = {d: {m: [] for m in methods} for d in noise_dims}

        for data_point in data:
            if data_point['split'] != split:
                continue
            d = data_point['noise_dim']
            if d not in noise_dims:
                continue
            for m, score in data_point['result'].items():
                if m == 'error' or m not in methods:
                    continue
                grouped_scores[d][m].append(score)

        # Build list of proportions and labels
        proportions = []
        y_labels = []
        group_midpoints = {}
        y_index = 0

        # When building the y_labels list:
        real_row_indices = []

        # Replace this block when building y_labels:
        for d in noise_dims:
            group_start = y_index
            for m in methods:
                scores = grouped_scores[d][m]
                if not scores:
                    continue
                prop = [
                    np.sum(np.asarray(scores) == level) / len(scores)
                    for level in score_levels
                ]
                proportions.append(prop)
                y_labels.append(m)
                real_row_indices.append(y_index)  # store this row index
                y_index += 1
            group_end = y_index
            group_midpoints[d] = (group_start + group_end - 1) / 2
            # Add spacer row
            proportions.append([0.0] * len(score_levels))
            y_labels.append("")
            y_index += group_gap

        # Plot stacked bars
        left = np.zeros(len(proportions))
        for i, level in enumerate(score_levels):
            widths = [p[level] for p in proportions]
            ax.barh(range(len(proportions)), widths, left=left, color=colors[i],
                    label=str(level) if idx == num_splits - 1 else None)
            left += widths

        # Show y-ticks only on real rows
        ax.set_yticks(real_row_indices)
        ax.set_yticklabels(
            [y_labels[i] for i in real_row_indices],
            rotation=30,  # or 45 if more space is needed
            ha='right',  # align nicely after rotation
            va='center'
        )
        ax.set_xlabel("Proportion")
        ax.set_title(f"{split}")

        # Add noise_dim labels only on the rightmost subplot
        # Inside the plotting loop
        if idx == num_splits - 1:
            # Get max X to place labels outside the plot
            xlim = ax.get_xlim()
            x_offset = xlim[1] * 1.005  # adjust as needed

            for d, y_pos in group_midpoints.items():
                ax.text(
                    x_offset, y_pos,
                    f"d={d}",
                    transform=ax.transData,
                    va='center',
                    ha='left',
                    fontsize=9,
                    color='black',
                    rotation=-90  # 90° clockwise
                )

            # Adjust legend so it doesn’t overlap with d labels
            ax.legend(
                title=r"$\hat{q}$",
                bbox_to_anchor=(1.10, 1),  # further to the right
                loc='upper left',
                fontsize=9,
                title_fontsize=10
            )

    # Adjust figure layout to allow for labels and legend
    plt.subplots_adjust(right=0.8)
    plt.tight_layout()
    if save:
        dest = file.split("/")[-1]

        plt.savefig(f"plots/{dest[:-4]}.pdf", dpi=300, format='pdf')
    if plot:
        plt.show()


def fuse_rank_plots(file_in, file_out):
    with open(file_out, 'rb') as g:
        data_o = pickle.load(g)
        print(len(data_o))
        with open(file_in, 'rb') as f:
            data_i = pickle.load(f)
            print(len(data_i))
            data_o.extend(data_i)
            print(len(data_o))

    with open(file_out, 'wb') as g:
        pickle.dump(data_o, g)


from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap
import matplotlib.patches as patches
from matplotlib.colors import Normalize
def grid_plot():

    num_points = 200
    sl = int(np.sqrt(num_points))  # side length of square
    num_buckets = 7
    split = (4, 4)
    # Randomly scattered coordinates within the square [0, sl] × [0, sl]
    coords = generate_coordinates(num_points, sl)
    part = partition_coordinates(coords, split[0], split[1], sl)
    rectangles = [(p[0], p[1], p[2]) for p in part]
    coords, part = sort_by_partition(coords, part)
    cov_mat = ssa_matern_covariance(coords)
    # Example data values (real-valued)
    data = generate_spatial_data(cov_mat)
    data = (data - np.mean(data)) / np.std(data)

    means = [np.mean(data[p[-1]]) for p in part]
    # ----------------------------------------------------
    # Define discrete color buckets
    # ----------------------------------------------------
    vmin, vmax = np.min(data), np.max(data)
    bounds = np.linspace(vmin, vmax, num_buckets + 1)
    norm = BoundaryNorm(boundaries=bounds, ncolors=num_buckets)
    cmap = get_cmap('viridis', num_buckets)

    # ----------------------------------------------------
    # Create scatter plot
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    # sc = plt.scatter(coords[:, 0], coords[:, 1], c=data, cmap=cmap, norm=norm, s=50, edgecolor='k')

    # Overlay rectangles
    # ----------------------------------------------------
    gray_cmap = get_cmap("Greys")
    gray_norm = Normalize(vmin=np.nanmin(means), vmax=np.nanmax(means))

    for (rect_info, mean_val) in zip(rectangles, means):
        p, h, w = rect_info
        if np.isnan(mean_val):
            facecolor = (0.9, 0.9, 0.9)  # light gray for empty
        else:
            facecolor = gray_cmap(gray_norm(mean_val))

        rect = patches.Rectangle(
            p,
            w, h,
            linewidth=0.5,
            edgecolor='white',
            facecolor=facecolor
        )
        ax.add_patch(rect)
        ax.add_patch(rect)

    sc = plt.scatter(coords[:, 0], coords[:, 1], c=data, cmap=cmap, norm=norm, s=50, edgecolor='k')


    plt.colorbar(sc, spacing='proportional', label='Data value')
    plt.xlim(0, sl)
    plt.ylim(0, sl)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"2D Data Colored by Value ({num_buckets} buckets)")
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


res_dict = {
    "spsir": {
        400: {
            (2,2): [1.61072932,1.61072932],
            (3,3): [0.93925497,0.93925497],
            (4,4): [0.43522394,0.43522394]
        },
        900: {
            (2,2): [1.60668966,1.60668966],
            (3,3): [0.93925497,0.93925497],
            (4,4): [0.20614724,0.20614724]
        },
        1600: {
            (2,2): [1.60016255,1.60016255],
            (3,3): [0.89637754,0.89637754],
            (4,4): [0.11789792,0.11789792]
        },
        2500: {
            (2, 2): [1.60976362, 1.60976362],
            (3, 3): [0.88713596, 0.88713596],
            (4, 4): [0.07698501, 0.07698501]
        },
        3600: {
            (2, 2): [1.58838916, 1.58838916],
            (3, 3): [0.88338764, 0.88338764],
            (4, 4): [0.05375329, 0.05375329]
        },
        4900: {
            (2, 2): [1.61422294, 1.61422294],
            (3, 3): [0.8866145, 0.8866145],
            (4, 4): [0.04120975, 0.04120975]
        },
    },
    "spsave": {
        400: {
            (2,2): [0.7905408,0.7905408],
            (3,3): [0.67715086,0.67715086],
            (4,4): [0.94620842,0.94620842]
        },
        900: {
            (2,2): [0.63617008,0.63617008],
            (3,3): [0.44827826,0.44827826],
            (4,4): [0.81487832,0.81487832]
        },
        1600: {
            (2,2): [0.5130278,0.5130278],
            (3,3): [0.2589615,0.2589615],
            (4,4): [0.67516348,0.67516348]
        },
        2500: {
            (2, 2): [0.35203118, 0.35203118],
            (3, 3): [0.1512421, 0.1512421],
            (4, 4): [0.47548037, 0.47548037]
        },
        3600: {
            (2, 2): [0.2740608, 0.2740608],
            (3, 3): [0.09757176, 0.09757176],
            (4, 4): [0.28022613, 0.28022613]
        },
        4900: {
            (2, 2): [0.18506885, 0.18506885],
            (3, 3): [0.07035967, 0.07035967],
            (4, 4): [0.15478441, 0.15478441]
        },
    },
    "splcor": {
        400: {
            (2, 2): [1.34495408, 1.34495408],
            (3, 3): [2.00062688, 2.00062688],
            (4, 4): [2.0313821, 2.0313821]
        },
        900: {
            (2, 2): [1.19854914, 1.19854914],
            (3, 3): [1.90998267, 1.90998267],
            (4, 4): [1.94521107, 1.94521107]
        },
        1600: {
            (2, 2): [1.12244242, 1.12244242],
            (3, 3): [1.87334257, 1.87334257],
            (4, 4): [1.92409588, 1.92409588]
        },
        2500: {
            (2, 2): [1.0760879, 1.0760879],
            (3, 3): [1.84981061, 1.84981061],
            (4, 4): [1.93344402, 1.93344402]
        },
        3600: {
            (2, 2): [1.05176942, 1.05176942],
            (3, 3): [1.83023808, 1.83023808],
            (4, 4): [1.94981707, 1.94981707]
        },
        4900: {
            (2, 2): [1.03939149, 1.03939149],
            (3, 3): [1.83119213, 1.83119213],
            (4, 4): [1.95762468, 1.95762468]
        },
    },  # made it this far
    "spcomb": {
        400: {
            (2, 2): [0.82924784, 0.82924784],
            (3, 3): [0.67170988, 0.67170988],
            (4, 4): [0.54444739, 0.54444739]
        },
        900: {
            (2, 2): [0.67872704, 0.67872704],
            (3, 3): [0.45968442, 0.45968442],
            (4, 4): [0.20685853, 0.20685853]
        },
        1600: {
            (2, 2): [0.5469045, 0.5469045],
            (3, 3): [0.2633808, 0.2633808],
            (4, 4): [0.09841526, 0.09841526]
        },
        2500: {
            (2, 2): [0.38303477, 0.38303477],
            (3, 3): [0.14657181, 0.14657181],
            (4, 4): [0.06166598, 0.06166598]
        },
        3600: {
            (2, 2): [0.29637255, 0.29637255],
            (3, 3): [0.09511739, 0.09511739],
            (4, 4): [0.04300429, 0.04300429]
        },
        4900: {
            (2, 2): [0.19853594, 0.19853594],
            (3, 3): [0.06904989, 0.06904989],
            (4, 4): [0.03328103, 0.03328103]
        },
    },
    "random": {
        400: {
            (2, 2): [1.87499507, 1.87499507],
            (3, 3): [1.87499507, 1.87499507],
            (4, 4): [1.87499507, 1.87499507 ]
        },
        900: {
            (2, 2): [1.88124546, 1.88124546],
            (3, 3): [1.88124546, 1.88124546],
            (4, 4): [1.88124546, 1.88124546]
        },
        1600: {
            (2, 2): [1.87685617, 1.87685617],
            (3, 3): [1.87685617, 1.87685617],
            (4, 4): [1.87685617, 1.87685617]
        },
        2500: {
            (2, 2): [1.88027344, 1.88027344],
            (3, 3): [1.88027344, 1.88027344],
            (4, 4): [1.88027344, 1.88027344]
        },
        3600: {
            (2, 2): [1.87647631, 1.87647631],
            (3, 3): [1.87647631, 1.87647631],
            (4, 4): [1.87647631, 1.87647631]
        },
        4900: {
            (2, 2): [1.8804669, 1.8804669],
            (3, 3): [1.8804669, 1.8804669],
            (4, 4): [1.8804669, 1.8804669]
        },
    }
}


def combine_rank_stats(folder):
    combined = {
        m: np.zeros(10, dtype=int) for m in METHODS[:4]
    }
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), 'rb') as f:
            data = pickle.load(f)
            for m, counts in data.items():
                combined[m] += counts

    with open(os.path.join(folder, "comb.pkl"), 'wb') as f:
        pickle.dump(combined, f)


def new_rank_plot_box(data, save=False, plot=True, save_file=None):
    methods = METHODS[:-1]             # same as before
    noise_dims = [1, 5, 10, 15]        # same as before

    # ======================================================================
    # 1. SCORE LEVELS AND CENTERED COLOR MAP
    # ======================================================================
    max_score = 8
    center_score = 3   # correct score

    score_levels = list(range(max_score + 1))

    # Reorder score levels so that center_score is visually centered
    left = [s for s in score_levels if s < center_score]
    right = [s for s in score_levels if s > center_score]
    plot_order = left + [center_score] + right

    # Distances from correct score (for coloring)
    distances = np.array([abs(s - center_score) for s in plot_order])
    norm_dist = distances / distances.max()

    # Color scheme: neutral center, darker edges
    edge_color = np.array([0.2, 0.2, 0.2])       # darkest wrong
    center_color = np.array([0.85, 0.85, 0.85])  # neutral correct score

    colors = []
    for score in plot_order:

        if score < center_score:  # UNDERSHOOT → BLUE → center_color
            # normalize 0→1 range for distance
            d = (center_score - score) / (center_score - 0)
            # dark blue (overshoot) → light gray (center)
            blue = np.array([0.2, 0.4, 1.0])  # dark-ish blue
            color = blue * d + center_color * (1 - d)

        elif score > center_score:  # OVERSHOOT → RED → center_color
            d = (score - center_score) / (max_score - center_score)
            red = np.array([1.0, 0.3, 0.3])  # red tones
            color = red * d + center_color * (1 - d)

        else:  # EXACT HIT
            color = center_color

        colors.append(color)

    colors = np.array(colors)

    # ======================================================================
    # 2. GROUP DATA BY NOISE-DIM AND METHOD
    # ======================================================================
    grouped_scores = {
        d: {m: [] for m in methods}
        for d in noise_dims
    }

    # Data is simply data[method] = list_of_scores
    # We assume same data for every noise dimension
    for d in noise_dims:
        for m in methods:
            grouped_scores[d][m] = data[d][m]

    # ======================================================================
    # 3. BUILD PROPORTIONS TABLE
    # ======================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    proportions = []
    y_labels = []
    real_row_indices = []
    group_midpoints = {}
    y_index = 0
    group_gap = 1

    for d in noise_dims:
        group_start = y_index

        for m in methods:
            scores = grouped_scores[d][m]
            if len(scores) == 0:
                continue

            # distribution of scores (0–8)
            counts = np.asarray(scores)  # scores is actually histogram counts
            N = counts.sum()
            prop = counts / N
            proportions.append(prop)
            y_labels.append(m)
            real_row_indices.append(y_index)
            y_index += 1

        group_end = y_index
        group_midpoints[d] = (group_start + group_end - 1) / 2

        # spacer row
        proportions.append([0.0] * len(score_levels))
        y_labels.append("")
        y_index += group_gap

    # ======================================================================
    # 4. PLOT STACKED BAR CHART
    # ======================================================================
    left = np.zeros(len(proportions))
    for i, score_val in enumerate(plot_order):
        widths = [p[score_val] for p in proportions]
        ax.barh(
            np.arange(len(proportions)),
            widths,
            left=left,
            color=colors[i],
            label=str(score_val)
        )
        left += widths

    # Y-ticks for real rows only
    ax.set_yticks(real_row_indices)
    ax.set_yticklabels(
        [y_labels[i] for i in real_row_indices],
        rotation=30,
        ha='right',
        va='center'
    )

    ax.set_xlabel("Proportion")
    #ax.set_title("Score distribution by method and noise dimension")

    # ======================================================================
    # 5. ADD NOISE-DIM LABELS ON THE RIGHT
    # ======================================================================
    xlim = ax.get_xlim()
    x_offset = xlim[1] * 1.005
    for d, y in group_midpoints.items():
        ax.text(
            x_offset, y,
            f"d={d}",
            va='center',
            ha='left',
            fontsize=9,
            rotation=-90
        )

    # Legend centered around score=3
    # --- FIX LEGEND ORDER: numerical 0 → max_score ---
    # Collect legend handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Convert labels to ints, sort them
    labels_int = np.array([int(l) for l in labels])
    sort_idx = np.argsort(labels_int)

    # Reorder handles and labels
    handles = [handles[i] for i in sort_idx]
    labels = [labels[i] for i in sort_idx]

    # Now pass sorted items to the legend
    ax.legend(
        handles,
        labels,
        title="Score",
        bbox_to_anchor=(1.10, 1),
        loc='upper left'
    )

    plt.subplots_adjust(right=0.8)

    if save:
        if save_file is None:
            plt.savefig("plots/plot.pdf", dpi=300, format='pdf')
        else:
            plt.savefig(save_file, dpi=300, format='pdf')

    if plot:
        plt.show()


def stuff(folder):
    data_dict = {
        1: {},
        5: {},
        10: {},
        15: {}
    }

    for subdir in os.listdir(folder):
        dir_path = os.path.join(folder, subdir)
        if len(subdir) == 4:
            key = int(subdir[-1])
        else:
            key = int(subdir[-2:])

        with open(os.path.join(dir_path, "comb.pkl"), 'rb') as f:
            data = pickle.load(f)
            data_dict[key] = data

    with open(os.path.join(folder, "comb.pkl"), "wb") as f:
        pickle.dump(data_dict, f)


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

    # --- generate coordinates ---
    for i in range(num_units):
        box = generate_coordinates(points_per_unit, hi=side_length / 3)
        offset = np.array([(i // 3) * side_length / 3,
                           (i % 3) * side_length / 3])
        coords[i * points_per_unit:(i + 1) * points_per_unit] = box + offset

    x, y = coords[:, 0], coords[:, 1]

    # --- generate values ---
    cov = ssa_matern_covariance(coords)
    values = generate_spatial_data(cov)

    if type_sel == "m":
        means = generate_means()
        for i in range(num_units):
            sl = slice(i * points_per_unit, (i + 1) * points_per_unit)
            values[sl] -= values[sl].mean()
            values[sl] += means[i // 3, i % 3]
        label_stat = "Mean"

    elif type_sel == "v":
        var = np.array([
            [0.2, 1.0, 0.2],
            [1.0, 3.0, 1.0],
            [0.2, 1.0, 0.2]
        ])
        for i in range(num_units):
            sl = slice(i * points_per_unit, (i + 1) * points_per_unit)
            values[sl] -= values[sl].mean()
            values[sl] /= values[sl].std()
            values[sl] *= np.sqrt(var[i // 3, i % 3])
        label_stat = "Variance"

    values = (values - values.mean()) / values.std()

    # --- bucket values ---
    num_buckets = 7
    bins = np.linspace(values.min(), values.max(), num_buckets + 1)
    bucket_idx = np.digitize(values, bins) - 1

    markers = ['o', 's', '^', 'D', 'P', '*', 'x'][:num_buckets]

    # --- grid setup ---
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

    # --- plotting ---
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
        fontsize=9
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])

    plt.tight_layout()

    if save:
        if save_file is None:
            plt.savefig(f"plots/partition_plot_{nx}.pdf", dpi=300, format='pdf')
        else:
            plt.savefig(save_file, dpi=300, format='pdf')

    if plot:
        plt.show()



def labor():
    data_point_lengths = [x**2 for x in range(20, 80, 10)]
    full_data = {
        m: {
            d: {
                s: np.zeros((2, 2000)) for s in SPLITS
            } for d in data_point_lengths
        } for m in METHODS
    }
    path = "data/final/setting4"
    for num_points in data_point_lengths:
        for idx in range(10):
            file = f"{path}/data_{num_points}_{idx}.pkl"
            with open(file, "rb") as f:
                data = pickle.load(f)
                for m, data_by_split in data.items():
                    for split, data_array in data_by_split.items():
                        full_data[m][num_points][split][:, idx * 200 : (idx + 1) * 200] = data_array

    final_result = {
        m: {
            d: {
                s: np.zeros(2) for s in SPLITS
            } for d in data_point_lengths
        } for m in METHODS
    }
    for m, data_by_points in full_data.items():
        for num_points, data_by_split in data_by_points.items():
            for split, data_array in data_by_split.items():
                final_result[m][num_points][split] = data_array.mean(axis=1)

    with open("data/final/results/sim4.pkl", "wb") as f:
        pickle.dump(final_result, f)


def labor2():
    methods = METHODS[:-1]
    path = "data/striped/rank/setting4"
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
    for dim in data_dict.keys():
        for idx in range(10):
            file = f"{path}/s{dim}_{idx}.pkl"
            with open(file, "rb") as f:
                data = pickle.load(f)
                for m, data_arr in data.items():
                    data_dict[dim][m] +=  data_arr

    with open("data/final/rank/sl40split4x4/rank_sim4.pkl", "wb") as f:
        pickle.dump(data_dict, f)



if __name__ == '__main__':

    #labor2()
    #exit()
    nonstationarity_example(seed=2343, type_sel="m", nx=2, ny=2, save=True)
    nonstationarity_example(seed=2343, type_sel="m", nx=3, ny=3, save=True)
    #folder = "data/final/results"
    #plot_folder_alt(folder, show=False, save=True)


    exit()
    folder = 'data/final/rank/sl40split4x4'
    for file in os.listdir(folder):
        print(file)
        with open(os.path.join(folder, file), "rb") as f:
            data = pickle.load(f)
            new_rank_plot_box(data, save=True, plot=True, save_file=f"plots/{file[:-4]}.pdf")

    exit()
    with open(folder + '/s1_0.pkl', 'rb') as f:
        data = pickle.load(f)
        print()
    exit()

    folder = 'data/striped/raw/local_results/handmade'
    with open(os.path.join(folder, 'sim1.pkl'), 'wb') as f:
        pickle.dump(res_dict, f)
    #folder = 'data/striped/results'
    #rank_file = 'data/full_data/csc/rank/rank_sim2_res_1600.pkl'
    #grid_plot()
    plot_folder_alt(folder,show=False,save=True)
    #plot_clustering_example()
    #rank_plot_box(rank_file)
    exit()
    for file in os.listdir(folder):
        if 'scaled' in file or '400' in file:
            continue

        print(file)
        rank_plot_box(os.path.join(folder, file), save=True)
