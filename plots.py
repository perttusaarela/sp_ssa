import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle
import pickletools
import seaborn as sns
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
NOISE_DIMS = [1, 2, 3, 4, 5, 6, 10, 15, 20]
METHODS = ["spsir", "spsave", "splcor", "spcomb", "spcomb_s", "random"]
MARKERS ={
    "spsir": 'o',
    "spsave": '^',
    "splcor": 'v',
    "spcomb": '*',
    "spcomb_s": 's',
    "random": 'x'
}


def plot_floder(folder, show=True, save=False):
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
            plt.savefig(f"plots/{file[:-4]}.pdf", dpi=300, format='pdf')

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

    score_levels = list(range(6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(score_levels)))

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
    coords = sort_by_partition(coords, part)
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



if __name__ == '__main__':
    folder = 'data/full_data/csc/rank'
    rank_file = 'data/full_data/csc/rank/rank_sim2_res_1600.pkl'
    grid_plot()
    #plot_floder(folder,show=False,save=True)
    #plot_clustering_example()
    #rank_plot_box(rank_file)
    exit()
    for file in os.listdir(folder):
        if 'scaled' in file or '400' in file:
            continue

        print(file)
        rank_plot_box(os.path.join(folder, file), save=True)
