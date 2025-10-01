import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle
import seaborn as sns
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
    'text.usetex': False,  # Turn on if you're using LaTeX
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


if __name__ == '__main__':
    splits = [(2, 2), (3, 3), (4, 4)]
    methods = ["spsir", "spsave", "splcor", "spcomb", "random"]
    markers ={
        "spsir": 'o',
        "spsave": '^',
        "splcor": 'v',
        "spcomb": '*',
        "random": 'x'
    }
    folder = 'data/full_data/csc/'
    x_axis =[x**2 for x in range(20, 80, 10)]
    for file in os.listdir(folder):  # for each setting
        fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(5.5, 3.5))  # we make three plots, one for each split
        separated_data_ss = [{x: [] for x in methods} for _ in range(len(splits))]
        separated_data_ns = [{x: [] for x in methods} for _ in range(len(splits))]
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

        for row_idx, row in enumerate(axes):
            for col_idx, ax in enumerate(row):
                for method in methods:
                    line, = ax.plot(x_axis, plot_data[row_idx][col_idx][method], label=method,
                                    marker=markers[method], linestyle='-')
                    # Only collect legend handles from the first subplot
                    if row_idx == 0 and col_idx == 0:
                        method_lines.append(line)
                        method_labels.append(method)

                    ax.set_ylim(-0.2, 3)

                    if row_idx == 0:
                        ax.set_title(f"{splits[col_idx]}")
                    if row_idx == 1 and col_idx == 0:
                        ax.set_ylabel("Non-stationary")
                    if col_idx == 0 and row_idx == 0:
                        ax.set_ylabel("Stationary")
                    if row_idx == 1 and col_idx == 1:
                        ax.set_xlabel("Number of points")


        # Add a single legend to the entire figure
        fig.legend(method_lines, method_labels, loc='lower center', ncol=len(methods), bbox_to_anchor=(0.5, -0.02))
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the legend and title
        plt.savefig(f"plots/setting_{file[-9]}", dpi=300)
        plt.show()

    #plot_non_stationary_mean_example()
