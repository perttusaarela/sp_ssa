import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import re
import os


sns.set_palette("colorblind")

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
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.7
})

METHODS = ["stsir", "stsave", "stlcor", "stcomb", "random"]

MARKERS = {
    "stsir": 'o',
    "stsave": '^',
    "stlcor": 'v',
    "stcomb": '*',
    "random": 'x'
}

def get_method_name(method):
    return {
        "stsir": "stSSA-SIR",
        "stsave": "stSSA-SAVE",
        "stlcor": "stSSA-LCOR",
        "stcomb": "stSSA-COMB",
        "random": "Random"
    }.get(method, method)


def extract_total_obs(group_name):
    # example: data_160_10_space
    match = re.search(r"data_(\d+)_(\d+)", group_name)
    if match:
        loc = int(match.group(1))
        time = int(match.group(2))
        return loc * time
    return None


def plot_setting(df, setting):

    df = df[df["setting"] == setting].copy()

    # compute x-axis
    df["total_obs"] = df["group"].apply(extract_total_obs)

    splits = sorted(df["split"].unique())

    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True,
                             figsize=(5.5, 3.5))

    legend_lines = []
    legend_labels = []

    for col_idx, split in enumerate(splits):

        ax_top = axes[0, col_idx]
        ax_bottom = axes[1, col_idx]

        for method in METHODS:

            sub = df[(df["split"] == split) & (df["method"] == method)]

            if sub.empty:
                continue

            sub = sub.sort_values("total_obs")

            x = sub["total_obs"].values
            y_ns = sub["nonstationary_error"].values
            y_s = sub["stationary_error"].values

            label = get_method_name(method)

            line, = ax_top.plot(
                x,
                y_ns,
                marker=MARKERS[method],
                linestyle="-",
                label=label
            )

            ax_bottom.plot(
                x,
                y_s,
                marker=MARKERS[method],
                linestyle="-"
            )

            if col_idx == 0:
                legend_lines.append(line)
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

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/setting{setting}_new.pdf")

    plt.show()


if __name__ == "__main__":

    df = pd.read_csv("all_settings_results.csv")

    for setting in [1, 2, 3, 4]:
        plot_setting(df, setting)