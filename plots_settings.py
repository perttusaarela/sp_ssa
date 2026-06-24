import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import re
import os

sns.set_palette("colorblind")

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "lines.linewidth": 1.3,
    "lines.markersize": 5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

METHODS = ["stsir", "stsave", "stlcor", "stcomb", "random"]

MARKERS = {
    "stsir": "o",
    "stsave": "^",
    "stlcor": "v",
    "stcomb": "*",
    "random": "x"
}

def get_method_name(method):
    return {
        "stsir": "stSSA-SIR",
        "stsave": "stSSA-SAVE",
        "stlcor": "stSSA-LCOR",
        "stcomb": "stSSA-COMB",
        "random": "Random"
    }.get(method, method)

def parse_group(group_name):
    """
    Example:
    data_160_10_space
    data_160_10_space_time
    data_80_50_time
    """
    match = re.match(r"data_(\d+)_(\d+)_(.+)", group_name)
    if not match:
        return None, None, None

    loc = int(match.group(1))
    time = int(match.group(2))
    structure = match.group(3)
    return loc, time, structure

def prepare_dataframe(df):
    parsed = df["group"].apply(parse_group)
    df["locations"] = parsed.apply(lambda x: x[0])
    df["times"] = parsed.apply(lambda x: x[1])
    df["structure"] = parsed.apply(lambda x: x[2])
    df["total_obs"] = df["locations"] * df["times"]
    return df

def plot_setting_structure(df, setting, structure):
    subdf = df[(df["setting"] == setting) & (df["structure"] == structure)].copy()

    if subdf.empty:
        print(f"No data for setting={setting}, structure={structure}")
        return

    splits = sorted(subdf["split"].unique())
    unique_x = sorted(subdf["total_obs"].dropna().unique())

    fig, axes = plt.subplots(
        2, 3,
        sharex=True,
        sharey="row",
        figsize=(9.5, 5.6)
    )

    legend_lines = []
    legend_labels = []

    for col_idx, split in enumerate(splits):
        ax_top = axes[0, col_idx]
        ax_bottom = axes[1, col_idx]

        for method in METHODS:
            method_df = subdf[
                (subdf["split"] == split) &
                (subdf["method"] == method)
            ].copy()

            if method_df.empty:
                continue

            method_df = method_df.sort_values("total_obs")

            x = method_df["total_obs"].values
            y_ns = method_df["nonstationary_error"].values
            y_s = method_df["stationary_error"].values

            label = get_method_name(method)

            alpha = 0.6 if method == "random" else 0.95
            linewidth = 1.0 if method == "random" else 1.3
            markersize = 4 if method != "stcomb" else 6

            line_top, = ax_top.plot(
                x, y_ns,
                marker=MARKERS[method],
                linestyle="-",
                linewidth=linewidth,
                markersize=markersize,
                alpha=alpha,
                label=label
            )

            ax_bottom.plot(
                x, y_s,
                marker=MARKERS[method],
                linestyle="-",
                linewidth=linewidth,
                markersize=markersize,
                alpha=alpha
            )

            if col_idx == 0:
                legend_lines.append(line_top)
                legend_labels.append(label)

        ax_top.set_title(f"Split = {split}")
        ax_top.set_ylim(-0.1, 3.0)
        ax_bottom.set_ylim(-0.1, 3.0)

        ax_top.set_xticks(unique_x)
        ax_bottom.set_xticks(unique_x)
        ax_top.set_xticklabels([])
        ax_bottom.set_xticklabels([str(v) for v in unique_x])

        # horizontal grid lines only
        ax_top.yaxis.grid(True, linestyle=":", alpha=0.6)
        ax_bottom.yaxis.grid(True, linestyle=":", alpha=0.6)

        if col_idx == 0:
            ax_top.set_ylabel(r"$n_{\mathrm{perf}}$")
            ax_bottom.set_ylabel(r"$s_{\mathrm{perf}}$")

    pretty_structure = structure.replace("_", "-")
    fig.suptitle(f"Setting {setting} ({pretty_structure})", y=0.98, fontsize=12)
    fig.supxlabel("Total observations = locations × times", y=0.08)

    fig.legend(
        legend_lines,
        legend_labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.00),
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.5
    )

    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.88,
        bottom=0.20,
        wspace=0.28,
        hspace=0.20
    )

    os.makedirs("plots", exist_ok=True)
    out_base = f"plots/setting{setting}_{structure}"
    plt.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.savefig(f"{out_base}.png", bbox_inches="tight")
    plt.show()
    plt.close()

def main():
    df = pd.read_csv("all_settings_results.csv")
    df = prepare_dataframe(df)

    structures = ["space", "time", "space_time"]

    for setting in [1, 2, 3, 4]:
        for structure in structures:
            plot_setting_structure(df, setting, structure)

if __name__ == "__main__":
    main()