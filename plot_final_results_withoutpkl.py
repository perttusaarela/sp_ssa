"""
plot_final_results_withoutpkl.py
Complete plotting script for stSSA simulation results.
Reads only CSV files — does NOT need pkl files.
"""

import os
import re
import ast

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set_palette("colorblind")


mpl.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "lines.linewidth":   1.4,
    "lines.markersize":  5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
})



OUTPUT_DIR = "final_plots"
METHODS = ["stsir", "stsave", "stlcor", "stcomb", "random"]

METHOD_LABELS = {
    "stsir":  "stSSA-SIR",
    "stsave": "stSSA-SAVE",
    "stlcor": "stSSA-LCOR",
    "stcomb": "stSSA-COMB",
    "random": "Random",
}

STRUCTURE_LABELS = {
    "space":      "Spatial",
    "time":       "Temporal",
    "space_time": "Spatio-temporal",
}

SEGMENTATION_LABELS = {
    "spatial":         "Spatial",
    "temporal":        "Temporal",
    "spatio-temporal": "Spatio-temporal",
}

SETTING_LABELS = {
    1: "Mean",
    2: "Variance",
    3: "Covariance",
    4: "Combined mean + covariance",
}


_viridis = plt.cm.get_cmap("viridis")
MATCH_COLOURS = {
    "Matched":           _viridis(0.85),
    "Partially matched": _viridis(0.50),
    "Mismatched":        _viridis(0.10),
}

MATCH_ORDER = ["Matched", "Partially matched", "Mismatched"]

# Markers for line plots
MARKERS = {
    "stsir":  "o",
    "stsave": "^",
    "stlcor": "v",
    "stcomb": "*",
    "random": "x",
}


def safe_savefig(out_name):
    """Save both PDF and PNG. Handles Windows file permission errors."""
    for ext in [".pdf", ".png"]:
        path = out_name + ext
        try:
            plt.savefig(path, bbox_inches="tight")
        except PermissionError:
            alt = out_name + "_new" + ext
            print(f"  Permission denied for {path}. Saving as {alt}")
            plt.savefig(alt, bbox_inches="tight")


def parse_group(group_name):
    """
    Parses group name like 'data_160_100_space' into
    (locations, times, structure).
    Returns (None, None, None) if format does not match.
    """
    match = re.match(r"data_(\d+)_(\d+)_(.+)", str(group_name))
    if not match:
        return None, None, None
    return int(match.group(1)), int(match.group(2)), match.group(3)


def parse_split(split):
    """Converts split string '(4,4,1)' or tuple (4,4,1) to a tuple."""
    if isinstance(split, tuple):
        return split
    return ast.literal_eval(str(split))


def get_segmentation_type(split):
    """Returns 'spatial', 'temporal', 'spatio-temporal', or 'other'."""
    s = parse_split(split)
    if s[0] > 1 and s[1] > 1 and s[2] == 1:
        return "spatial"
    elif s[0] == 1 and s[1] == 1 and s[2] > 1:
        return "temporal"
    elif s[0] > 1 and s[1] > 1 and s[2] > 1:
        return "spatio-temporal"
    return "other"


def split_label(split):
    """Returns string like '(4,4,1)'."""
    s = parse_split(split)
    return f"({s[0]},{s[1]},{s[2]})"


def prepare_dataframe(df):
    df = df.copy()

    parsed             = df["group"].apply(parse_group)
    df["locations"]    = parsed.apply(lambda x: x[0])
    df["times"]        = parsed.apply(lambda x: x[1])
    df["structure"]    = parsed.apply(lambda x: x[2])
    df["total_obs"]    = df["locations"] * df["times"]

    df["split_tuple"]        = df["split"].apply(parse_split)
    df["segmentation"]       = df["split"].apply(get_segmentation_type)

    df["method_label"]       = df["method"].map(METHOD_LABELS)
    df["structure_label"]    = df["structure"].map(STRUCTURE_LABELS)
    df["segmentation_label"] = df["segmentation"].map(SEGMENTATION_LABELS)

    return df


def is_matched(row):
    """Returns True if segmentation type matches nonstationarity structure."""
    return (
        (row["structure"] == "space"      and row["segmentation"] == "spatial")
        or (row["structure"] == "time"       and row["segmentation"] == "temporal")
        or (row["structure"] == "space_time" and row["segmentation"] == "spatio-temporal")
    )


def get_match_type_fixed4(row):
    split  = row["split_tuple"]
    struct = row["structure"]
    seg    = row["segmentation"]

    valid_splits = {
        "spatial":         (4, 4, 1),
        "temporal":        (1, 1, 4),
        "spatio-temporal": (4, 4, 4),
    }

    if split != valid_splits.get(seg):
        return None

    if struct == "space"      and seg == "spatial":         return "Matched"
    if struct == "time"       and seg == "temporal":        return "Matched"
    if struct == "space_time" and seg == "spatio-temporal": return "Matched"

    if struct == "space"      and seg == "temporal":        return "Mismatched"
    if struct == "time"       and seg == "spatial":         return "Mismatched"

    return "Partially matched"


def heatmap_cmap():
    return plt.get_cmap("viridis_r")


def annotate_heatmap(ax, mat, fmt=".2f", fontsize=8):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, format(val, fmt),
                    ha="center", va="center",
                    fontsize=fontsize, color="black")


# Plot 1 — Grand summary heatmap (fixed split level)

def plot_grand_summary(df, split_level, output_dir):
    """
    Grand summary heatmap using one fixed split level.

    Rows    : setting x method  (20 rows total)
    Columns : structure x segmentation  (9 columns total)
    Values  : nperf at largest sample size
    """
    os.makedirs(output_dir, exist_ok=True)

    max_obs = df["total_obs"].max()

    wanted_splits = {
        "spatial":         (split_level, split_level, 1),
        "temporal":        (1, 1, split_level),
        "spatio-temporal": (split_level, split_level, split_level),
    }

    subdf = df[
        (df["total_obs"] == max_obs)
        & (df["segmentation"] != "other")
    ].copy()

    subdf = subdf[
        subdf.apply(
            lambda row: row["split_tuple"] == wanted_splits.get(
                row["segmentation"]
            ),
            axis=1,
        )
    ]

    if subdf.empty:
        print(f"  [Grand summary] No data for split level {split_level}")
        return

    subdf["setting_method"] = subdf.apply(
        lambda row: f"S{int(row['setting'])} {METHOD_LABELS[row['method']]}",
        axis=1,
    )

    subdf["struct_seg"] = subdf.apply(
        lambda row: (
            f"{STRUCTURE_LABELS.get(row['structure'], row['structure'])}\n"
            f"{SEGMENTATION_LABELS.get(row['segmentation'], row['segmentation'])}"
        ),
        axis=1,
    )

    summary = (
        subdf
        .groupby(["setting_method", "struct_seg"])["nonstationary_error"]
        .mean()
        .reset_index()
    )


    methods_no_random = [m for m in METHODS if m != "random"]
    row_order = [
        f"S{s} {METHOD_LABELS[m]}"
        for s in [1, 2, 3, 4]
        for m in methods_no_random
    ]

    col_order = [
        f"{STRUCTURE_LABELS[struct]}\n{SEGMENTATION_LABELS[seg]}"
        for struct in ["space", "time", "space_time"]
        for seg    in ["spatial", "temporal", "spatio-temporal"]
    ]

    heat = (
        summary
        .pivot(index="setting_method",
               columns="struct_seg",
               values="nonstationary_error")
        .reindex(index=row_order, columns=col_order)
    )

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(bottom=0.22, right=0.98, top=0.97, left=0.18)

    sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        cmap=heatmap_cmap(),
        linewidths=0.5,
        annot_kws={"size": 7},
        cbar=False,
        ax=ax,
    )

    for i in [5, 10, 15]:
        ax.axhline(i, color="black", linewidth=1.5)
    for j in [3, 6]:
        ax.axvline(j, color="black", linewidth=1.5)

    ax.set_xlabel("True structure and segmentation strategy", fontsize=10, labelpad=6)
    ax.set_ylabel("Setting and method", fontsize=10)

    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

  
    cbar_ax = fig.add_axes([0.25, 0.04, 0.50, 0.025])
    sm = plt.cm.ScalarMappable(
        cmap=heatmap_cmap(),
        norm=plt.Normalize(vmin=float(heat.min().min()),
                           vmax=float(heat.max().max()))
    )
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_label(r"$n_{\mathrm{perf}}$ (lower is better)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    out = os.path.join(
        output_dir,
        f"heatmap_grand_summary_fixed_split_{split_level}",
    )
    safe_savefig(out)
    plt.close()
    print(f"  Saved: {out}.pdf/.png")



# Plot 2 — segmentation sensitivity heatmap (stSSA-COMB only)

def plot_type_b(df, setting, output_dir):
    """
    stSSA-COMB segmentation sensitivity heatmap.

    Rows    : nonstationarity structure
    Columns : all 9 individual splits grouped by segmentation type
    Top     : nperf
    Bottom  : sperf
    """
    os.makedirs(output_dir, exist_ok=True)

    sub = df[
        (df["setting"] == setting)
        & (df["method"]  == "stcomb")
    ].copy()

    if sub.empty:
        print(f"   No stcomb data for setting {setting}")
        return

    max_obs = sub["total_obs"].max()
    sub     = sub[sub["total_obs"] == max_obs]

    structures = ["space", "time", "space_time"]
    seg_order  = ["spatial", "temporal", "spatio-temporal"]

    all_splits = sorted(
        sub["split_tuple"].unique(),
        key=lambda s: (seg_order.index(get_segmentation_type(s)), max(s)),
    )

    col_labels = [
        f"{split_label(s)}\n[{SEGMENTATION_LABELS[get_segmentation_type(s)]}]"
        for s in all_splits
    ]

    row_labels = [STRUCTURE_LABELS[st] for st in structures]

    def build_matrix(metric):
        mat = np.full((len(structures), len(all_splits)), np.nan)
        for r, struct in enumerate(structures):
            for c, sp in enumerate(all_splits):
                cell = sub[
                    (sub["structure"]    == struct)
                    & (sub["split_tuple"] == sp)
                ]
                if not cell.empty:
                    mat[r, c] = cell[metric].mean()
        return mat

    mat_n = build_matrix("nonstationary_error")
    mat_s = build_matrix("stationary_error")
    vmax  = min(float(np.nanmax([mat_n, mat_s])), 2.0)
    cmap  = heatmap_cmap()

    fig, axes = plt.subplots(
        2, 1, figsize=(13, 5.8),
        gridspec_kw={"hspace": 0.65},
    )

    for ax, mat, title_suffix in zip(
        axes,
        [mat_n, mat_s],
        [r"$n_{\mathrm{perf}}$", r"$s_{\mathrm{perf}}$"],
    ):
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)

        ax.set_xticks(np.arange(len(col_labels)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(row_labels)) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)

        annotate_heatmap(ax, mat, fmt=".2f", fontsize=8)

        ax.set_title(
            f"stSSA-COMB – {title_suffix}  (largest sample size)",
            fontsize=10, pad=6,
        )

        cb = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("Error", fontsize=8)

        seg_types = [get_segmentation_type(s) for s in all_splits]
        for idx in range(1, len(seg_types)):
            if seg_types[idx] != seg_types[idx - 1]:
                ax.axvline(idx - 0.5, color="black", linewidth=1.2)


    out = os.path.join(output_dir,
                       f"heatmap_setting{setting}_comb_segmentation")
    safe_savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}.pdf/.png")



# Plot 3 — Matched / Partially matched / Mismatched bar chart
def plot_matched_bars(df, output_dir):
    """
    Four separate bar chart figures — one per setting.

    Each figure:
        x-axis : methods (including Random)
        y-axis : average nperf  
        bars   : 3 per method — Matched / Partially matched / Mismatched

    Uses fixed split level 4 only: (4,4,1), (1,1,4), (4,4,4).
    Averages across the three nonstationarity structures.
    """
    os.makedirs(output_dir, exist_ok=True)

    max_obs = df["total_obs"].max()

    subdf = df[
        (df["total_obs"] == max_obs)
        & (df["segmentation"] != "other")
    ].copy()

    subdf["match_type"] = subdf.apply(get_match_type_fixed4, axis=1)
    subdf = subdf[subdf["match_type"].notna()].copy()

    if subdf.empty:
        print("  [Bar chart] No data found for fixed split level 4.")
        return

   
    method_order_keys   = [m for m in METHODS if m != "random"]
    method_order_labels = [METHOD_LABELS[m] for m in method_order_keys]

    bar_width   = 0.22
    group_gap   = 0.08
    group_width = len(MATCH_ORDER) * bar_width + group_gap

    for setting in sorted(subdf["setting"].unique()):

        setting_df = subdf[subdf["setting"] == setting].copy()

        summary = (
            setting_df
            .groupby(["method", "match_type"])["nonstationary_error"]
            .mean()
            .reset_index()
        )
        summary["method_label"] = summary["method"].map(METHOD_LABELS)

        fig, ax = plt.subplots(figsize=(8.5, 5.0))

        x_positions = np.arange(len(method_order_labels)) * group_width

        for bar_idx, match_type in enumerate(MATCH_ORDER):

            match_data = summary[summary["match_type"] == match_type]
            heights    = []

            for method_key in method_order_keys:
                label = METHOD_LABELS[method_key]
                row   = match_data[match_data["method_label"] == label]
                heights.append(
                    float(row["nonstationary_error"].values[0])
                    if not row.empty else np.nan
                )

            bar_x = x_positions + bar_idx * bar_width

            bars = ax.bar(
                bar_x,
                heights,
                width     = bar_width,
                label     = match_type,
                color     = MATCH_COLOURS[match_type],
                edgecolor = "white",
                linewidth = 0.5,
                alpha     = 0.88,
            )

            for bar, height in zip(bars, heights):
                if not np.isnan(height):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.03,
                        f"{height:.2f}",
                        ha="center", va="bottom",
                        fontsize=7.5, color="black",
                    )

        ax.set_xticks(x_positions + bar_width)
        ax.set_xticklabels(method_order_labels, fontsize=9.5)

        ax.set_ylim(0, 2.0)
        ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.set_ylabel(r"Average $n_{\mathrm{perf}}$", fontsize=10)
        ax.set_xlabel("Method", fontsize=10)

        ax.yaxis.grid(True, linestyle=":", alpha=0.55, zorder=0)
        ax.set_axisbelow(True)

        random_row = summary[
            (summary["method_label"] == "Random")
            & (summary["match_type"]  == "Matched")
        ]
        if not random_row.empty:
            rv = float(random_row["nonstationary_error"].values[0])
            ax.axhline(
                rv,
                color="grey", linestyle="--",
                linewidth=1.4, alpha=0.8,
                label=f"Random mean = {rv:.2f}",
                zorder=5,
            )

        ax.legend(loc="lower center",
                  bbox_to_anchor=(0.5, -0.22),
                  ncol=3, frameon=False, fontsize=8.5)

        ax.set_title(
            f"Setting {setting}: {SETTING_LABELS.get(setting, '')}\n"
            r"Matched vs Partially matched vs Mismatched"
            f"\n(fixed split level 4 — total obs = {int(max_obs)})",
            fontsize=10, pad=8,
        )

        plt.tight_layout(rect=[0, 0.12, 1, 1])

        out = os.path.join(output_dir,
                           f"bar_matched_vs_mismatched_setting{setting}")
        safe_savefig(out)
        plt.close(fig)
        print(f"  Saved: {out}.pdf/.png")



# Plot 4 — Representative line plots (spatio-temporal, matched)
def plot_line_plots(df, output_dir):
    """
    One figure per setting.
    Shows convergence of nperf and sperf with total sample size.

    Uses:
        structure    = space_time
        segmentation = spatio-temporal
        split        = (4,4,4)
    """
    os.makedirs(output_dir, exist_ok=True)

    target_split = (4, 4, 4)

    for setting in sorted(df["setting"].unique()):

        subdf = df[
            (df["setting"]      == setting)
            & (df["structure"]    == "space_time")
            & (df["segmentation"] == "spatio-temporal")
            & (df["split_tuple"]  == target_split)
        ].copy()

        if subdf.empty:
            print(f"  [Line plot] No data for setting {setting}")
            continue

        summary = (
            subdf
            .groupby(["method", "total_obs"])[
                ["nonstationary_error", "stationary_error"]
            ]
            .mean()
            .reset_index()
        )

        fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.0), sharex=True)

        for method in METHODS:
            tmp = (
                summary[summary["method"] == method]
                .sort_values("total_obs")
            )
            if tmp.empty:
                continue

            label  = METHOD_LABELS[method]
            alpha  = 0.55 if method == "random" else 0.95
            lw     = 1.1  if method == "random" else 1.5
            marker = MARKERS[method]

            axes[0].plot(
                tmp["total_obs"], tmp["nonstationary_error"],
                marker=marker, linewidth=lw, alpha=alpha, label=label,
            )
            axes[1].plot(
                tmp["total_obs"], tmp["stationary_error"],
                marker=marker, linewidth=lw, alpha=alpha, label=label,
            )

        for ax in axes:
            ax.set_ylim(-0.05, 2.0)
            ax.yaxis.grid(True, linestyle=":", alpha=0.6)

        axes[0].set_ylabel(r"$n_{\mathrm{perf}}$", fontsize=10)
        axes[1].set_ylabel(r"$s_{\mathrm{perf}}$", fontsize=10)
        axes[1].set_xlabel(
            "Total observations = locations × times", fontsize=10
        )

        axes[0].set_title(
            f"Setting {setting}: {SETTING_LABELS.get(setting, '')}\n"
            "Spatio-temporal structure, spatio-temporal segmentation (4,4,4)",
            fontsize=10,
        )

        axes[0].legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.35),
            ncol=3, frameon=False,
        )

        plt.tight_layout()

        out = os.path.join(output_dir,
                           f"line_plot_setting{setting}_spacetime_44")
        safe_savefig(out)
        plt.close()
        print(f"  Saved: {out}.pdf/.png")



# Plot  — Matched condition line plots for all three structures
def plot_matched_line_plots(df, output_dir):
    """
    One figure per setting — 4 figures total.

    Each figure:
        Rows    : 2  (top = nperf, bottom = sperf)
        Columns : 3  (one per matched condition)
            Col 1: Spatial structure    + spatial segmentation    (4,4,1)
            Col 2: Temporal structure   + temporal segmentation   (1,1,4)
            Col 3: Spatio-temporal str. + spatio-temporal seg.    (4,4,4)

    x-axis : total observations
    Lines  : one per method
    """
    os.makedirs(output_dir, exist_ok=True)

    matched_conditions = [
        ("space",      "spatial",         (4, 4, 1)),
        ("time",       "temporal",        (1, 1, 4)),
        ("space_time", "spatio-temporal", (4, 4, 4)),
    ]

    col_titles = [
        "Spatial structure\nspatial segmentation $(4,4,1)$",
        "Temporal structure\ntemporal segmentation $(1,1,4)$",
        "Spatio-temporal structure\nspatio-temporal segmentation $(4,4,4)$",
    ]

    for setting in sorted(df["setting"].unique()):

        fig, axes = plt.subplots(
            2, 3,
            figsize=(12.0, 6.5),
            sharex=False,
            sharey="row",
            gridspec_kw={"hspace": 0.35, "wspace": 0.22},
        )

        legend_lines  = []
        legend_labels = []

        for col_idx, (struct, seg, split) in enumerate(matched_conditions):

            subdf = df[
                (df["setting"]      == setting)
                & (df["structure"]    == struct)
                & (df["segmentation"] == seg)
                & (df["split_tuple"]  == split)
            ].copy()

            ax_top = axes[0, col_idx]
            ax_bot = axes[1, col_idx]

            if subdf.empty:
                ax_top.text(0.5, 0.5, "No data",
                            ha="center", va="center",
                            transform=ax_top.transAxes, fontsize=9)
                ax_bot.text(0.5, 0.5, "No data",
                            ha="center", va="center",
                            transform=ax_bot.transAxes, fontsize=9)
                ax_top.set_title(col_titles[col_idx], fontsize=9, pad=6)
                continue

            summary = (
                subdf
                .groupby(["method", "total_obs"])[
                    ["nonstationary_error", "stationary_error"]
                ]
                .mean()
                .reset_index()
            )

            x_vals = sorted(summary["total_obs"].unique())

            for method in METHODS:
                tmp = (
                    summary[summary["method"] == method]
                    .sort_values("total_obs")
                )
                if tmp.empty:
                    continue

                label  = METHOD_LABELS[method]
                alpha  = 0.60 if method == "random" else 0.95
                lw     = 1.0  if method == "random" else 1.4
                marker = MARKERS[method]
                ls     = "--" if method == "random" else "-"
                colour = "grey" if method == "random" else None

                line, = ax_top.plot(
                    tmp["total_obs"],
                    tmp["nonstationary_error"],
                    marker=marker,
                    linestyle=ls,
                    linewidth=lw,
                    alpha=alpha,
                    label=label,
                    color=colour,
                )

                ax_bot.plot(
                    tmp["total_obs"],
                    tmp["stationary_error"],
                    marker=marker,
                    linestyle=ls,
                    linewidth=lw,
                    alpha=alpha,
                    label=label,
                    color=colour,
                )

                if col_idx == 0:
                    legend_lines.append(line)
                    legend_labels.append(label)

            for ax in [ax_top, ax_bot]:
                ax.set_ylim(-0.05, 2.05)
                ax.set_xticks(x_vals)
                ax.set_xticklabels(
                    [str(int(v)) for v in x_vals],
                    rotation=30,
                    ha="right",
                    fontsize=7.5,
                )
                ax.yaxis.grid(True, linestyle=":", alpha=0.55)
                ax.xaxis.grid(True, linestyle=":", alpha=0.3)

            ax_top.set_title(col_titles[col_idx], fontsize=9, pad=6)

            if col_idx == 0:
                ax_top.set_ylabel(r"$n_{\mathrm{perf}}$", fontsize=10)
                ax_bot.set_ylabel(r"$s_{\mathrm{perf}}$", fontsize=10)

            ax_bot.set_xlabel("Total observations", fontsize=8)

        fig.legend(
            legend_lines,
            legend_labels,
            loc="lower center",
            ncol=5,
            bbox_to_anchor=(0.5, -0.04),
            frameon=False,
            fontsize=9,
            columnspacing=1.2,
            handletextpad=0.5,
        )

        fig.suptitle(
            f"Setting {setting}: {SETTING_LABELS.get(setting, '')}"
            r" — $n_{\mathrm{perf}}$ and $s_{\mathrm{perf}}$"
            " under matched segmentation (fixed split level 4)",
            fontsize=11,
            y=1.01,
        )

        plt.tight_layout()

        out = os.path.join(
            output_dir,
            f"matched_line_plots_setting{setting}",
        )
        safe_savefig(out)
        plt.close(fig)
        print(f"  Saved: {out}.pdf/.png")



# Plot 5 — Boxplot nperf under matched segmentation
def plot_boxplot_nperf(raw_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    max_obs = raw_df["total_obs"].max()

    subdf = raw_df[
        (raw_df["total_obs"]    == max_obs)
        & (raw_df["segmentation"] != "other")
    ].copy()

    subdf = subdf[subdf.apply(is_matched, axis=1)].copy()
    subdf["method_label"] = subdf["method"].map(METHOD_LABELS)

  
    methods_no_random = [m for m in METHODS if m != "random"]
    method_order      = [METHOD_LABELS[m] for m in methods_no_random]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    axes = axes.flatten()

    for ax, setting in zip(axes, [1, 2, 3, 4]):
        tmp = subdf[subdf["setting"] == setting].copy()

        
        tmp_no_random = tmp[tmp["method"] != "random"]
        sns.boxplot(
            data=tmp_no_random,
            x="method_label",
            y="nonstationary_error",
            order=method_order,
            ax=ax,
            showfliers=True,
            color="#a8d8ea",
            linewidth=1.2,
            flierprops=dict(
                marker="o",
                markerfacecolor="#3a7ebf",
                markeredgecolor="#3a7ebf",
                markersize=3,
                alpha=0.45,
            ),
        )

    
        random_mean = tmp[
            tmp["method"] == "random"
        ]["nonstationary_error"].mean()
        ax.axhline(
            random_mean,
            color="grey", linestyle="--",
            linewidth=1.4, alpha=0.85,
            zorder=5,
        )

        ax.set_title(
            f"Setting {setting}: {SETTING_LABELS[setting]}",
            fontsize=10,
        )
        ax.set_xlabel("")
        ax.set_ylabel(
            r"$n_{\mathrm{perf}}$" if setting in [1, 3] else "",
            fontsize=10,
        )
        ax.set_ylim(-0.05, 2.1)
        ax.tick_params(axis="x", rotation=25)
        ax.yaxis.grid(True, linestyle=":", alpha=0.6)


    plt.tight_layout()
    out = os.path.join(output_dir, "boxplot_nperf_matched")
    safe_savefig(out)
    plt.close()
    print(f"  Saved: {out}.pdf/.png")


def plot_boxplot_sperf(raw_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    max_obs = raw_df["total_obs"].max()

    subdf = raw_df[
        (raw_df["total_obs"]    == max_obs)
        & (raw_df["segmentation"] != "other")
    ].copy()

    subdf = subdf[subdf.apply(is_matched, axis=1)].copy()
    subdf["method_label"] = subdf["method"].map(METHOD_LABELS)

    methods_no_random = [m for m in METHODS if m != "random"]
    method_order      = [METHOD_LABELS[m] for m in methods_no_random]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    axes = axes.flatten()

    for ax, setting in zip(axes, [1, 2, 3, 4]):
        tmp = subdf[subdf["setting"] == setting].copy()

        tmp_no_random = tmp[tmp["method"] != "random"]
        sns.boxplot(
            data=tmp_no_random,
            x="method_label",
            y="stationary_error",
            order=method_order,
            ax=ax,
            showfliers=True,
            color="#a8d8ea",
            linewidth=1.2,
            flierprops=dict(
                marker="o",
                markerfacecolor="#3a7ebf",
                markeredgecolor="#3a7ebf",
                markersize=3,
                alpha=0.45,
            ),
        )

       
        random_mean = tmp[
            tmp["method"] == "random"
        ]["stationary_error"].mean()
        ax.axhline(
            random_mean,
            color="grey", linestyle="--",
            linewidth=1.4, alpha=0.85,
            zorder=5,
        )

        ax.set_title(
            f"Setting {setting}: {SETTING_LABELS[setting]}",
            fontsize=10,
        )
        ax.set_xlabel("")
        ax.set_ylabel(
            r"$s_{\mathrm{perf}}$" if setting in [1, 3] else "",
            fontsize=10,
        )
        ax.set_ylim(-0.05, 2.1)
        ax.tick_params(axis="x", rotation=25)
        ax.yaxis.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        r"Distribution of $s_{\mathrm{perf}}$ under matched segmentation"
        f"\n(total obs = {int(max_obs)}, 500 repetitions  |  "
        "dashed line = Random mean baseline  |  "
        "box height = estimation consistency)",
        fontsize=11, y=1.02,
    )

    plt.tight_layout()
    out = os.path.join(output_dir, "boxplot_sperf_matched")
    safe_savefig(out)
    plt.close()
    print(f"  Saved: {out}.pdf/.png")



# Plot 7 — Method ranking plot per setting
def plot_method_ranking(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    max_obs = df["total_obs"].max()

    wanted_splits = {
        "spatial":         (4, 4, 1),
        "temporal":        (1, 1, 4),
        "spatio-temporal": (4, 4, 4),
    }

    subdf = df[
        (df["total_obs"]    == max_obs)
        & (df["segmentation"] != "other")
    ].copy()

    subdf = subdf[
        subdf.apply(
            lambda row: row["split_tuple"] == wanted_splits.get(
                row["segmentation"]
            ),
            axis=1,
        )
    ].copy()

    subdf = subdf[subdf.apply(is_matched, axis=1)].copy()

    if subdf.empty:
        print("  [Ranking plot] No data found.")
        return

    cmap = plt.get_cmap("viridis")

   
    methods_no_random = [m for m in METHODS if m != "random"]


    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(
        2, 3,
        figure=fig,
        hspace=0.45, wspace=0.40,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1:])   # Panel 5 spans last 2 columns
    axes_flat = [ax1, ax2, ax3, ax4, None, None]  # placeholder for loop

    for panel_idx, setting in enumerate([1, 2, 3, 4]):

        ax         = axes_flat[panel_idx]
        setting_df = subdf[subdf["setting"] == setting].copy()

        if setting_df.empty:
            ax.set_visible(False)
            continue

        # Average nperf per method across the three matched structures
        summary = (
            setting_df
            .groupby("method")["nonstationary_error"]
            .mean()
            .reset_index()
        )
        summary["method_label"] = summary["method"].map(METHOD_LABELS)

       
        random_val = float(
            summary.loc[
                summary["method"] == "random", "nonstationary_error"
            ].values[0]
        )

      
        summary_no_random = summary[
            summary["method"] != "random"
        ].sort_values("nonstationary_error", ascending=True
        ).reset_index(drop=True)
        summary_no_random["rank"] = np.arange(
            1, len(summary_no_random) + 1
        )

        max_val = 2.0
        colours = [
            cmap(1.0 - val / max_val)
            for val in summary_no_random["nonstationary_error"]
        ]

        bars = ax.barh(
            y         = summary_no_random["method_label"],
            width     = summary_no_random["nonstationary_error"],
            color     = colours,
            edgecolor = "white",
            linewidth = 0.8,
            height    = 0.55,
        )

        # Value labels
        for bar, val in zip(bars, summary_no_random["nonstationary_error"]):
            ax.text(
                val + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center", ha="left",
                fontsize=8.5, color="black",
            )

        # Rank labels inside bars
        for bar, rank in zip(bars, summary_no_random["rank"]):
            bw = bar.get_width()
            if bw > 0.08:
                ax.text(
                    bw * 0.05,
                    bar.get_y() + bar.get_height() / 2,
                    f"#{rank}",
                    va="center", ha="left",
                    fontsize=8, color="white", fontweight="bold",
                )

       
        ax.axvline(
            random_val,
            color="grey", linestyle="--",
            linewidth=1.2, alpha=0.8,
            label=f"Random mean = {random_val:.2f}",
        )
        ax.legend(fontsize=7.5, loc="lower center",
                  bbox_to_anchor=(0.5, -0.18),
                  ncol=1, frameon=False)

        ax.set_xlim(0, 2.15)
        ax.set_xlabel(
            r"Average $n_{\mathrm{perf}}$  (lower = better)",
            fontsize=9,
        )
        ax.set_title(
            f"Setting {setting}: {SETTING_LABELS[setting]}",
            fontsize=10, pad=6,
        )
        ax.xaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.invert_yaxis()

   
    overall = (
        subdf
        .groupby("method")["nonstationary_error"]
        .mean()
        .reset_index()
    )
    overall["method_label"] = overall["method"].map(METHOD_LABELS)

    random_overall = float(
        overall.loc[
            overall["method"] == "random", "nonstationary_error"
        ].values[0]
    )

    overall_no_random = overall[
        overall["method"] != "random"
    ].sort_values("nonstationary_error", ascending=True
    ).reset_index(drop=True)
    overall_no_random["rank"] = np.arange(
        1, len(overall_no_random) + 1
    )

    max_val = 2.0
    colours_overall = [
        cmap(1.0 - val / max_val)
        for val in overall_no_random["nonstationary_error"]
    ]

    bars5 = ax5.barh(
        y         = overall_no_random["method_label"],
        width     = overall_no_random["nonstationary_error"],
        color     = colours_overall,
        edgecolor = "white",
        linewidth = 0.8,
        height    = 0.55,
    )

    # Value labels
    for bar, val in zip(bars5,
                        overall_no_random["nonstationary_error"]):
        ax5.text(
            val + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center", ha="left",
            fontsize=8.5, color="black",
        )

    # Rank labels inside bars
    for bar, rank in zip(bars5, overall_no_random["rank"]):
        bw = bar.get_width()
        if bw > 0.08:
            ax5.text(
                bw * 0.05,
                bar.get_y() + bar.get_height() / 2,
                f"#{rank}",
                va="center", ha="left",
                fontsize=8, color="white", fontweight="bold",
            )

    # Random as vertical dashed line
    ax5.axvline(
        random_overall,
        color="grey", linestyle="--",
        linewidth=1.4, alpha=0.85,
        label=f"Random mean = {random_overall:.2f}",
    )
    ax5.legend(fontsize=8, loc="lower center",
               bbox_to_anchor=(0.5, -0.18),
               ncol=1, frameon=False)

    ax5.set_xlim(0, 2.15)
    ax5.set_xlabel(
        r"Average $n_{\mathrm{perf}}$  (lower = better)",
        fontsize=9,
    )
    ax5.set_title(
        "Overall — averaged across all 4 settings\n"
        "Use this panel to choose your method",
        fontsize=10, pad=6,
        color="#185fa5", fontweight="bold",
    )
    ax5.xaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
    ax5.set_axisbelow(True)
    ax5.invert_yaxis()

   
    for spine in ax5.spines.values():
        spine.set_edgecolor("#185fa5")
        spine.set_linewidth(2.0)
        spine.set_visible(True)

    
   

    fig.suptitle(
        r"Method ranking by average $n_{\mathrm{perf}}$ under matched segmentation"
        f"\n(fixed split level 4, total obs = {int(max_obs)}, averaged across nonstationarity structures)\n"
        "Panel 5 (blue border) = overall ranking — use this to choose your method",
        fontsize=11, y=1.02,
    )

    plt.tight_layout()

    out = os.path.join(output_dir, "method_ranking")
    safe_savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}.pdf/.png")



# Plot 8 — Rectangle convergence plot
def plot_rectangle_convergence(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    matched_conditions = [
        ("space",      "spatial",         (4, 4, 1)),
        ("time",       "temporal",        (1, 1, 4)),
        ("space_time", "spatio-temporal", (4, 4, 4)),
    ]

    col_titles = [
        "Spatial structure\nspatial segmentation $(4,4,1)$",
        "Temporal structure\ntemporal segmentation $(1,1,4)$",
        "Spatio-temporal structure\nspatio-temporal segmentation $(4,4,4)$",
    ]

    # One colour per method
    _palette        = sns.color_palette("colorblind", n_colors=len(METHODS))
    method_colours  = {m: _palette[i] for i, m in enumerate(METHODS)}

    time_values = sorted(df["times"].dropna().unique())
    loc_values  = sorted(df["locations"].dropna().unique())

    # Solid for fewer time points, dashed for more
    time_linestyles = {}
    for i, t in enumerate(time_values):
        time_linestyles[t] = "-" if i == 0 else "--"

    time_labels = {t: f"time={int(t)}" for t in time_values}

    for setting in sorted(df["setting"].unique()):

        fig, axes = plt.subplots(
            2, 3,
            figsize=(13.0, 7.0),
            sharey="row",
            gridspec_kw={"hspace": 0.45, "wspace": 0.22},
        )

        legend_handles     = []
        legend_labels_list = []

        for col_idx, (struct, seg, split) in enumerate(matched_conditions):

            subdf = df[
                (df["setting"]      == setting)
                & (df["structure"]    == struct)
                & (df["segmentation"] == seg)
                & (df["split_tuple"]  == split)
            ].copy()

            ax_top = axes[0, col_idx]
            ax_bot = axes[1, col_idx]

            if subdf.empty:
                ax_top.text(0.5, 0.5, "No data",
                            ha="center", va="center",
                            transform=ax_top.transAxes, fontsize=9)
                ax_top.set_title(col_titles[col_idx], fontsize=9, pad=6)
                continue

            summary = (
                subdf
                .groupby(["method", "locations", "times"])[
                    ["nonstationary_error", "stationary_error"]
                ]
                .mean()
                .reset_index()
            )

            for method in METHODS:
                colour = method_colours[method]
                marker = MARKERS[method]
                label  = METHOD_LABELS[method]
                alpha  = 0.55 if method == "random" else 0.92
                lw     = 1.0  if method == "random" else 1.6

                for t_val in time_values:
                    tmp = (
                        summary[
                            (summary["method"] == method)
                            & (summary["times"]  == t_val)
                        ]
                        .sort_values("locations")
                    )
                    if tmp.empty:
                        continue

                    ls = time_linestyles[t_val]

                    line_top, = ax_top.plot(
                        tmp["locations"],
                        tmp["nonstationary_error"],
                        color=colour, marker=marker,
                        linestyle=ls, linewidth=lw, alpha=alpha,
                    )
                    ax_bot.plot(
                        tmp["locations"],
                        tmp["stationary_error"],
                        color=colour, marker=marker,
                        linestyle=ls, linewidth=lw, alpha=alpha,
                    )

                    if col_idx == 0:
                        legend_handles.append(line_top)
                        legend_labels_list.append(
                            f"{label}  ({time_labels[t_val]})"
                        )

            for ax in [ax_top, ax_bot]:
                ax.set_xticks(loc_values)
                ax.set_xticklabels(
                    [str(int(v)) for v in loc_values], fontsize=9
                )
                ax.set_xlim(
                    loc_values[0]  - 15,
                    loc_values[-1] + 15,
                )
                ax.set_ylim(-0.05, 2.05)
                ax.yaxis.grid(True, linestyle=":", alpha=0.55)

            ax_top.set_title(col_titles[col_idx], fontsize=9, pad=6)

            if col_idx == 0:
                ax_top.set_ylabel(r"$n_{\mathrm{perf}}$", fontsize=10)
                ax_bot.set_ylabel(r"$s_{\mathrm{perf}}$", fontsize=10)

            ax_bot.set_xlabel("Number of locations", fontsize=9)

        fig.text(
            0.5, -0.01,
            "How to read:  "
            "left -> right = more locations (time fixed)   |   "
            "solid -> dashed = more time (locations fixed)   |   "
            "bottom-left solid -> top-right dashed = both increase",
            ha="center", fontsize=7.5,
            color="dimgrey", style="italic",
        )

        fig.legend(
            legend_handles, legend_labels_list,
            loc="lower center", ncol=5,
            bbox_to_anchor=(0.5, -0.09),
            frameon=False, fontsize=8.5,
            columnspacing=1.0, handletextpad=0.5,
        )

        fig.suptitle(
            f"Setting {setting}: {SETTING_LABELS.get(setting, '')}"
            r" — Effect of increasing locations vs time points"
            "\n"
            r"$n_{\mathrm{perf}}$ (top) and $s_{\mathrm{perf}}$ (bottom)"
            "  |  matched segmentation, fixed split level 4",
            fontsize=11, y=1.03,
        )

        plt.tight_layout()

        out = os.path.join(
            output_dir,
            f"rectangle_convergence_setting{setting}",
        )
        safe_savefig(out)
        plt.close(fig)
        print(f"  Saved: {out}.pdf/.png")



# Plot 9 — 2x2 Heatmap Grid (rectangle sample size effect)
def plot_rectangle_heatmap(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    loc_vals  = sorted(df["locations"].dropna().unique())
    time_vals = sorted(df["times"].dropna().unique())

    matched_conditions = [
        ("space",      "spatial",         (4, 4, 1)),
        ("time",       "temporal",        (1, 1, 4)),
        ("space_time", "spatio-temporal", (4, 4, 4)),
    ]
    col_titles = [
        "Spatial structure\nspatial seg. $(4,4,1)$",
        "Temporal structure\ntemporal seg. $(1,1,4)$",
        "Spatio-temporal structure\nspatio-temporal seg. $(4,4,4)$",
    ]

    cmap = plt.get_cmap("viridis_r")

    # Remove Random
    methods_no_random = [m for m in METHODS if m != "random"]

    for setting in sorted(df["setting"].unique()):

        fig, axes = plt.subplots(
            len(methods_no_random), 3,
            figsize=(11, 9),
            gridspec_kw={"hspace": 0.60, "wspace": 0.35},
        )
        fig.subplots_adjust(bottom=0.10)

        for col_idx, (struct, seg, split) in enumerate(matched_conditions):

            subdf = df[
                (df["setting"]      == setting)
                & (df["structure"]    == struct)
                & (df["segmentation"] == seg)
                & (df["split_tuple"]  == split)
            ].copy()

            summary = (
                subdf
                .groupby(["method", "locations", "times"])[
                    "nonstationary_error"
                ]
                .mean()
                .reset_index()
                if not subdf.empty else pd.DataFrame()
            )

            for row_idx, method in enumerate(methods_no_random):

                ax = axes[row_idx, col_idx]

                mat = np.full((len(time_vals), len(loc_vals)), np.nan)
                if not summary.empty:
                    tmp = summary[summary["method"] == method]
                    for ti, t in enumerate(time_vals):
                        for li, loc in enumerate(loc_vals):
                            cell = tmp[
                                (tmp["times"]     == t)
                                & (tmp["locations"] == loc)
                            ]
                            if not cell.empty:
                                mat[ti, li] = float(
                                    cell["nonstationary_error"].values[0]
                                )

                ax.imshow(
                    mat, cmap=cmap, vmin=0, vmax=2.0, aspect="auto"
                )

                # Cell borders — lines between the 4 cells
                ax.axvline(0.5, color="black", linewidth=1.8, zorder=5)
                ax.axhline(0.5, color="black", linewidth=1.8, zorder=5)

                # Outer border frame
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("black")
                    spine.set_linewidth(1.8)

                # Annotate cells
                for ti in range(len(time_vals)):
                    for li in range(len(loc_vals)):
                        val = mat[ti, li]
                        if not np.isnan(val):
                            txt_color = "black" if val < 0.6 else "white"
                            ax.text(
                                li, ti, f"{val:.2f}",
                                ha="center", va="center",
                                fontsize=8.5, color=txt_color,
                                fontweight="bold", zorder=6,
                            )

                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])

                if row_idx == len(methods_no_random) - 1:
                    ax.set_xticklabels(
                        [f"{int(l)}\nloc" for l in loc_vals],
                        fontsize=7.5,
                    )
                else:
                    ax.set_xticklabels(["", ""])

                if col_idx == 0:
                    ax.set_yticklabels(
                        [f"t={int(t)}" for t in time_vals],
                        fontsize=7.5,
                    )
                    ax.set_ylabel(
                        METHOD_LABELS[method],
                        fontsize=9, rotation=90, labelpad=4,
                    )
                else:
                    ax.set_yticklabels(["", ""])

                if row_idx == 0:
                    ax.set_title(col_titles[col_idx], fontsize=8.5, pad=6)

        # Shared colourbar — placed at the bottom
        cbar_ax = fig.add_axes([0.15, 0.02, 0.70, 0.015])
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=0, vmax=2.0)
        )
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cb.set_label(
            r"$n_{\mathrm{perf}}$ (bright = low error = good)",
            fontsize=8,
        )
        cb.ax.tick_params(labelsize=7)


        out = os.path.join(
            output_dir, f"rectangle_heatmap_setting{setting}"
        )
        safe_savefig(out)
        plt.close(fig)
        print(f"  Saved: {out}.pdf/.png")



# Plot 10 — Split level line plots
def plot_split_level_lines(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    max_obs = df["total_obs"].max()

    SPLIT_LEVELS = [2, 3, 4]

    SPLIT_TUPLES = {
        "spatial": {
            2: (2, 2, 1), 3: (3, 3, 1), 4: (4, 4, 1),
        },
        "temporal": {
            2: (1, 1, 2), 3: (1, 1, 3), 4: (1, 1, 4),
        },
        "spatio-temporal": {
            2: (2, 2, 2), 3: (3, 3, 3), 4: (4, 4, 4),
        },
    }

    XTICK_LABELS = {
        "spatial": {
            2: "level 2\n$(2,2,1)$",
            3: "level 3\n$(3,3,1)$",
            4: "level 4\n$(4,4,1)$",
        },
        "temporal": {
            2: "level 2\n$(1,1,2)$",
            3: "level 3\n$(1,1,3)$",
            4: "level 4\n$(1,1,4)$",
        },
        "spatio-temporal": {
            2: "level 2\n$(2,2,2)$",
            3: "level 3\n$(3,3,3)$",
            4: "level 4\n$(4,4,4)$",
        },
    }

    MATCHED_STRUCTURE = {
        "spatial":         "space",
        "temporal":        "time",
        "spatio-temporal": "space_time",
    }

    COL_TITLES_SL = {
        "spatial":         "Spatial segmentation",
        "temporal":        "Temporal segmentation",
        "spatio-temporal": "Spatio-temporal segmentation",
    }

    _pal = sns.color_palette("colorblind", n_colors=len(METHODS))
    METHOD_COLOURS_SL = {m: _pal[i] for i, m in enumerate(METHODS)}

    seg_types = ["spatial", "temporal", "spatio-temporal"]

    for setting in sorted(df["setting"].unique()):

        fig, axes = plt.subplots(
            2, 3,
            figsize=(12.5, 6.5),
            sharey="row",
            sharex=False,
            gridspec_kw={"hspace": 0.42, "wspace": 0.18},
        )

        legend_lines  = []
        legend_labels = []

        for col_idx, seg in enumerate(seg_types):

            struct = MATCHED_STRUCTURE[seg]
            ax_top = axes[0, col_idx]
            ax_bot = axes[1, col_idx]

            for method in METHODS:
                colour = "grey" if method == "random" else METHOD_COLOURS_SL[method]
                marker = MARKERS[method]
                label  = METHOD_LABELS[method]
                ls     = "--" if method == "random" else "-"
                lw     = 1.0  if method == "random" else 1.7
                alpha  = 0.55 if method == "random" else 0.95
                ms     = 5    if method == "random" else (6 if method != "stcomb" else 9)

                nperf_vals = []
                sperf_vals = []

                for level in SPLIT_LEVELS:
                    split = SPLIT_TUPLES[seg][level]
                    cell  = df[
                        (df["setting"]      == setting)
                        & (df["structure"]    == struct)
                        & (df["segmentation"] == seg)
                        & (df["split_tuple"]  == split)
                        & (df["total_obs"]    == max_obs)
                        & (df["method"]       == method)
                    ]
                    nperf_vals.append(
                        float(cell["nonstationary_error"].mean())
                        if not cell.empty else np.nan
                    )
                    sperf_vals.append(
                        float(cell["stationary_error"].mean())
                        if not cell.empty else np.nan
                    )

                line_top, = ax_top.plot(
                    SPLIT_LEVELS, nperf_vals,
                    color=colour, marker=marker, linestyle=ls,
                    linewidth=lw, alpha=alpha, markersize=ms,
                    label=label,
                )
                ax_bot.plot(
                    SPLIT_LEVELS, sperf_vals,
                    color=colour, marker=marker, linestyle=ls,
                    linewidth=lw, alpha=alpha, markersize=ms,
                    label=label,
                )

                if col_idx == 0:
                    legend_lines.append(line_top)
                    legend_labels.append(label)

            tick_labels = [
                XTICK_LABELS[seg][level] for level in SPLIT_LEVELS
            ]
            for ax in [ax_top, ax_bot]:
                ax.set_xticks(SPLIT_LEVELS)
                ax.set_xticklabels(
                    tick_labels, fontsize=8.5, linespacing=1.4
                )
                ax.set_xlim(1.6, 4.4)
                ax.set_ylim(-0.05, 2.05)
                ax.yaxis.grid(True, linestyle=":", alpha=0.50)
                ax.xaxis.grid(True, linestyle=":", alpha=0.25)

            ax_top.set_title(
                COL_TITLES_SL[seg], fontsize=10, pad=7
            )

            if col_idx == 2:
                ax_top.annotate(
                    "Nonstationary",
                    xy=(1.02, 0.5), xycoords="axes fraction",
                    fontsize=9, rotation=270,
                    va="center", ha="left", color="#444444",
                )
                ax_bot.annotate(
                    "Stationary",
                    xy=(1.02, 0.5), xycoords="axes fraction",
                    fontsize=9, rotation=270,
                    va="center", ha="left", color="#444444",
                )

            if col_idx == 0:
                ax_top.set_ylabel(r"$n_{\mathrm{perf}}$", fontsize=11)
                ax_bot.set_ylabel(r"$s_{\mathrm{perf}}$", fontsize=11)

            ax_bot.set_xlabel("Split level", fontsize=9)

        fig.legend(
            legend_lines, legend_labels,
            loc="lower center", ncol=5,
            bbox_to_anchor=(0.5, -0.06),
            frameon=False, fontsize=9.5,
            columnspacing=1.5, handletextpad=0.5,
        )


        plt.tight_layout()

        out = os.path.join(
            output_dir, f"split_level_lines_setting{setting}"
        )
        safe_savefig(out)
        plt.close(fig)
        print(f"  Saved: {out}.pdf/.png")


def main():
    avg_csv = "all_settings_results.csv"
    raw_csv = "all_settings_results_raw.csv"

    for path in [avg_csv, raw_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\nCannot find '{path}'.\n"
                "Please make sure it is in the same folder as this script."
            )

    print("\nLoading data...")
    df     = pd.read_csv(avg_csv)
    raw_df = pd.read_csv(raw_csv)

    df     = prepare_dataframe(df)
    raw_df = prepare_dataframe(raw_df)

    print(f"  Averaged CSV : {len(df)} rows")
    print(f"  Raw CSV      : {len(raw_df)} rows")
    print(f"  Settings     : {sorted(df['setting'].unique())}")
    print(f"  Methods      : {sorted(df['method'].unique())}")
    print(f"  Sample sizes : {sorted(df['total_obs'].unique())}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nSaving all plots to: {OUTPUT_DIR}/")

  
    # Plot 1 — Grand summary heatmap (split levels 3 and 4)
    print("\n--- Plot 1: Grand summary heatmaps ---")
    plot_grand_summary(df, split_level=3, output_dir=OUTPUT_DIR)
    plot_grand_summary(df, split_level=4, output_dir=OUTPUT_DIR)

 
    # Plot 2 — segmentation sensitivity (all settings)
    print("\n--- Plot 2: Segmentation sensitivity heatmaps ---")
    for setting in [1, 2, 3, 4]:
        plot_type_b(df, setting, OUTPUT_DIR)

  
    # Plot 3 — Matched / Partially matched / Mismatched bar charts
    print("\n--- Plot 3: Matched vs mismatched bar charts ---")
    plot_matched_bars(df, OUTPUT_DIR)


    # Plot 4 — Representative line plots (spatio-temporal only)
    print("\n--- Plot 4: Representative line plots (spatio-temporal) ---")
    plot_line_plots(df, OUTPUT_DIR)

  
    # Matched condition line plots (all three structures)
    print("\n--- Plot 4b: Matched condition line plots (all structures) ---")
    plot_matched_line_plots(df, OUTPUT_DIR)


    # Plot 5 — Boxplot nperf
    print("\n--- Plot 5: Boxplot nperf ---")
    plot_boxplot_nperf(raw_df, OUTPUT_DIR)

  
    # Plot 6 — Boxplot sperf
    print("\n--- Plot 6: Boxplot sperf ---")
    plot_boxplot_sperf(raw_df, OUTPUT_DIR)

  
    # Plot 7 — Method ranking plot
    print("\n--- Plot 7: Method ranking plot ---")
    plot_method_ranking(df, OUTPUT_DIR)

   
    # Plot 8 — Rectangle convergence plot
    print("\n--- Plot 8: Rectangle convergence plots ---")
    plot_rectangle_convergence(df, OUTPUT_DIR)

   
    # Plot 9 — Rectangle heatmap (sample size effect)
    print("\n--- Plot 9: Rectangle heatmap (2x2 grid) ---")
    plot_rectangle_heatmap(df, OUTPUT_DIR)

 
    # Plot 10 — Split level line plots
    print("\n--- Plot 10: Split level line plots ---")
    plot_split_level_lines(df, OUTPUT_DIR)

    print(f"\nAll plots saved to: {OUTPUT_DIR}/")
    print("\nFiles produced:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".pdf"):
            print(f"  {f}")


if __name__ == "__main__":
    main()