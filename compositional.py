import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from spatial import partition_points_by_polygons
from utils import standardize_data
from rank_estimation import ssa_procedure_from_segs, augmented_eigenvector_estimator, normalized_scree_plot
from ssa import SSA
from functools import partial


def clr(data):
    log_data = np.log(data)
    p, n = data.shape
    left_mul = np.identity(p) - 1 / p * np.ones((p, p))
    right_mul = np.identity(n) - 1 / n * np.ones((n, n))
    clr_data = left_mul @ log_data @ right_mul
    return clr_data


def ilr(data):
    p = data.shape[0]
    V = np.zeros((p, p - 1))
    for i in range(1, p):
        v_i = np.zeros(p)
        v_i[0:i] = 1 / i
        v_i[i] = -1
        V[:, i - 1] = np.sqrt(i / (i + 1)) * v_i

    data_ilr = V.T @ data
    return data_ilr


# Precompute shared percentiles for all vals arrays
def compute_percentile_bins(all_vals, percentiles=None):
    if percentiles is None:
        percentiles = [0, 5, 25, 45, 55, 75, 95, 100]
    combined = np.concatenate(all_vals)
    bins = np.percentile(combined, percentiles)
    return bins, percentiles

# Modified create_map to accept precomputed bins and only draw legend if requested
def create_map(ax=None, polygons=None, vals=None, bins=None, percentiles=None, show_legend=True):
    if ax is None:
        fig, ax = plt.subplots(frameon=False)

    # Load data
    borders = pd.read_csv("data/kola/kola_borders.csv")
    boundary = pd.read_csv("data/kola/kola_boundary.csv")
    coast = pd.read_csv("data/kola/kola_coast.csv")
    lakes = pd.read_csv("data/kola/kola_lakes.csv")

    ax.plot(coast.V1, coast.V2, color="#4F4F4F", zorder=1)
    ax.plot(lakes.V1, lakes.V2, color="#B3E5FC", zorder=1)
    ax.plot(borders.V1, borders.V2, color="#81C784", zorder=1)
    ax.plot(boundary.V1, boundary.V2, color="#4DB6AC", zorder=2)

    # Compute bounds
    x_min = min(coast.V1.min(), lakes.V1.min(), borders.V1.min(), boundary.V1.min())
    x_max = max(coast.V1.max(), lakes.V1.max(), borders.V1.max(), boundary.V1.max())
    y_min = min(coast.V2.min(), lakes.V2.min(), borders.V2.min(), boundary.V2.min())
    y_max = max(coast.V2.max(), lakes.V2.max(), borders.V2.max(), boundary.V2.max())

    # Moss data
    df = pd.read_csv("data/kola/moss_data.csv")
    coords = df.values[:, 2:4]

    if vals is None:
        ax.scatter(coords[:, 0], coords[:, 1], marker="x", color="#D3D3D3", zorder=3)

    ax.set_xlim(x_min - 5000, x_max + 5000)
    ax.set_ylim(y_min - 5000, y_max + 5000)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")

    if polygons is not None:
        for polygon in polygons:
            polygon = np.asarray(polygon)
            ax.plot(polygon[:, 0], polygon[:, 1], color="#4F4F4F")

    if vals is not None:
        num_buckets = len(bins) - 1
        bucket_idx = np.digitize(vals, bins, right=False) - 1
        bucket_idx = np.clip(bucket_idx, 0, num_buckets - 1)

        markers = ['o', 's', '^', 'D', 'P', 'H', 'X'][:num_buckets]
        x, y = coords[:, 0], coords[:, 1]

        grays = [
            "#FFFFFF",  # white
            "#D9D9D9",
            "#B3B3B3",
            "#808080",
            "#4D4D4D",
            "#262626",
            "#000000",  # black
        ]

        for b, marker in enumerate(markers):
            mask = bucket_idx == b
            if not np.any(mask):
                continue

            ax.scatter(
                x[mask], y[mask],
                marker=marker,
                s=45,
                facecolor=grays[b],
                edgecolors="black",
                linewidths=0.4,
                label=f'{percentiles[b]}–{percentiles[b + 1]}%' if show_legend else None,
                zorder=10
            )

        if show_legend:
            ax.legend(
                loc='upper right',
                bbox_to_anchor=(1.00, 1.02),
                frameon=True,
                fontsize=15,
                markerscale=2.5,
            )

def read_polygons(poly_file, old=False):
    with open(poly_file) as f:
        data = json.load(f)
    # If old is True, we used the original polygon JSON which has no colors
    # and can thus be returned as is.
    if old:
        return data

    polygons = []
    for item in data:
        poly = np.array(item["vertices"])
        color = item["color"]
        polygons.append(poly)
    return polygons


def plot_part(part, coords):
    borders = pd.read_csv("data/kola/kola_borders.csv")
    boundary = pd.read_csv("data/kola/kola_boundary.csv")
    coast = pd.read_csv("data/kola/kola_coast.csv")
    lakes = pd.read_csv("data/kola/kola_lakes.csv")
    plt.figure(frameon=False)
    plt.plot(coast.V1, coast.V2, color="#4F4F4F")
    plt.plot(lakes.V1, lakes.V2, color="#B3E5FC")
    plt.plot(borders.V1, borders.V2, color="#C8E6C9")
    plt.plot(boundary.V1, boundary.V2, color="#4DB6AC")
    plt.axis('off')
    plt.scatter(coords[part, 0], coords[part, 1], marker="x", color="#D3D3D3")
    plt.show()


def ladle_plots(k_vec, f_vec, phi_vec, g_vec, q_hat):
    # Turn on ggplot-like styling
    plt.style.use("ggplot")

    highlight_idx = q_hat

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)

    # --- First subplot: f ---
    axes[0].plot(k_vec, f_vec, marker="o", color='black')
    axes[0].set_title(r"$f(k)$")

    # --- Second subplot: phi ---
    axes[1].plot(k_vec, phi_vec, marker="o", color='black')
    axes[1].set_title(r"$\Phi(k)$")

    # --- Third subplot: g ---
    axes[2].plot(k_vec, g_vec, marker="o", color='black')
    axes[2].scatter(
        k_vec[highlight_idx],
        g_vec[highlight_idx],
        color="blue",
        zorder=3,
        s=80,
    )
    axes[2].set_title(r"$g(k)$")

    for ax in axes:
        ax.set_xlabel("k")

    fig.tight_layout()
    plt.savefig("plots/ladle.pdf", dpi=800)
    plt.show()


def analyse_kola_data(s=10, r=10):
    df = pd.read_csv("data/kola/moss_data.csv")
    noise_dim = r
    num_trials = s
    cols = ["Ag", "Al", "As", "B", "Ba", "Be", "Bi", "Ca", "Cd", "Co", "Cr", "Cu", "Fe", "Hg", "K", "La", "Mg", "Mn",
            "Mo", "Na",
            "Ni", "P", "Pb", "Rb", "S", "Sb", "Sc", "Se", "Si", "Sr", "Th", "Tl", "U", "V", "Y",
            "Zn"]  # Au, Pd, Pt have NaNs
    values = df[cols].values
    coords = df[["XCOO", "YCOO"]].values

    polygons = read_polygons("polygons1.json", old=True)
    partition = partition_points_by_polygons(coords, polygons)

    values = values.T
    values = ilr(clr(values))

    std_values, whitener = standardize_data(values)

    func = partial(ssa_procedure_from_segs, coords=coords, segs=partition[0], kernel=('sb', 50000))

    # subresults
    f_vec = augmented_eigenvector_estimator(std_values, func, noise_dim, num_trials)
    phi_vec = normalized_scree_plot(func(std_values)[1])
    cum_f_vec = np.cumsum(f_vec)
    g_vec = cum_f_vec + phi_vec

    rank = np.argmin(g_vec)

    ssa_obj = SSA(values, num_non_stationary=rank)
    ss, ns = ssa_obj.comb(coords, partition[0], kernel=('sb', 50000))

    ss_vals = (ss @ std_values)
    ns_vals = (ns @ std_values)

    res_dict = {
        "f_vec": cum_f_vec,
        "phi_vec": phi_vec,
        "g_vec": g_vec,
        "ss_vals": ss_vals,
        "ns_vals": ns_vals,
        "rank": rank
    }

    return res_dict


def plot_kola_results(ns_vals, ss_vals, ss_idx):
    vals_list = [ns_vals[0], ns_vals[1], ns_vals[2], ns_vals[3], ns_vals[4], ss_vals[ss_idx]]
    labels = [r"$\mathbf{n}_1$", r"$\mathbf{n}_2$", r"$\mathbf{n}_3$", r"$\mathbf{n}_4$", r"$\mathbf{n}_5$",
              r"$\mathbf{{s}}_{21}$"]

    # Compute shared bins
    bins, percentiles = compute_percentile_bins(vals_list)

    # Create subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    axes = axes.ravel()  # flatten to 1D for easy iteration

    # Plot each
    for i, ax in enumerate(axes):
        # Only show legend on the first subplot
        # show_legend = (i == 1)
        create_map(ax=ax, vals=vals_list[i], bins=bins, percentiles=percentiles, show_legend=False)

        # Add subplot label/title
        ax.text(
            0.69, 0.79,  # x=0.5 centers, y=1.02 slightly above top
            labels[i],
            transform=ax.transAxes,  # relative to axes [0,1]
            ha='center', va='bottom',
            fontsize=14, fontweight='bold'
        )

    # ----- Create shared legend -----
    num_buckets = len(bins) - 1

    markers = ['o', 's', '^', 'D', 'P', 'H', 'X'][:num_buckets]

    grays = [
        "#FFFFFF",
        "#D9D9D9",
        "#B3B3B3",
        "#808080",
        "#4D4D4D",
        "#262626",
        "#000000",
    ][:num_buckets]

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0], [0],
            marker=markers[b],
            color='black',
            markerfacecolor=grays[b],
            markersize=10,
            linestyle='None',
            label=f'{percentiles[b]}–{percentiles[b + 1]}%'
        )
        for b in range(num_buckets)
    ]

    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=num_buckets,  # single row
        frameon=True,
        fontsize=14,
        bbox_to_anchor=(0.5, 0.06)
    )

    # Reduce space between subplots
    plt.subplots_adjust(
        left=0.05, right=0.95,
        top=0.80, bottom=0.02,  # increase bottom space
        hspace=0.05, wspace=0.02
    )
    plt.tight_layout(rect=[0.05, 0.08, 0.95, 0.98])
    plt.savefig("plots/kola_ssa_2.pdf", dpi=800)

    plt.show()

GEN_RANDOM = False
if __name__ == "__main__":
    #create_map(polygons, save=True)
    ret = analyse_kola_data(s=10, r=10)
    ladle_plots(range(len(ret["g_vec"])), ret["f_vec"], ret["phi_vec"], ret["g_vec"], q_hat=ret["rank"])
    if GEN_RANDOM:
        ss_idx = np.random.randint(low=0, high=len(ret["ss_vals"]))
        print(ss_idx)  # this gave 21, keep it fixed for the plots
    else:
        ss_idx = 21
    plot_kola_results(ret["ns_vals"], ret["ss_vals"], ss_idx)

