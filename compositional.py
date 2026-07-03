import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from matplotlib.pyplot import scatter
from pyssaBSS import SPSSA_COMB
from pyssaBSS.spatial import partition_points_by_polygons
from pyssaBSS.kernels import ScaledBallKernel, BallKernel
from pyssaBSS.scatter import SIRScatter, SAVEScatter, LCORScatter
from sklearn.decomposition import PCA, FastICA


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
    return data_ilr, V


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



def scatter_test(data, coords, segments):
    kernel=ScaledBallKernel(50000)
    mean_scat = SIRScatter().compute(data=data, coords=coords, segments=segments)
    var_scat = SAVEScatter().compute(data=data, coords=coords, segments=segments)
    cor_scat = LCORScatter(kernel).compute(data=data, coords=coords, segments=segments)

    res = np.asarray([np.linalg.norm(mean_scat),np.linalg.norm(var_scat),np.linalg.norm(cor_scat)])

    return res



def kola_pca_and_ica(values):
    # pca_vals = values.copy()
    # ica_vals = values.copy()
    pca = PCA(n_components=values.shape[0])
    ica = FastICA(n_components=values.shape[0])

    pca_vals = pca.fit(values.T).transform(values.T)
    ica_vals = ica.fit(values.T).transform(values.T)

    ret = {
        "pca_vals": pca_vals,
        "pca_loadings": pca.components_,
        "ica_vals": ica_vals,
        "ica_loadings": ica.components_,
        "ica_mixing": ica.mixing_
    }

    return ret


def local_vs_global(original, ssa, ica, pca, bss, coords, partition):
    results = {
        dat: np.zeros((35, 3)) for dat in ["Data", "spSSA", "ICA", "PCA", "SBSS"]
    }
    n = original.shape[1]
    for i in range(35):
        results["Data"][i] = scatter_test(original[i].reshape(1, n), coords, partition)
        results["spSSA"][i] = scatter_test(ssa[i].reshape(1, n), coords, partition)
        results["ICA"][i] = scatter_test(ica[i].reshape(1, n), coords, partition)
        results["PCA"][i] = scatter_test(pca[i].reshape(1, n), coords, partition)
        results["SBSS"][i] = scatter_test(bss[i].reshape(1, n), coords, partition)

    return results

import seaborn as sns

sns.set_theme(style="whitegrid")

stat_names = ["Mean", "Variance", "Spatial covariance"]

def plot_heatmaps(method_data):
    """
    Parameters
    ----------
    method_data : dict[str, np.ndarray]
        Dictionary mapping method -> (35,3) array.
    """
    plt.rcParams.update({
        "font.size": 14,  # Base font size
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    })
    stat_names = ["Mean", "Variance", "Spatial covariance"]
    methods = list(method_data.keys())

    # Common color scale
    all_values = np.concatenate([arr.ravel() for arr in method_data.values()])
    vmin = all_values.min()
    vmax = all_values.max()

    # Figure layout: 3 heatmaps + 1 narrow colorbar
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[1, 1, 1, 0.06],
        wspace=0.30
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    for i, stat in enumerate(stat_names):

        heat = np.column_stack([
            method_data[m][:, i]
            for m in methods
        ])

        sns.heatmap(
            heat,
            ax=axes[i],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            xticklabels=methods,
            yticklabels=np.arange(1, heat.shape[0] + 1),
            cbar=(i == 0),
            cbar_ax=cax if i == 0 else None,
            linewidths=0,
        )

        axes[i].set_title(stat, fontsize=16)

        axes[i].set_xlabel("")

        if i == 0:
            axes[i].set_ylabel("")
        else:
            axes[i].set_ylabel("")
            axes[i].set_yticklabels([])

        # Keep method names horizontal
        axes[i].tick_params(axis="x", rotation=0)
        axes[i].tick_params(axis="y", labelsize=10)

    cax.set_ylabel("Deviation", rotation=270, labelpad=20)
    plt.savefig("plots/heatmaps.pdf", dpi=800)
    plt.show()

def bss(white_data, coords):
    kernel=BallKernel(50000)
    lcov = kernel.global_covariance(data=white_data, coords=coords)
    vals, vecs = np.linalg.eigh(lcov)
    return vecs


def kola_plot_bss(bss, pca, ica):

    vals_list = [bss[-1], bss[-2], bss[-3], pca[0], pca[1], pca[2], ica[0], ica[1], ica[2]]
    labels = [r"$\mathrm{SBSS}_1$", r"$\mathrm{SBSS}_2$", r"$\mathrm{SBSS}_3$",
              r"$\mathrm{PCA}_1$", r"$\mathrm{PCA}_2$", r"$\mathrm{PCA}_3$",
              r"$\mathrm{ICA}_1$", r"$\mathrm{ICA}_2$", r"$\mathrm{ICA}_3$"]

    # Compute global min/max across all subplots
    all_vals = np.concatenate([v.ravel() for v in vals_list])
    global_min, global_max = all_vals.min(), all_vals.max()

    # Create subplots; reserve right margin for the shared colorbar
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
    axes = axes.ravel()

    sc_last = None
    for i, ax in enumerate(axes):
        _, sc = new_create_map(
            ax=ax,
            vals=vals_list[i],
            val_min=global_min,
            val_max=global_max,
            show_legend=False,   # no per-subplot colorbars
        )
        if sc is not None:
            sc_last = sc

        ax.text(
            0.69, 0.79,
            labels[i],
            transform=ax.transAxes,
            ha='center', va='bottom',
            fontsize=14, fontweight='bold'
        )

    # Single shared colorbar on the right
    plt.tight_layout(rect=[0.05, 0.02, 0.90, 0.98])   # leave room on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])   # [left, bottom, width, height]
    fig.colorbar(sc_last, cax=cbar_ax, orientation='vertical')
    plt.savefig("plots/kola/kola_bss.pdf")
    plt.show()


def loadings_to_csv(V, W, labels, file=None):
    # Compute CLR loadings
    # V: (p, p-1)
    # W: (k, p-1)
    loadings = V @ W.T  # shape: (p, k)

    # Optional: normalize each component (column) to unit norm
    loadings /= np.linalg.norm(loadings, axis=0, keepdims=True)

    # Create DataFrame
    df = pd.DataFrame(
        loadings,
        index=labels,
        columns=[f"Component_{i + 1}" for i in range(loadings.shape[1])]
    )

    # Save
    if file is not None:
        df.to_csv(file)

    print(df)


def analyse_kola_data(s=50, r=5):
    df = pd.read_csv("data/kola/moss_data.csv")
    noise_dim = r
    num_trials = s
    cols = ["Ag", "Al", "As", "B", "Ba", "Be", "Bi", "Ca", "Cd", "Co", "Cr", "Cu", "Fe", "Hg", "K", "La", "Mg", "Mn",
            "Mo", "Na", "Ni", "P", "Pb", "Rb", "S", "Sb", "Sc", "Se", "Si", "Sr", "Th", "Tl", "U", "V", "Y",
            "Zn"]  # Au, Pd, Pt have NaNs

    values = df[cols].values
    coords = df[["XCOO", "YCOO"]].values

    polygons = read_polygons("polygons1.json", old=True)
    partition = partition_points_by_polygons(coords, polygons)


    values = values.T

    values, ilr_V = ilr(clr(values))

    pca_ica_ret = kola_pca_and_ica(values)

    # fit model
    ssa_obj = SPSSA_COMB(data=values, coords=coords, partition=partition[0], kernel=ScaledBallKernel(50000), s=num_trials, r=noise_dim)

    # estimate rank
    ssa_obj.estimate_rank(individual=False)
    rank_summary = ssa_obj.rank_summary_

    # get subspaces
    ss, ns = ssa_obj.subspaces(rank_summary.rank)

    # compute projected values
    ss_vals = (ss @ (values - values.mean(axis=1, keepdims=True)))
    ns_vals = (ns @ (values - values.mean(axis=1, keepdims=True)))

    ssa_data = np.vstack((ns_vals, ss_vals))
    bss_mat = bss(ssa_obj._white_data_, coords)
    bss_vals = bss_mat.T @ ssa_obj._white_data_

    print("Nonstationary loadings: ")
    loadings_to_csv(ilr_V, ns, cols, file="data/kola/loadings/ns.csv")
    print("Stationary loadings: ")
    loadings_to_csv(ilr_V, ss, cols, file="data/kola/loadings/ns.csv")

    print("PCA loadings: ")
    loadings_to_csv(ilr_V, pca_ica_ret["pca_loadings"], cols, file="data/kola/loadings/pca.csv")
    print("ICA loadings: ")
    loadings_to_csv(ilr_V, pca_ica_ret["ica_loadings"], cols, file="data/kola/loadings/ica.csv")
    print("SBSS loadings: ")
    loadings_to_csv(ilr_V, bss_mat.T @ ssa_obj.whitener, cols, file="data/kola/loadings/sbss.csv")


    #stats = local_vs_global(values, ssa_data, ret["pca_vals"].T, ret["ica_vals"].T, bss_vals, coords, partition[0])

    #plot_heatmaps(stats)

    # kola_plot_bss(bss_vals, ret["pca_vals"].T, ret["ica_vals"].T)

    res_dict = {
        "f_vec": rank_summary.cum_f_vec,
        "phi_vec": rank_summary.phi,
        "g_vec": rank_summary.g_vec,
        "ss_vals": ss_vals,
        "ns_vals": ns_vals,
        "rank": rank_summary.rank,
        "pseudoeigs": ssa_obj.pseudoeigenvalues_
    }

    return res_dict


def new_create_map(ax=None, polygons=None, vals=None, val_min=None, val_max=None, show_legend=False):
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

    sc = None
    if vals is not None:
        x, y = coords[:, 0], coords[:, 1]

        norm = plt.Normalize(vmin=val_min, vmax=val_max)
        #cmap = plt.cm.gray_r
        cmap = plt.cm.RdBu_r

        sc = ax.scatter(
            x, y,
            marker='o',
            s=45,
            c=vals,
            cmap=cmap,
            norm=norm,
            edgecolors="black",
            linewidths=0.4,
            zorder=10
        )

    return ax, sc


def new_plot_kola_results(ns_vals, ss_vals, ss_idx):
    if len(ns_vals) < 5:
        vals_list = [ns_vals[i] for i in range(len(ns_vals))]
        idx = -1
        while len(vals_list) < 5:
            vals_list.append(ss_vals[idx])
            idx -= 1
    else:
        vals_list = [ns_vals[0], ns_vals[1], ns_vals[2], ns_vals[3], ns_vals[4], ss_vals[ss_idx]]
    labels = [r"$\mathbf{n}_1$", r"$\mathbf{n}_2$", r"$\mathbf{n}_3$", r"$\mathbf{n}_4$", r"$\mathbf{n}_5$",
              r"$\mathbf{{s}}_{" + f"{ss_idx}" + r"}$"]

    # Compute global min/max across all subplots
    all_vals = np.concatenate([v.ravel() for v in vals_list])
    global_min, global_max = all_vals.min(), all_vals.max()

    # Create subplots; reserve right margin for the shared colorbar
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    axes = axes.ravel()

    sc_last = None
    for i, ax in enumerate(axes):
        _, sc = new_create_map(
            ax=ax,
            vals=vals_list[i],
            val_min=global_min,
            val_max=global_max,
            show_legend=False,   # no per-subplot colorbars
        )
        if sc is not None:
            sc_last = sc

        ax.text(
            0.69, 0.79,
            labels[i],
            transform=ax.transAxes,
            ha='center', va='bottom',
            fontsize=14, fontweight='bold'
        )

    # Single shared colorbar on the right
    plt.tight_layout(rect=[0.05, 0.02, 0.90, 0.98])   # leave room on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])   # [left, bottom, width, height]
    fig.colorbar(sc_last, cax=cbar_ax, orientation='vertical')
    plt.savefig("plots/kola/kola_ssa.pdf")
    plt.show()



GEN_RANDOM = False
if __name__ == "__main__":
    ret = analyse_kola_data(s=50, r=5)
    ladle_plots(range(len(ret["g_vec"])), ret["f_vec"], ret["phi_vec"], ret["g_vec"], q_hat=ret["rank"])
    if GEN_RANDOM:
        ss_idx = np.random.randint(low=0, high=len(ret["ss_vals"]))
        print(ss_idx)  # this gave 19, keep it fixed for the plots
    else:
        ss_idx=19

    pe = ret["pseudoeigs"]
    for i in range(ret["rank"]):
        print(f"{i}th pseudoeigs: ")
        print("Mean: ", pe[0][i])
        print("Var: ", pe[1][i])
        print("Cor: ", pe[2][i])

    print(f"{ss_idx}th pseudoeigs: ")
    print("Mean: ", pe[0][ss_idx])
    print("Var: ", pe[1][ss_idx])
    print("Cor: ", pe[2][ss_idx])


    new_plot_kola_results(ret["ns_vals"], ret["ss_vals"], ss_idx)

