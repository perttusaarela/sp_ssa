import os
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from scipy import linalg

warnings.filterwarnings("ignore")
sns.set_palette("colorblind")

DATA_FILES = [
    "GRINS_AQCLIM_points_Italy_y20172018.csv",
    "GRINS_AQCLIM_points_Italy_y20192020.csv",
    "GRINS_AQCLIM_points_Italy_y20212022.csv",
]

STATION_FILE = "Station_registry_information.CSV"

VARIABLES = [
    "AQ_mean_NO2",
    "AQ_mean_O3",
    "AQ_mean_PM10",
    "AQ_mean_PM2.5",
]

VAR_LABELS = {
    "AQ_mean_NO2":   "NO\u2082 (\u00b5g/m\u00b3)",
    "AQ_mean_O3":    "O\u2083 (\u00b5g/m\u00b3)",
    "AQ_mean_PM10":  "PM10 (\u00b5g/m\u00b3)",
    "AQ_mean_PM2.5": "PM2.5 (\u00b5g/m\u00b3)",
}

OUTPUT_DIR = "real_data_plots_v2"

MIN_OBS = 200


# geo6x10 = 6 geographic groups x 10 temporal segments
SEG_MODE = "geo6x10"

# Radius-based spatial neighbourhood for LCOR
KERNEL_RADIUS_KM = 50

# Temporal lag for LCOR
LAG = 1

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
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
})


def safe_savefig(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for ext in [".pdf", ".png"]:
        try:
            plt.savefig(path + ext, bbox_inches="tight")
        except PermissionError:
            plt.savefig(path + "_new" + ext, bbox_inches="tight")
    print(f"  Saved: {path}.pdf/.png")


def is_island(lon, lat):
    sicily   = (lat < 38.5) and (11.5 < lon < 15.5)
    sardinia = (lon < 10.2) and (38.0 < lat < 41.5)
    return sicily or sardinia


def haversine_distance_matrix(lon, lat):
    """Pairwise great-circle distance matrix in kilometres."""
    lon = np.radians(np.asarray(lon))
    lat = np.radians(np.asarray(lat))
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


# Load data
def load_data():
    print("\n" + "=" * 60)
    print(" Step 1: Loading data")
    print("=" * 60)

    stations = pd.read_csv(STATION_FILE, low_memory=False)
    stations.columns = [c.strip() for c in stations.columns]
    stations["Longitude"] = pd.to_numeric(stations["Longitude"], errors="coerce")
    stations["Latitude"]  = pd.to_numeric(stations["Latitude"],  errors="coerce")
    stations = stations.dropna(subset=["Longitude", "Latitude"]).reset_index(drop=True)
    print(f"  {len(stations)} stations with coordinates")

    n_before = len(stations)
    stations = stations[
        ~stations.apply(lambda r: is_island(r["Longitude"], r["Latitude"]), axis=1)
    ].reset_index(drop=True)
    print(f"  Removed {n_before - len(stations)} island stations (Sicily + Sardinia)")
    print(f"  Mainland stations: {len(stations)}")

    dfs = []
    for fname in DATA_FILES:
        print(f"Loading: {fname}")
        df = pd.read_csv(fname, low_memory=False,
                         usecols=["AirQualityStation", "time"] + VARIABLES)
        for v in VARIABLES:
            df[v] = pd.to_numeric(df[v], errors="coerce")
            df.loc[df[v] < 0, v] = np.nan
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["time"] = pd.to_datetime(df_all["time"], errors="coerce")
    df_all = df_all.dropna(subset=["time"]).sort_values("time")

    print(f"\nCombined rows: {len(df_all):,}")
    print(f"Date range: {df_all['time'].min().date()} to {df_all['time'].max().date()}")

    counts = df_all.groupby("AirQualityStation")[VARIABLES].count()
    good   = counts[(counts >= MIN_OBS).all(axis=1)].index
    good   = [s for s in good if s in stations["AirQualityStation"].values]

    df_all   = df_all[df_all["AirQualityStation"].isin(good)]
    stations = stations[stations["AirQualityStation"].isin(good)].reset_index(drop=True)
    print(f"\nMainland stations with all 4 variables (>={MIN_OBS} obs): {len(stations)}")

    wide = {}
    for v in VARIABLES:
        w = df_all.pivot_table(index="time", columns="AirQualityStation",
                               values=v, aggfunc="mean").sort_index()
        w = w[[s for s in good if s in w.columns]]
        wide[v] = w

    dates = wide[VARIABLES[0]].index
    for v in VARIABLES[1:]:
        dates = dates.intersection(wide[v].index)

    sids = list(wide[VARIABLES[0]].columns)
    for v in VARIABLES[1:]:
        sids = [s for s in sids if s in wide[v].columns]

    for v in VARIABLES:
        wide[v] = wide[v].loc[dates, sids]

    stations = stations.set_index("AirQualityStation").loc[sids].reset_index()

    print(f"Final stations: {len(stations)}")
    print(f"Final dates: {len(dates)}")
    print("Missing per variable:")
    for v in VARIABLES:
        pct = wide[v].isna().sum().sum() / wide[v].size * 100
        print(f"  {v:25s}: {pct:.1f}%")

    return stations, wide, dates, sids



def _load_italy_mainland():
    """Load Italy mainland border from naturalearth, removing islands."""
    try:
        import requests, geopandas as gpd, io
        url = ("https://raw.githubusercontent.com/nvkelso/"
               "natural-earth-vector/master/geojson/"
               "ne_50m_admin_0_countries.geojson")
        r = requests.get(url, timeout=20)
        world = gpd.read_file(io.BytesIO(r.content))
        italy_full = world[world["NAME"] == "Italy"]
        italy_exp  = italy_full.explode(index_parts=False).reset_index(drop=True)
        italy_mainland = italy_exp[italy_exp.geometry.area > 3.0]
        print("  Italy mainland border loaded.")
        return italy_mainland
    except Exception as e:
        print(f"  Italy border unavailable: {e}")
        return None



# Station map
def plot_station_map(stations, output_dir):
    print("\n--- Step 2: Station map ---")
    fig, ax = plt.subplots(figsize=(6, 9))

    italy_geom = _load_italy_mainland()
    if italy_geom is not None and not italy_geom.empty:
        italy_geom.boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=1)

    ax.scatter(stations["Longitude"], stations["Latitude"],
               color="#3a7ebf", s=18, alpha=0.85,
               edgecolors="white", linewidths=0.3, zorder=3)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(6, 19)
    ax.set_ylim(36, 48)
    ax.yaxis.grid(True, linestyle=":", alpha=0.3)
    ax.xaxis.grid(True, linestyle=":", alpha=0.3)

    plt.tight_layout()
    safe_savefig(os.path.join(output_dir, "01_station_map"))
    plt.close()


# Time series with temporal segment boundaries
def plot_time_series(wide, dates, stations, output_dir):
    print("\n--- Step 3: Time series ---")
    north = stations[stations["Latitude"] > 44].sort_values("Latitude")
    if len(north) == 0:
        north = stations.sort_values("Latitude", ascending=False)
    row = north.iloc[len(north) // 2]
    sid = row["AirQualityStation"]

    # Compute 10 equal temporal segment boundaries
    all_dates   = wide[VARIABLES[0]].index
    n_dates     = len(all_dates)
    seg_indices = np.array_split(np.arange(n_dates), 10)
    seg_boundaries = [all_dates[seg[0]] for seg in seg_indices[1:]]

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True,
                             gridspec_kw={"hspace": 0.35})
    colours = sns.color_palette("colorblind", 4)

    for i, (v, ax) in enumerate(zip(VARIABLES, axes)):
        if sid not in wide[v].columns:
            continue
        ts   = wide[v][sid].dropna()
        ts   = ts[ts.index <= "2022-01-15"]
        roll = ts.rolling(30, min_periods=10).mean()
        ax.plot(ts.index, ts.values, color=colours[i], lw=0.5, alpha=0.30)
        ax.plot(roll.index, roll.values, color=colours[i], lw=2.0, alpha=0.95,
                label="30-day mean")

        # Temporal segment boundaries
        for j, bd in enumerate(seg_boundaries):
            ax.axvline(bd, color="grey", linestyle="--", linewidth=0.8,
                       alpha=0.6,
                       label="Temporal segment boundary" if (i == 0 and j == 0) else "")

        ax.set_ylabel(VAR_LABELS[v], fontsize=9)
        ax.yaxis.grid(True, linestyle=":", alpha=0.4)

    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_xlim(pd.Timestamp("2017-01-01"), pd.Timestamp("2022-01-15"))
    plt.tight_layout()
    safe_savefig(os.path.join(output_dir, "02_time_series"))
    plt.close()


# Spatial maps with Italy border
def _get_italy_mainland_coords():
    """Return Italy mainland border as (lons, lats) numpy arrays."""
    try:
        import requests, geopandas as gpd, io
        from shapely.geometry import MultiPolygon, Polygon

        url = ("https://raw.githubusercontent.com/nvkelso/"
               "natural-earth-vector/master/geojson/"
               "ne_50m_admin_0_countries.geojson")
        r     = requests.get(url, timeout=20)
        world = gpd.read_file(io.BytesIO(r.content))
        italy = world[world["NAME"] == "Italy"]
        if italy.empty:
            return None

        italy_exp = italy.explode(index_parts=False).reset_index(drop=True)
        mainland  = italy_exp.loc[[italy_exp.geometry.area.idxmax()]]
        geom      = mainland.geometry.iloc[0]

        if isinstance(geom, Polygon):
            lons, lats = geom.exterior.xy
            return np.array(lons), np.array(lats)
        elif isinstance(geom, MultiPolygon):
            largest    = max(geom.geoms, key=lambda g: g.area)
            lons, lats = largest.exterior.xy
            return np.array(lons), np.array(lats)
        return None
    except Exception as e:
        print(f"  Note: Italy border not available ({e})")
        return None


def plot_spatial_maps(wide, dates, stations, output_dir):
    print("\n--- Step 4: Spatial maps ---")
    target_dates = {"Winter": pd.Timestamp("2020-01-15"),
                    "Summer": pd.Timestamp("2020-07-15")}
    resolved = {}
    for season, target in target_dates.items():
        found = None
        for v in VARIABLES:
            if target in wide[v].index:
                found = target; break
        if found is None:
            idx   = np.argmin(np.abs(wide[VARIABLES[0]].index - target))
            found = wide[VARIABLES[0]].index[idx]
        resolved[season] = found
        print(f"  {season} date: {found.date()}")

    season_names = list(resolved.keys())
    season_dates = list(resolved.values())

    italy_coords = _get_italy_mainland_coords()

    fig, axes = plt.subplots(2, 4, figsize=(16, 9),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.30})
    for col_idx, v in enumerate(VARIABLES):
        all_vals = []
        for d in season_dates:
            if d in wide[v].index:
                all_vals.extend(wide[v].loc[d].dropna().tolist())
        if not all_vals:
            continue
        vmin = np.percentile(all_vals, 2)
        vmax = np.percentile(all_vals, 98)

        for row_idx, (season, d) in enumerate(zip(season_names, season_dates)):
            ax = axes[row_idx, col_idx]

            if italy_coords is not None:
                ax.plot(italy_coords[0], italy_coords[1],
                        color="black", linewidth=0.8, zorder=1)

            vals  = wide[v].loc[d]
            valid = stations[stations["AirQualityStation"].isin(vals.dropna().index)]
            pv    = vals[valid["AirQualityStation"]].values
            sc    = ax.scatter(valid["Longitude"], valid["Latitude"], c=pv,
                               cmap="YlOrRd", vmin=vmin, vmax=vmax,
                               s=22, alpha=0.85, edgecolors="none", zorder=3)
            ax.set_xlim(6, 19); ax.set_ylim(36, 48)
            ax.yaxis.grid(True, linestyle=":", alpha=0.3)
            ax.xaxis.grid(True, linestyle=":", alpha=0.3)

            if row_idx == 0:
                ax.set_title(VAR_LABELS[v], fontsize=11, fontweight="bold", pad=6)
            if col_idx == 0:
                ax.set_ylabel(f"{season}\n{d.strftime('%d %b %Y')}\n\nLatitude",
                              fontsize=9)
            else:
                ax.set_ylabel("")
            if row_idx == 1:
                ax.set_xlabel("Longitude", fontsize=8)
            else:
                ax.set_xlabel("")
            ax.text(0.02, 0.02, f"n = {len(valid)}", transform=ax.transAxes,
                    fontsize=7.5, color="#555555", va="bottom")

        cbar = fig.colorbar(sc, ax=axes[:, col_idx], label=VAR_LABELS[v],
                            fraction=0.04, pad=0.02, aspect=30)
        cbar.ax.tick_params(labelsize=7.5)

    plt.tight_layout()
    safe_savefig(os.path.join(output_dir, "03_spatial_maps"))
    plt.close()


def build_data_tensor(wide, dates, sids):
    n_stations = len(sids); n_times = len(dates); p = len(VARIABLES)
    X_tensor = np.full((n_stations, n_times, p), np.nan)
    for j, v in enumerate(VARIABLES):
        for i, sid in enumerate(sids):
            if sid in wide[v].columns:
                X_tensor[i, :, j] = wide[v][sid].values
    print(f"\n  Data tensor: {n_stations} stations x {n_times} times x {p} variables")
    print(f"  NaN values: {int(np.isnan(X_tensor).sum())} (filled after seasonal adjustment)")
    return X_tensor, n_stations, n_times, p


def whitening(X):
    mean = X.mean(axis=0); Xc = X - mean
    cov  = np.cov(Xc, rowvar=False)
    vals, vecs = linalg.eigh(cov)
    idx  = np.argsort(vals)[::-1]
    vals = np.maximum(vals[idx], 1e-10); vecs = vecs[:, idx]
    W    = vecs @ np.diag(1.0 / np.sqrt(vals))
    return Xc @ W, W, mean, cov


def _make_segments_from_groups(spatial_groups, n_times, nt):
    time_groups      = np.array_split(np.arange(n_times), nt)
    time_boundaries  = [int(tg[-1]) for tg in time_groups[:-1]]
    segments = []
    for sg in spatial_groups:
        sg = np.asarray(sg)
        for tg in time_groups:
            obs_idx = [int(s) * n_times + int(t) for s in sg for t in tg]
            if obs_idx:
                segments.append(np.array(obs_idx))
    return segments, time_boundaries


def get_segments(n_stations, n_times, stations_df, seg_mode):
    lat_vals = stations_df["Latitude"].values
    lon_vals = stations_df["Longitude"].values
    all_idx  = np.arange(len(lat_vals))

    LAT1 = 41.51; LAT2 = 43.90; LAT3 = 45.31

    def geo6_split(nt):
        lbs       = [LAT1, LAT2, LAT3]
        LON_SPLIT = 11.0
        g1 = all_idx[lat_vals < LAT1]
        g2 = all_idx[(lat_vals >= LAT1) & (lat_vals < LAT2)]
        g3 = all_idx[(lat_vals >= LAT2) & (lat_vals < LAT3) & (lon_vals < LON_SPLIT)]
        g4 = all_idx[(lat_vals >= LAT2) & (lat_vals < LAT3) & (lon_vals >= LON_SPLIT)]
        g5 = all_idx[(lat_vals >= LAT3) & (lon_vals < LON_SPLIT)]
        g6 = all_idx[(lat_vals >= LAT3) & (lon_vals >= LON_SPLIT)]
        sgroups = [g1, g2, g3, g4, g5, g6]
        segs, tbs = _make_segments_from_groups(sgroups, n_times, nt)
        return segs, sgroups, lbs, tbs

    def rectangular_split(nx, ny, nt):
        lat_order  = np.argsort(lat_vals)
        lat_bands  = np.array_split(lat_order, ny)
        lbs = [0.5 * (lat_vals[lat_bands[i]].max() + lat_vals[lat_bands[i+1]].min())
               for i in range(len(lat_bands) - 1)]
        sgroups = []
        for lb in lat_bands:
            lon_order = lb[np.argsort(lon_vals[lb])]
            for sg in np.array_split(lon_order, nx):
                sgroups.append(sg)
        segs, tbs = _make_segments_from_groups(sgroups, n_times, nt)
        return segs, sgroups, lbs, tbs

    # Dispatch
    if seg_mode == "geo6x10":
        segments, spatial_groups, lat_boundaries, time_boundaries = geo6_split(10)
        label = "6 geographic groups, 10 temporal"
    elif seg_mode == "4x4x4":
        segments, spatial_groups, lat_boundaries, time_boundaries = rectangular_split(4, 4, 4)
        label = "4x4 rectangular grid, 4 temporal"
    else:
        segments, spatial_groups, lat_boundaries, time_boundaries = geo6_split(10)
        label = f"geo6x10 (fallback for unknown mode: {seg_mode})"

    seg_sizes = [len(s) for s in segments]
    print(f"  Segmentation: {label}")
    print(f"  Spatial groups: {len(spatial_groups)}, "
          f"Temporal: {len(time_boundaries)+1}, "
          f"Total segments: {len(segments)}")
    print(f"  Segment sizes: min={min(seg_sizes)}, "
          f"max={max(seg_sizes)}, mean={np.mean(seg_sizes):.0f}")
    for i, sg in enumerate(spatial_groups):
        print(f"    Group {i+1}: {len(np.asarray(sg))} stations")

    return segments, spatial_groups, lat_boundaries, time_boundaries


# Segmentation plot 
def plot_segmentation(stations_df, spatial_groups, lat_boundaries,
                      seg_mode, output_dir):
    print("\n--- Segmentation plot ---")
    from scipy.spatial import ConvexHull
    from matplotlib.patches import Polygon as MplPolygon

    lat_vals = stations_df["Latitude"].values
    lon_vals = stations_df["Longitude"].values

    CB_COLOURS = ["#1f77b4","#d62728","#2ca02c","#ff7f0e",
                  "#9467bd","#8c564b","#e377c2","#17becf"]
    group_colours = [CB_COLOURS[i % len(CB_COLOURS)] for i in range(len(spatial_groups))]

    italy_geom = _load_italy_mainland()
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    if italy_geom is not None and not italy_geom.empty:
        italy_geom.boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=1)

    for g_idx, sg in enumerate(spatial_groups):
        sg     = np.asarray(sg)
        colour = group_colours[g_idx]
        lons_g = lon_vals[sg]; lats_g = lat_vals[sg]
        pts    = np.column_stack([lons_g, lats_g])

        if len(pts) >= 3:
            try:
                hull      = ConvexHull(pts)
                hull_pts  = pts[hull.vertices]
                hull_patch = MplPolygon(hull_pts, closed=True,
                                        facecolor=colour, alpha=0.25,
                                        edgecolor=colour, linewidth=1.5, zorder=2)
                ax.add_patch(hull_patch)
            except Exception:
                pass

        ax.scatter(lons_g, lats_g, facecolors=colour, edgecolors="white",
                   s=22, alpha=1.0, linewidths=0.6, zorder=3)
        cx, cy = lons_g.mean(), lats_g.mean()
        ax.text(cx, cy, f"S{g_idx+1}", fontsize=6.5,
                ha="center", va="center", color="black",
                fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec=colour, linewidth=0.8, alpha=0.85))

    for lb in lat_boundaries:
        ax.axhline(lb, color="black", linewidth=1.2,
                   linestyle="--", alpha=0.7, zorder=4)

    ax.set_xlim(6, 19); ax.set_ylim(36, 48)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.yaxis.grid(True, linestyle=":", alpha=0.2)
    ax.xaxis.grid(True, linestyle=":", alpha=0.2)
    plt.tight_layout()
    safe_savefig(os.path.join(output_dir, "06_segmentation"))
    plt.close()
    print("  Segmentation plot saved.")


# Radius visualisation
def plot_radius_visualisation(stations_df, spatial_groups, lat_boundaries,
                              seg_mode, output_dir,
                              radius_km=50, rep_lon=10.0, rep_lat=45.0):
    print("\n--- Radius visualisation figure ---")
    from matplotlib.patches import Ellipse

    lat_vals = stations_df["Latitude"].values
    lon_vals = stations_df["Longitude"].values

    italy_geom = _load_italy_mainland()

    def haversine_single(lon1, lat1, lons, lats):
        R    = 6371.0
        phi1 = np.radians(lat1); phi2 = np.radians(lats)
        dphi = np.radians(lats - lat1); dlam = np.radians(lons - lon1)
        a    = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    # Neighbour counts for every station
    all_neighbour_counts = np.zeros(len(lat_vals), dtype=int)
    for s in range(len(lat_vals)):
        dists = haversine_single(lon_vals[s], lat_vals[s], lon_vals, lat_vals)
        all_neighbour_counts[s] = np.sum(dists <= radius_km) - 1

    mean_count = all_neighbour_counts.mean()
    print(f"  Neighbour counts: min={all_neighbour_counts.min()}, "
          f"mean={mean_count:.1f}, max={all_neighbour_counts.max()}")

  
    s1_west_mask = ((lat_vals < 41.51) & (lat_vals >= 40.0) & (lon_vals < 14.5))
    s1_west_idx  = np.where(s1_west_mask)[0]

    best_idx   = None
    best_score = -1

    for i in s1_west_idx:
        dists_i    = haversine_single(lon_vals[i], lat_vals[i], lon_vals, lat_vals)
        nb_i       = np.where(dists_i <= radius_km)[0]
        nb_i       = nb_i[nb_i != i]
        n_count    = len(nb_i)
        all_in_s1  = np.all(lat_vals[nb_i] < 41.51) if n_count > 0 else False

        if n_count >= 5 and all_in_s1:
            score = -abs(n_count - mean_count)  # closer to mean = better
            if score > best_score:
                best_score = score
                best_idx   = i

    if best_idx is not None:
        rep_idx = best_idx
    else:
        print("  Warning: no station with 5+ neighbours all in S1 found, using best available")
        best_count = -1
        for i in s1_west_idx:
            dists_i = haversine_single(lon_vals[i], lat_vals[i], lon_vals, lat_vals)
            nb_i    = np.where(dists_i <= radius_km)[0]
            nb_i    = nb_i[nb_i != i]
            if len(nb_i) > best_count:
                best_count = len(nb_i)
                best_idx   = i
        rep_idx = best_idx if best_idx is not None else 0

    rep_lon_actual = lon_vals[rep_idx]
    rep_lat_actual = lat_vals[rep_idx]
    dist_from_rep  = haversine_single(rep_lon_actual, rep_lat_actual, lon_vals, lat_vals)
    neighbours     = np.where(dist_from_rep <= radius_km)[0]
    neighbours     = neighbours[neighbours != rep_idx]

    print(f"  Representative station: index={rep_idx}, "
          f"lon={rep_lon_actual:.2f}, lat={rep_lat_actual:.2f}")
    print(f"  Neighbours within {radius_km} km: {len(neighbours)}")

    # Find spatial group of representative station
    rep_group_idx = None
    for g_idx, sg in enumerate(spatial_groups):
        if rep_idx in np.asarray(sg):
            rep_group_idx = g_idx; break
    group_members = (np.asarray(spatial_groups[rep_group_idx])
                     if rep_group_idx is not None else np.array([]))
    group_members = group_members[group_members != rep_idx]

    radius_deg_lat = radius_km / 111.0
    radius_deg_lon = radius_km / (111.0 * np.cos(np.radians(rep_lat_actual)))

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    if italy_geom is not None and not italy_geom.empty:
        italy_geom.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=1)

    ax.scatter(lon_vals, lat_vals, color="lightgrey", s=15, alpha=0.8,
               edgecolors="white", linewidths=0.3, zorder=2, label="All stations")

    non_neighbour_group = group_members[~np.isin(group_members, neighbours)]
    if len(non_neighbour_group) > 0:
        ax.scatter(lon_vals[non_neighbour_group], lat_vals[non_neighbour_group],
                   color="#1f77b4", s=35, alpha=0.9, edgecolors="white",
                   linewidths=0.5, zorder=3,
                   label=f"Same segment, outside {radius_km} km (n={len(non_neighbour_group)})")

    ax.scatter(lon_vals[neighbours], lat_vals[neighbours],
               color="#d62728", s=35, alpha=0.9, edgecolors="white",
               linewidths=0.5, zorder=4,
               label=f"Within {radius_km} km radius (n={len(neighbours)})")

    ax.scatter(rep_lon_actual, rep_lat_actual,
               color="black", s=120, marker="*",
               edgecolors="white", linewidths=0.5, zorder=5,
               label="Representative station")

    circle = Ellipse((rep_lon_actual, rep_lat_actual),
                     width=2 * radius_deg_lon, height=2 * radius_deg_lat,
                     fill=False, edgecolor="#d62728",
                     linewidth=1.8, linestyle="--", zorder=6)
    ax.add_patch(circle)

    for lb in lat_boundaries:
        ax.axhline(lb, color="black", linewidth=1.2,
                   linestyle="--", alpha=0.6, zorder=4)

    ax.annotate(f"r = {radius_km} km",
                xy=(rep_lon_actual + radius_deg_lon, rep_lat_actual),
                xytext=(rep_lon_actual + radius_deg_lon + 0.5, rep_lat_actual + 0.3),
                fontsize=15, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))

    ax.set_xlim(6, 19); ax.set_ylim(36, 48)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=11, frameon=True, framealpha=0.9, loc="lower left")
    ax.yaxis.grid(True, linestyle=":", alpha=0.2)
    ax.xaxis.grid(True, linestyle=":", alpha=0.2)
    plt.tight_layout()
    safe_savefig(os.path.join(output_dir, "07_radius_visualisation"))
    plt.close()
    print("  Radius visualisation saved.")


# Scatter matrices
def compute_M_mean(Xw, segments):
    p = Xw.shape[1]; N = len(Xw); M = np.zeros((p, p))
    for idx in segments:
        if len(idx) == 0: continue
        mu = Xw[idx].mean(axis=0, keepdims=True)
        w  = len(idx) / N
        M += w * (mu.T @ mu)
    return M


def compute_M_var(Xw, segments):
    p = Xw.shape[1]; N = len(Xw); M = np.zeros((p, p)); Ip = np.eye(p)
    for idx in segments:
        if len(idx) < p + 1: continue
        C = np.cov(Xw[idx], rowvar=False)
        w = len(idx) / N
        D = Ip - C
        M += w * (D @ D.T)
    return M


def compute_M_lcor(Xw, n_stations, n_times, nt, coords, radius_km, lag=1):
    p   = Xw.shape[1]; M = np.zeros((p, p))
    X3d = Xw.reshape(n_stations, n_times, p)
    lon = coords[:, 0]; lat = coords[:, 1]
    dist = haversine_distance_matrix(lon, lat)

    neighbours      = [np.where(dist[i] <= radius_km)[0] for i in range(n_stations)]
    neighbour_counts = np.array([len(nb) for nb in neighbours])
    print(f"    M_lcor: radius = {radius_km} km | lag = {lag} days")
    print(f"    Neighbours per station: "
          f"min={neighbour_counts.min()}, "
          f"mean={neighbour_counts.mean():.1f}, "
          f"median={np.median(neighbour_counts):.1f}, "
          f"max={neighbour_counts.max()}")

    def lagged_cov_from_block(station_idx, time_idx):
        if len(time_idx) <= lag or len(station_idx) == 0:
            return np.zeros((p, p))
        C = np.zeros((p, p)); used = 0
        for s in station_idx:
            x = X3d[s, time_idx, :]
            if x.shape[0] <= lag: continue
            C += (x[:-lag].T @ x[lag:]) / (x.shape[0] - lag)
            used += 1
        if used == 0: return np.zeros((p, p))
        return C / used

    all_times    = np.arange(n_times)
    all_stations = np.arange(n_stations)
    C_global     = lagged_cov_from_block(all_stations, all_times)
    time_segments = np.array_split(np.arange(n_times), nt)
    count = 0

    for nb in neighbours:
        if len(nb) == 0: continue
        for ts in time_segments:
            if len(ts) <= lag: continue
            C_local = lagged_cov_from_block(nb, ts)
            D = C_global - C_local
            M += D @ D.T
            count += 1

    if count > 0: M /= count
    return M


def approx_joint_diag(matrices, n_iter=1000):
    p  = matrices[0].shape[0]; V = np.eye(p); Ms = [M.copy() for M in matrices]
    for _ in range(n_iter):
        for i in range(p - 1):
            for j in range(i + 1, p):
                num = sum(2 * M[i, j] * (M[i, i] - M[j, j]) for M in Ms)
                den = sum(4 * M[i, j] ** 2 - (M[i, i] - M[j, j]) ** 2 for M in Ms)
                if abs(den) < 1e-14: continue
                theta = 0.5 * np.arctan2(num, den)
                c, s  = np.cos(theta), np.sin(theta)
                G = np.eye(p); G[i,i]=c; G[j,j]=c; G[i,j]=-s; G[j,i]=s
                Ms = [G.T @ M @ G for M in Ms]; V = V @ G
    return V, Ms


def run_stssa_comb(Xw, n_stations, n_times, stations_df, seg_mode):
    print(f"\n  Segmentation mode: {seg_mode}")
    segments, spatial_groups, lat_boundaries, time_boundaries = get_segments(
        n_stations, n_times, stations_df, seg_mode)

    print("  Computing M_mean...")
    Mm = compute_M_mean(Xw, segments)
    print("  Computing M_var...")
    Mv = compute_M_var(Xw, segments)
    print("  Computing M_lcor...")
    coords   = stations_df[["Longitude", "Latitude"]].values
    nt_lcor  = len(time_boundaries) + 1
    Ml = compute_M_lcor(Xw, n_stations, n_times, nt=nt_lcor,
                        coords=coords, radius_km=KERNEL_RADIUS_KM, lag=LAG)

    print("\n  Scatter matrix norms:")
    print(f"    ||M_mean|| = {np.linalg.norm(Mm, 'fro'):.6f}")
    print(f"    ||M_var||  = {np.linalg.norm(Mv, 'fro'):.6f}")
    print(f"    ||M_lcor|| = {np.linalg.norm(Ml, 'fro'):.6f}")

    def normalise(M):
        n = np.linalg.norm(M, "fro")
        return M / n if n > 1e-14 else M

    print("\n  Running joint diagonalisation...")
    V, diag_Ms = approx_joint_diag([normalise(Mm), normalise(Mv), normalise(Ml)])

    scores = sum(np.abs(np.diag(Md)) for Md in diag_Ms)
    d_mean = np.abs(np.diag(diag_Ms[0]))
    d_var  = np.abs(np.diag(diag_Ms[1]))
    d_lcor = np.abs(np.diag(diag_Ms[2]))

    order  = np.argsort(scores)[::-1]
    V      = V[:, order]; scores = scores[order]
    d_mean = d_mean[order]; d_var = d_var[order]; d_lcor = d_lcor[order]

    print("\n  Individual pseudo-eigenvalues:")
    print(f"  {'Comp':>5} {'Combined':>10} {'d_mean':>10} {'d_var':>10} {'d_lcor':>10}")
    for i in range(len(scores)):
        print(f"  {i+1:>5} {scores[i]:>10.4f} {d_mean[i]:>10.4f} "
              f"{d_var[i]:>10.4f} {d_lcor[i]:>10.4f}")

    return (V, scores, d_mean, d_var, d_lcor, Mm, Mv, Ml,
            spatial_groups, lat_boundaries, time_boundaries)


# Scree plot
def plot_scree(scores, d_mean, d_var, d_lcor, output_dir):
    print("\n--- Step 6: Scree plot ---")
    p   = len(scores); idx = np.arange(1, p + 1)
    q_hat = max(1, int(np.sum(scores / scores[0] > 0.30)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"wspace": 0.40})
    colours   = sns.color_palette("colorblind", p)

    bars = axes[0].bar(idx, scores, color=colours, alpha=0.85,
                       edgecolor="white", linewidth=0.5)
    for bar, s in zip(bars, scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{s:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_xlabel("Component index")
    axes[0].set_ylabel("Combined pseudo-eigenvalue")
    axes[0].set_title("stSSA-COMB combined pseudo-eigenvalues")
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels([f"Comp {i}" for i in idx])
    axes[0].yaxis.grid(True, linestyle=":", alpha=0.5)

    width = 0.25; x = np.array(idx, dtype=float)
    for bars_data, label, colour in [
        (d_mean, r"$d_{\mathrm{mean}}$ (SIR)", "#185fa5"),
        (d_var,  r"$d_{\mathrm{var}}$ (SAVE)",  "#e07b39"),
        (d_lcor, r"$d_{\mathrm{lcor}}$ (LCOR)", "#2ca02c"),
    ]:
        offset = [-width, 0, width][["#185fa5", "#e07b39", "#2ca02c"].index(colour)]
        b = axes[1].bar(x + offset, bars_data, width, label=label,
                        color=colour, alpha=0.85, edgecolor="white")
        for bar in b:
            h = bar.get_height()
            if h > 0.02:
                axes[1].text(bar.get_x() + bar.get_width()/2, h + 0.005,
                             f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    axes[1].set_xlabel("Component index")
    axes[1].set_ylabel("Individual pseudo-eigenvalue")
    axes[1].set_title("Individual pseudo-eigenvalues\n"
                      "Shows what kind of nonstationarity each component has")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels([f"Comp {i}" for i in idx])
    axes[1].yaxis.grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(fontsize=8.5, frameon=True, framealpha=0.85)

    plt.tight_layout()
    safe_savefig(os.path.join(output_dir, "04_scree_plot"))
    plt.close()
    print(f"\n  q\u0302 = {q_hat} nonstationary, {p-q_hat} stationary")
    return q_hat


# Components
def plot_components(Xw, V, scores, stations_df, n_stations, n_times, dates, q_hat, output_dir):
    print("\n--- Step 7: Components ---")
    p        = V.shape[0]; comps = Xw @ V; comps_3d = comps.reshape(n_stations, n_times, p)
    pal      = sns.color_palette("colorblind", p)

    fig, axes = plt.subplots(p, 2, figsize=(13, 4.0 * p),
                             gridspec_kw={"wspace": 0.35, "hspace": 0.55})
    for i in range(p):
        ctype         = "Nonstationary" if i < q_hat else "Stationary"
        colour        = pal[i]
        spatial_score = np.abs(comps_3d[:, :, i]).mean(axis=1)
        weights       = V[:, i]
        ts            = pd.Series(comps_3d[:, :, i].mean(axis=0), index=dates)
        roll          = ts.rolling(30, min_periods=10).mean()

        ax_sp = axes[i, 0]
        sc = ax_sp.scatter(stations_df["Longitude"], stations_df["Latitude"],
                           c=spatial_score, cmap="YlOrRd", s=22, alpha=0.85, edgecolors="none")
        plt.colorbar(sc, ax=ax_sp, label="Mean |component score|", fraction=0.03, pad=0.02)
        ax_sp.set_xlim(6, 19); ax_sp.set_ylim(36, 48)
        weight_str = "  ".join(f"{v.replace('AQ_mean_','')}: {w:+.2f}"
                               for v, w in zip(VARIABLES, weights))
        ax_sp.set_xlabel(f"Longitude\nWeights: {weight_str}", fontsize=7.5)
        ax_sp.set_ylabel("Latitude", fontsize=8)
        ax_sp.set_title(f"Component {i+1} [{ctype}]\n"
                        f"Spatial activity (score = {scores[i]:.4f})", fontsize=9)
        ax_sp.yaxis.grid(True, linestyle=":", alpha=0.3)
        ax_sp.xaxis.grid(True, linestyle=":", alpha=0.3)

        ax_ts = axes[i, 1]
        ax_ts.plot(ts.index, ts.values, color=colour, lw=0.6, alpha=0.30)
        ax_ts.plot(roll.index, roll.values, color=colour, lw=2.0, alpha=0.95,
                   label="30-day mean")
        ax_ts.set_xlabel("Date", fontsize=8)
        ax_ts.set_ylabel("Component score", fontsize=8)
        ax_ts.set_title(f"Component {i+1} [{ctype}] \u2014 Time series\n"
                        f"(mean across all {n_stations} stations)", fontsize=9)
        ax_ts.yaxis.grid(True, linestyle=":", alpha=0.4)
        if i == 0: ax_ts.legend(fontsize=8, frameon=False)

    plt.tight_layout()
    safe_savefig(os.path.join(output_dir, "05_components"))
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n" + "=" * 60)
    print(" stSSA-COMB Real Data Analysis \u2014 Mainland Italy")
    print(f" Variables : {[v.replace('AQ_mean_', '') for v in VARIABLES]}")
    print(f" Seg mode  : {SEG_MODE}")
    print(f" Radius    : {KERNEL_RADIUS_KM} km")
    print(f" Lag       : {LAG} days")
    print(f" Output    : {OUTPUT_DIR}/")
    print("=" * 60)

    stations, wide, dates, sids = load_data()
    plot_station_map(stations, OUTPUT_DIR)
    plot_time_series(wide, dates, stations, OUTPUT_DIR)
    plot_spatial_maps(wide, dates, stations, OUTPUT_DIR)

    print("\n--- Step 5: Building data tensor ---")
    X_tensor, n_stations, n_times, p = build_data_tensor(wide, dates, sids)

    print("\n  Applying seasonal adjustment (day-of-year mean removal)...")
    doy_idx = pd.DatetimeIndex(dates).dayofyear
    X_adj   = X_tensor.copy()
    for doy_val in range(1, 367):
        t_mask = doy_idx == doy_val
        if t_mask.sum() == 0: continue
        seasonal_mean = np.nanmean(X_tensor[:, t_mask, :], axis=1, keepdims=True)
        X_adj[:, t_mask, :] -= seasonal_mean
    print("  Seasonal day-of-year means removed")
    n_nan = np.isnan(X_adj).sum()
    X_adj = np.nan_to_num(X_adj, nan=0.0)
    print(f"  Filled {n_nan} remaining NaN values with 0")

    X_flat_adj = X_adj.reshape(n_stations * n_times, p)
    Xw, W, mu, cov = whitening(X_flat_adj)
    print(f"\n  Whitened shape: {Xw.shape} | "
          f"n_stations={n_stations} | n_times={n_times} | p={p}")

    (V, scores, d_mean, d_var, d_lcor, Mm, Mv, Ml,
     spatial_groups, lat_boundaries, time_boundaries) = run_stssa_comb(
        Xw, n_stations, n_times, stations, SEG_MODE)

    plot_segmentation(stations, spatial_groups, lat_boundaries, SEG_MODE, OUTPUT_DIR)
    plot_radius_visualisation(stations, spatial_groups, lat_boundaries,
                              SEG_MODE, OUTPUT_DIR,
                              radius_km=KERNEL_RADIUS_KM,
                              rep_lon=12.0, rep_lat=41.0)

    q_hat = plot_scree(scores, d_mean, d_var, d_lcor, OUTPUT_DIR)
    plot_components(Xw, V, scores, stations, n_stations, n_times, dates, q_hat, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(" DONE")
    print(f" q_hat nonstationary : {q_hat}")
    print(f" p-q_hat stationary  : {p - q_hat}")
    print(f" Figures in          : {OUTPUT_DIR}/")
    print("=" * 60)
    print("\nFigures produced:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".pdf"):
            print(f"  {f}")


if __name__ == "__main__":
    main()