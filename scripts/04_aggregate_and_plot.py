# scripts/06_aggregate_and_plot.py
"""
Aggregate per-window IRFs and summary to produce tables/figures similar to the article.

Inputs:
 - models/svar_ith_results.parquet
 - models/irfs/*.npz

Outputs (PNG + CSV/TXT):
 - analysis_outputs/tables/table2_stats.csv
 - analysis_outputs/tables/table3_ann_regressions.csv (+ .txt summary)
 - analysis_outputs/tables/table_prepost_ann.csv
 - analysis_outputs/figures/figure2_intraday_<var>.png  (one per var)
 - analysis_outputs/figures/figure3_prepost_br.png, figure3_prepost_bf.png
 - analysis_outputs/figures/figure4_irf_rr.png, _rf.png, _fr.png, _ff.png
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import statsmodels.api as sm

# ---------------- Visual configuration ----------------
PALETTE = {
    "all":   "#1f77b4",   # blue
    "ann":   "#d62728",   # red
    "noann": "#7f7f7f"    # gray
}
MARKER_SIZE = 4
LINE_WIDTH = 1.2
FILL_ALPHA = 0.18
GRID_ALPHA = 0.25
LABEL_FONT_SIZE = 11
TICK_FONT_SIZE = 10
TITLE_FONT_SIZE = 13
LEGEND_FONTSIZE = 9

plt.style.use("seaborn-v0_8-whitegrid")

# ---------------- Paths ----------------
ROOT = Path(".")
MODEL_DIR = Path("models")
SUMMARY_PQ = MODEL_DIR / "svar_ith_results.parquet"
IRF_GLOB = MODEL_DIR / "irfs" / "*.npz"
OUT_DIR = Path("analysis_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
TAB_DIR = OUT_DIR / "tables"; TAB_DIR.mkdir(exist_ok=True)

# ---------------- Utilities ----------------
def load_all_irfs(irf_glob=IRF_GLOB):
    files = sorted(glob(str(irf_glob)))
    irf_list = []
    for f in tqdm(files, desc="loading irfs"):
        try:
            data = np.load(f, allow_pickle=True)
            s_irf = data["s_irf"]  # shape (H, 2, 2)
            meta = {}
            if "meta" in data:
                try:
                    m = data["meta"]
                    if hasattr(m, "item"):
                        item = m.item()
                        if isinstance(item, dict):
                            meta = item
                        else:
                            try:
                                meta = dict(item)
                            except Exception:
                                meta = {}
                    elif isinstance(m, dict):
                        meta = m
                    else:
                        try:
                            meta = dict(m.tolist())
                        except Exception:
                            meta = {}
                except Exception:
                    meta = {}
            irf_list.append((s_irf, meta))
        except Exception:
            continue
    return irf_list

def compute_long_run_impacts_from_irf(s_irf):
    cum = np.cumsum(s_irf, axis=0)
    return cum[-1]

def rescale_irf_to_original_units(s_irf, scale_ofi):
    if scale_ofi is None or scale_ofi == 1.0:
        return s_irf
    s = s_irf.copy()
    s[:, 1, :] = s[:, 1, :] * float(scale_ofi)
    return s

def minutes_of_day(h,m):
    return int(h*60 + m)

# ---------------- Table 2 (summary stats) ----------------
def save_table2_stats(df_sum, out_path):
    cols = ["br","bf","omega_r_1","omega_r_2","omega_r_3","omega_f_1","omega_f_2","omega_f_3","t_br","t_bf"]
    present = [c for c in cols if c in df_sum.columns]
    if len(present) == 0:
        print("No columns present to build Table 2 stats.")
        return None
    scale_median = None
    if "scale_ofi" in df_sum.columns:
        try:
            scale_median = float(df_sum["scale_ofi"].dropna().median())
            print(f"Detected scale_ofi median={scale_median}. Rescaling omega_f columns.")
        except Exception:
            scale_median = None
    rows = []
    for col in present:
        ser = df_sum[col].dropna().astype(float)
        if ser.empty:
            continue
        if scale_median is not None and col.startswith("omega_f"):
            ser = ser * scale_median
        rows.append({
            "var": col,
            "mean": ser.mean(),
            "sd": ser.std(ddof=0),
            "1%": ser.quantile(0.01),
            "5%": ser.quantile(0.05),
            "25%": ser.quantile(0.25),
            "50%": ser.quantile(0.5),
            "75%": ser.quantile(0.75),
            "95%": ser.quantile(0.95),
            "99%": ser.quantile(0.99),
            "n": ser.shape[0]
        })
    if rows:
        df_stats = pd.DataFrame(rows).set_index("var")
        df_stats.to_csv(out_path)
        print("Wrote Table 2 stats to", out_path)
        return df_stats
    else:
        print("No statistics computed for Table 2.")
        return None

# ---------------- Figure 2: intraday evolution (one PNG per variable) ----------------
def build_intraday_evolution(df_sum):
    if "window_start_utc" not in df_sum.columns:
        print("window_start_utc missing; cannot build intraday evolution figure.")
        return
    df = df_sum.copy()
    df["window_start_utc"] = pd.to_datetime(df["window_start_utc"], utc=True, errors="coerce")
    df["minutes"] = df["window_start_utc"].dt.hour*60 + df["window_start_utc"].dt.minute
    ann_col = "ANN_t" if "ANN_t" in df.columns else None
    plot_vars = [v for v in ["br","bf","omega_r_1","omega_f_1"] if v in df.columns]
    if not plot_vars:
        print("No variables (br/bf/omega) present for intraday evolution.")
        return
    agg_all = df.groupby("minutes")[plot_vars].median().reset_index().sort_values("minutes")
    agg_ann = df[df[ann_col]==1].groupby("minutes")[plot_vars].median().reset_index().sort_values("minutes") if ann_col else None
    agg_noann = df[df[ann_col]==0].groupby("minutes")[plot_vars].median().reset_index().sort_values("minutes") if ann_col else None

    xs = agg_all["minutes"].values
    for v in plot_vars:
        fig, ax = plt.subplots(figsize=(10,3.2))
        ax.plot(xs, agg_all[v].values, color=PALETTE["all"], marker='o', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label="all")
        if agg_ann is not None and v in agg_ann.columns:
            ax.plot(agg_ann["minutes"].values, agg_ann[v].values, color=PALETTE["ann"], linestyle='--', marker='o', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label="ann days")
        if agg_noann is not None and v in agg_noann.columns:
            ax.plot(agg_noann["minutes"].values, agg_noann[v].values, color=PALETTE["noann"], linestyle=':', marker='o', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label="no-ann days")
        ax.set_title(v, fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel("minutes since midnight (UTC)", fontsize=LABEL_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        ax.grid(alpha=GRID_ALPHA)
        ax.legend(fontsize=LEGEND_FONTSIZE, frameon=True, facecolor="white", framealpha=0.6)
        out_fig_png = FIG_DIR / f"figure2_intraday_{v}.png"
        fig.tight_layout()
        fig.savefig(out_fig_png, dpi=300)
        plt.close(fig)
        print("Wrote", out_fig_png)

# ---------------- Table 3: announcement regressions ----------------
def ann_regressions(df_sum):
    df = df_sum.copy()
    if "window_start_utc" in df.columns:
        df["window_start_utc"] = pd.to_datetime(df["window_start_utc"], utc=True, errors="coerce")
    if not {"market_date","symbol","window_start_utc"}.issubset(df.columns):
        print("market_date/symbol/window_start_utc required for announcement regressions. Skipping.")
        return
    df = df.sort_values(["market_date","symbol","window_start_utc"]).reset_index(drop=True)
    df["seq_within_day_sym"] = df.groupby(["market_date","symbol"])["window_start_utc"].rank(method="first").astype(int)
    if "ANN_t" not in df.columns:
        df["ANN_t"] = 0
    df["ANN_t_lag1"] = df.groupby(["market_date","symbol"])["ANN_t"].shift(1).fillna(0)
    df["ANN_t_lag2"] = df.groupby(["market_date","symbol"])["ANN_t"].shift(2).fillna(0)
    df["ANN_t_lead1"] = df.groupby(["market_date","symbol"])["ANN_t"].shift(-1).fillna(0)
    df["ANN_t_lead2"] = df.groupby(["market_date","symbol"])["ANN_t"].shift(-2).fillna(0)
    df["ANN_minus_t"] = df.get("ANN_minus_t", 0)
    df["ANN_minus_t_lag1"] = df.groupby(["market_date","symbol"])["ANN_minus_t"].shift(1).fillna(0)

    controls = []
    activity_pq = Path("windows_parquet/windows_activity.parquet")
    if activity_pq.exists():
        try:
            act = pd.read_parquet(activity_pq)
            if "market_date" in act.columns:
                try:
                    act["market_date"] = pd.to_datetime(act["market_date"]).dt.date
                except Exception:
                    pass
            merged = df.merge(act, on=["market_date","symbol","window_id"], how="left")
            df = merged
            print("Merged activity metrics for regressions from windows_activity.parquet")
            for c in ["D_inv","NE","ASE","SPR","avg_spread","depth"]:
                if c in df.columns:
                    controls.append(c)
        except Exception as e:
            print("Failed to load/merge windows_activity:", e)
    else:
        for c in ["D_inv","NE","ASE","SPR","avg_spread","depth"]:
            if c in df.columns:
                controls.append(c)

    regressors = [r for r in ["ANN_t","ANN_t_lag1","ANN_t_lag2","ANN_t_lead1","ANN_t_lead2","ANN_minus_t","ANN_minus_t_lag1"] if r in df.columns]
    out_rows = []
    table3_csv = TAB_DIR / "table3_ann_regressions.csv"
    table3_txt = TAB_DIR / "table3_ann_regressions.txt"
    # clear text
    open(table3_txt, "w").close()
    for dep in ["br","bf"]:
        if dep not in df.columns:
            continue
        Xcols = regressors + controls
        Xcols_present = [c for c in Xcols if c in df.columns]
        if len(Xcols_present) == 0:
            print(f"No regressors found for {dep}. Skipping.")
            continue
        sub = df.dropna(subset=[dep]+Xcols_present).copy()
        if sub.shape[0] < 20:
            print(f"Too few obs for regression of {dep}. n={sub.shape[0]}")
            continue
        X = sm.add_constant(sub[Xcols_present])
        y = sub[dep].astype(float)
        cluster_by = sub["market_date"] if "market_date" in sub.columns else None
        try:
            if cluster_by is not None:
                res = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': cluster_by})
            else:
                res = sm.OLS(y, X).fit()
            for coef in res.params.index:
                out_rows.append({
                    "dependent": dep,
                    "regressor": coef,
                    "coef": float(res.params[coef]),
                    "se": float(res.bse[coef]),
                    "pvalue": float(res.pvalues[coef]),
                    "nobs": int(res.nobs),
                    "r2": float(res.rsquared)
                })
            with open(table3_txt, "a") as f:
                f.write(f"Regression for {dep}\n")
                f.write(res.summary().as_text())
                f.write("\n\n")
        except Exception as e:
            print(f"Regression failed for {dep}: {e}")
            continue
    if out_rows:
        pd.DataFrame(out_rows).to_csv(table3_csv, index=False)
        print("Wrote announcement regressions to", table3_csv, "and text summary to", table3_txt)
    else:
        print("No regression outputs produced for announcements.")

# ---------------- Figure 3: pre/post announcement (separate PNGs for br and bf) ----------------
def build_prepost_ann_plot(df_sum, lags=2):
    if "ANN_t" not in df_sum.columns:
        print("No ANN_t column present; cannot build pre/post announcement plot.")
        return
    df = df_sum.copy()
    if not {"market_date","symbol","window_start_utc","window_id"}.issubset(df.columns):
        df = df.sort_values(["market_date","symbol","window_start_utc"]).reset_index(drop=True)
    df = df.sort_values(["market_date","symbol","window_start_utc"]).reset_index(drop=True)
    df["seq"] = df.groupby(["market_date","symbol"]).cumcount()
    events = df[df["ANN_t"]==1]
    if events.empty:
        print("No announcement windows found (ANN_t==1).")
        return
    records = []
    for _, ev in events.iterrows():
        mdate = ev["market_date"]; sym = ev["symbol"]; seq = int(ev["seq"])
        sel = df[(df["market_date"]==mdate) & (df["symbol"]==sym) & (df["seq"].between(seq-lags, seq+lags))]
        for _, r in sel.iterrows():
            rel = int(r["seq"]) - seq
            records.append({
                "market_date": mdate,
                "symbol": sym,
                "rel_lag": rel,
                "br": r.get("br", np.nan),
                "bf": r.get("bf", np.nan)
            })
    if len(records) == 0:
        print("No records collected for pre/post announcement analysis.")
        return
    df_r = pd.DataFrame(records)
    agg = df_r.groupby("rel_lag").agg({"br":["median","mean","std","count"], "bf":["median","mean","std","count"]})
    agg.columns = ["_".join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index().sort_values("rel_lag")
    agg.to_csv(TAB_DIR / "table_prepost_ann.csv", index=False)

    # BR plot
    if "br_median" in agg.columns:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(agg["rel_lag"], agg["br_median"], marker='o', color=PALETTE["all"], linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
        ax.fill_between(agg["rel_lag"], agg["br_median"] - agg["br_std"], agg["br_median"] + agg["br_std"], color=PALETTE["all"], alpha=FILL_ALPHA)
        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        ax.set_title("Pre/Post Announcement: br (median ± sd)", fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel("relative window (0 = announcement)", fontsize=LABEL_FONT_SIZE)
        ax.tick_params(labelsize=TICK_FONT_SIZE)
        ax.grid(alpha=GRID_ALPHA)
        out_png = FIG_DIR / "figure3_prepost_br.png"
        fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)
        print("Wrote", out_png)

    # BF plot
    if "bf_median" in agg.columns:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(agg["rel_lag"], agg["bf_median"], marker='o', color=PALETTE["all"], linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
        ax.fill_between(agg["rel_lag"], agg["bf_median"] - agg["bf_std"], agg["bf_median"] + agg["bf_std"], color=PALETTE["all"], alpha=FILL_ALPHA)
        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        ax.set_title("Pre/Post Announcement: bf (median ± sd)", fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel("relative window (0 = announcement)", fontsize=LABEL_FONT_SIZE)
        ax.tick_params(labelsize=TICK_FONT_SIZE)
        ax.grid(alpha=GRID_ALPHA)
        out_png = FIG_DIR / "figure3_prepost_bf.png"
        fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)
        print("Wrote", out_png)

# ---------------- Figure 4: IRFs (save each panel separately) ----------------
def build_irf_figure(irf_list):
    if not irf_list:
        print("No IRFs to plot.")
        return
    normalized_irfs = []
    H_target = None
    for s_irf, meta in irf_list:
        try:
            scale_ofi = None
            if isinstance(meta, dict) and "scale_ofi" in meta:
                try:
                    scale_ofi = float(meta["scale_ofi"])
                except Exception:
                    scale_ofi = None
            s_irf_rs = rescale_irf_to_original_units(s_irf, scale_ofi if scale_ofi is not None else 1.0)
            H = s_irf_rs.shape[0]
            if H_target is None:
                H_target = H
                normalized_irfs.append(s_irf_rs)
            else:
                if H == H_target:
                    normalized_irfs.append(s_irf_rs)
                else:
                    continue
        except Exception:
            continue
    if not normalized_irfs:
        print("No compatible IRFs to aggregate.")
        return
    arrs = np.stack(normalized_irfs, axis=0)  # (N, H, 2, 2)
    def stats_for_pair(i,j):
        data = arrs[:, :, i, j]
        median = np.median(data, axis=0)
        p05 = np.percentile(data, 5, axis=0)
        p95 = np.percentile(data, 95, axis=0)
        return median, p05, p95
    H = arrs.shape[1]
    xs = np.arange(H)
    rr_m, rr_05, rr_95 = stats_for_pair(0,0)
    rf_m, rf_05, rf_95 = stats_for_pair(0,1)
    fr_m, fr_05, fr_95 = stats_for_pair(1,0)
    ff_m, ff_05, ff_95 = stats_for_pair(1,1)

    # Return-to-Return
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(xs, rr_m, marker='o', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color=PALETTE["all"])
    ax.fill_between(xs, rr_05, rr_95, alpha=FILL_ALPHA, color=PALETTE["all"])
    ax.axhline(0, color='k', linewidth=0.6)
    ax.set_title("Return-to-Return (IRF_rr)", fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("horizon (s)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("bps", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.grid(alpha=GRID_ALPHA)
    out_png = FIG_DIR / "figure4_irf_rr.png"
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)
    print("Wrote", out_png)

    # Flow-to-Return
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(xs, rf_m, marker='o', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color=PALETTE["all"])
    ax.fill_between(xs, rf_05, rf_95, alpha=FILL_ALPHA, color=PALETTE["all"])
    ax.axhline(0, color='k', linewidth=0.6)
    ax.set_title("Flow-to-Return (IRF_rf)", fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("horizon (s)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("bps per flow-unit", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.grid(alpha=GRID_ALPHA)
    out_png = FIG_DIR / "figure4_irf_rf.png"
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)
    print("Wrote", out_png)

    # Return-to-Flow
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(xs, fr_m, marker='o', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color=PALETTE["all"])
    ax.fill_between(xs, fr_05, fr_95, alpha=FILL_ALPHA, color=PALETTE["all"])
    ax.axhline(0, color='k', linewidth=0.6)
    ax.set_title("Return-to-Flow (IRF_fr)", fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("horizon (s)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("OFI units", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.grid(alpha=GRID_ALPHA)
    out_png = FIG_DIR / "figure4_irf_fr.png"
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)
    print("Wrote", out_png)

    # Flow-to-Flow
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(xs, ff_m, marker='o', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color=PALETTE["all"])
    ax.fill_between(xs, ff_05, ff_95, alpha=FILL_ALPHA, color=PALETTE["all"])
    ax.axhline(0, color='k', linewidth=0.6)
    ax.set_title("Flow-to-Flow (IRF_ff)", fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("horizon (s)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("OFI units", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.grid(alpha=GRID_ALPHA)
    out_png = FIG_DIR / "figure4_irf_ff.png"
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)
    print("Wrote", out_png)

# ---------------- Main ----------------
def main():
    if not SUMMARY_PQ.exists():
        raise SystemExit("Missing summary parquet. Run scripts/05_estimate_svar_ith.py first.")
    df_sum = pd.read_parquet(SUMMARY_PQ)
    print("Loaded summary with", len(df_sum), "rows")
    if "window_start_utc" in df_sum.columns:
        df_sum["window_start_utc"] = pd.to_datetime(df_sum["window_start_utc"], utc=True, errors="coerce")
    if "market_date" in df_sum.columns:
        try:
            df_sum["market_date"] = pd.to_datetime(df_sum["market_date"]).dt.date
        except Exception:
            pass

    # Table 2
    table2_path = TAB_DIR / "table2_stats.csv"
    save_table2_stats(df_sum, table2_path)

    # Figure 2: intraday (one png per var)
    build_intraday_evolution(df_sum)

    # Table 3: regressions
    ann_regressions(df_sum)

    # Figure 3: pre/post announcement (separate PNGs)
    build_prepost_ann_plot(df_sum)

    # Figure 4: IRFs (separate PNGs)
    irf_list = load_all_irfs()
    build_irf_figure(irf_list)

    print("All done.")

if __name__ == "__main__":
    main()
