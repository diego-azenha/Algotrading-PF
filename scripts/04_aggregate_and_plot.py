# scripts/06_aggregate_and_plot.py
"""
Aggregate per-window IRFs and summary to produce tables/figures similar to the article.

Inputs:
 - models/svar_ith_results.parquet
 - models/irfs/*.npz

Outputs:
 - analysis_outputs/tables/*.csv
 - analysis_outputs/figures/figure3.png / .pdf
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import statsmodels.api as sm

ROOT = Path(".")
MODEL_DIR = Path("models")                     # <-- models directory (outside windows_parquet)
SUMMARY_PQ = MODEL_DIR / "svar_ith_results.parquet"
IRF_GLOB = MODEL_DIR / "irfs" / "*.npz"
OUT_DIR = Path("analysis_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
TAB_DIR = OUT_DIR / "tables"; TAB_DIR.mkdir(exist_ok=True)

def load_all_irfs(irf_glob=IRF_GLOB):
    files = sorted(glob(str(irf_glob)))
    irf_list = []
    for f in tqdm(files, desc="loading irfs"):
        try:
            data = np.load(f, allow_pickle=True)
            s_irf = data["s_irf"]
            meta = data["meta"].item() if "meta" in data else {}
            irf_list.append((s_irf, meta))
        except Exception:
            continue
    return irf_list

def compute_long_run_impacts_from_irf(s_irf):
    cum = np.cumsum(s_irf, axis=0)
    last = cum[-1]
    return last

def main():
    if not SUMMARY_PQ.exists():
        raise SystemExit("Missing summary parquet. Run scripts/05_estimate_svar_ith.py first.")
    df_sum = pd.read_parquet(SUMMARY_PQ)
    print("Loaded summary with", len(df_sum), "rows")

    cols_t3 = ["br","bf","omega_r_1","omega_r_2","omega_r_3","omega_f_1","omega_f_2","omega_f_3","t_br","t_bf"]
    t3 = df_sum[cols_t3].copy()
    stats = []
    for col in ["br","bf","omega_r_1","omega_r_2","omega_r_3","omega_f_1","omega_f_2","omega_f_3"]:
        ser = df_sum[col].dropna()
        stats.append({
            "var": col,
            "mean": ser.mean(),
            "sd": ser.std(),
            "1%": ser.quantile(0.01),
            "5%": ser.quantile(0.05),
            "25%": ser.quantile(0.25),
            "50%": ser.quantile(0.5),
            "75%": ser.quantile(0.75),
            "95%": ser.quantile(0.95),
            "99%": ser.quantile(0.99)
        })
    df_t3 = pd.DataFrame(stats).set_index("var")
    df_t3.to_csv(TAB_DIR / "table3.csv")
    print("Wrote table3.csv")

    irf_list = load_all_irfs()
    lr_list = []
    for s_irf, meta in tqdm(irf_list, desc="LR impacts"):
        try:
            last = compute_long_run_impacts_from_irf(s_irf)
            lr_list.append({
                "Irr": float(last[0,0]),
                "Irf": float(last[0,1]),
                "Ifr": float(last[1,0]),
                "Iff": float(last[1,1])
            })
        except Exception:
            continue
    df_lr = pd.DataFrame(lr_list)
    if not df_lr.empty:
        def summary_table(df, col):
            ser = df[col].dropna()
            return {
                "mean": ser.mean(), "sd": ser.std(),
                "1%": ser.quantile(0.01), "5%": ser.quantile(0.05),
                "25%": ser.quantile(0.25), "50%": ser.quantile(0.5),
                "75%": ser.quantile(0.75), "95%": ser.quantile(0.95), "99%": ser.quantile(0.99)
            }
        tbl4 = {c: summary_table(df_lr, c) for c in df_lr.columns}
        pd.DataFrame(tbl4).T.to_csv(TAB_DIR / "table4.csv")
        print("Wrote table4.csv")
    else:
        print("No IRFs found to build Table 4")

    if len(irf_list)==0:
        print("No IRF files to plot Figure 3.")
        return

    H = irf_list[0][0].shape[0]
    arrs = np.stack([s_irf for s_irf,meta in irf_list if s_irf.shape[0]==H], axis=0)
    def stats_for_pair(i,j):
        data = arrs[:,:,i,j]
        median = np.median(data, axis=0)
        p05 = np.percentile(data, 5, axis=0)
        p95 = np.percentile(data, 95, axis=0)
        return median, p05, p95

    irf_rr_m, irf_rr_05, irf_rr_95 = stats_for_pair(0,0)
    irf_rf_m, irf_rf_05, irf_rf_95 = stats_for_pair(0,1)
    irf_fr_m, irf_fr_05, irf_fr_95 = stats_for_pair(1,0)
    irf_ff_m, irf_ff_05, irf_ff_95 = stats_for_pair(1,1)
    xs = np.arange(H)

    fig, axes = plt.subplots(2,2, figsize=(10,8))
    def plot_panel(ax, median, p05, p95, title):
        ax.plot(xs, median, marker='o', linestyle='-', markersize=4)
        ax.fill_between(xs, p05, p95, alpha=0.25)
        ax.axhline(0, color='k', linewidth=0.6)
        ax.set_title(title)
    plot_panel(axes[0,0], irf_rr_m, irf_rr_05, irf_rr_95, "Return-to-Return Impact (IRF_rr)")
    plot_panel(axes[0,1], irf_rf_m, irf_rf_05, irf_rf_95, "Flow-to-Return Impact (IRF_rf)")
    plot_panel(axes[1,0], irf_fr_m, irf_fr_05, irf_fr_95, "Return-to-Flow Impact (IRF_fr)")
    plot_panel(axes[1,1], irf_ff_m, irf_ff_05, irf_ff_95, "Flow-to-Flow Impact (IRF_ff)")
    plt.tight_layout()
    fig_filename = FIG_DIR / "figure3.png"
    fig.savefig(fig_filename, dpi=300)
    fig.savefig(FIG_DIR / "figure3.pdf")
    print("Wrote Figure 3 to", fig_filename)

    activity_pq = Path("windows_parquet/windows_activity.parquet")
    if activity_pq.exists():
        act = pd.read_parquet(activity_pq)
        merged = df_sum.merge(act, on=["market_date","symbol","window_id"], how="inner")
        if set(["D_inv","NE","ASE","SPR"]).issubset(merged.columns):
            y_br = merged["br"]
            X = merged[["D_inv","NE","ASE","SPR"]]
            X = sm.add_constant(X)
            res_br = sm.OLS(y_br, X).fit(cov_type='cluster', cov_kwds={'groups': merged['market_date']})
            with open(TAB_DIR / "table5.txt", "w") as f:
                f.write(res_br.summary().as_text())
            print("Wrote regression results to table5.txt")
        else:
            print("Activity columns not found in windows_activity.parquet with expected names (D_inv,NE,ASE,SPR).")
    else:
        print("No activity file for Table 5 regression. Provide windows_parquet/windows_activity.parquet to compute Table 5.")

if __name__ == "__main__":
    main()
