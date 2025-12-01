# scripts/05_estimate_svar_ith.py
"""
Parallelized & optimized estimator for bivariate SVAR identified through heteroskedasticity (ITH),
with integration of macro news releases.

Outputs:
 - models/svar_ith_results.parquet
 - models/irfs/<market_date>_<symbol>_<window_id>.npz
 - models/svar_ith_failed.csv

Notes:
 - The script will attempt to find macro announcements CSV/Parquet automatically
   in likely locations (data/, clean_data/economic_releases/, /mnt/data).
 - It adds ANN flags per 15-min window and per substate (ANN_t, ANN_minus_t, ANN_sub_1..S, ANN_minus_sub_1..S).
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time
from glob import glob
from tqdm import tqdm
from statsmodels.tsa.api import VAR
from scipy.optimize import least_squares
import warnings
import multiprocessing as mp
import traceback

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
DESC_1S_PQ = Path("windows_parquet/descriptives/descriptive_1s.parquet")
WINDOWS_INDEX_PQ = Path("windows_parquet/windows_index.parquet")

OUT_DIR = Path("models")                     # models folder (root)
IRF_DIR = OUT_DIR / "irfs"
OUT_SUMMARY_PQ = OUT_DIR / "svar_ith_results.parquet"
FAILED_CSV = OUT_DIR / "svar_ith_failed.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
IRF_DIR.mkdir(parents=True, exist_ok=True)

# estimator config
S = 3                 # number of substates per window (paper uses 3: three 5-min blocks)
MAX_P = 6
IRF_HORIZON = 10      # horizon (in seconds / 1s steps) for IRFs
MIN_OBS_PER_SUBSTATE = 10
MIN_TOTAL_OBS = 30
EPS = 1e-8

# solver bounds / tolerances
BR_BF_BOUNDS = (-5.0, 5.0)
OMEGA_UPPER = 1e6
XTOL = 1e-8
FTOL = 1e-8
MAX_NFEV = 2000

# parallel settings (tune these)
N_JOBS = max(1, mp.cpu_count() - 1)   # default: all but one core
CHUNKSIZE = 32                       # mapping chunksize for imap_unordered

# safety
POOL_START_METHOD = "fork" if hasattr(mp, "get_start_method") and mp.get_start_method(allow_none=True) != "spawn" else None

# candidate news file paths to try (will pick first that exists)
CANDIDATE_NEWS_PATHS = [
    Path("economic_releases") / "macro_announcements_2022_final.csv",
]

# ----------------------------------------

# Global variables used by worker processes (set in initializer)
GLOBAL_DF1S = None
GLOBAL_NEWS = None   # DataFrame of news with 'release_time_utc' (UTC tz-aware) and 'neg_surprise' columns

# ---------------- utility / estimation functions ----------------

def build_B_from_params(br, bf):
    return np.array([[1.0, -br], [-bf, 1.0]])

def rank_condition_ok(Sigma_s_list, tol=1e-14):
    S_local = len(Sigma_s_list)
    sig_r = [Sigma[0,0] for Sigma in Sigma_s_list]
    sig_rf = [Sigma[0,1] for Sigma in Sigma_s_list]
    for i in range(S_local):
        for j in range(i+1, S_local):
            val = sig_r[i]*sig_rf[j] - sig_r[j]*sig_rf[i]
            if abs(val) > tol:
                return True
    return False

def Sigma_pred_from_params(params, Sigma_s_list):
    br = params[0]
    bf = params[1]
    S_local = len(Sigma_s_list)
    omega_r = params[2:2+S_local]
    omega_f = params[2+S_local:2+2*S_local]
    B = build_B_from_params(br, bf)
    residuals = []
    for s in range(S_local):
        Sigma = Sigma_s_list[s]
        pred = B @ Sigma @ B.T
        diag = np.diag([omega_r[s]**2, omega_f[s]**2])
        scale = float(max(np.sqrt(np.max(np.abs(Sigma))), 1.0))
        residuals.append((pred[0,0] - diag[0,0]) / scale)
        residuals.append((pred[1,1] - diag[1,1]) / scale)
        residuals.append(pred[0,1] / scale)
    return np.array(residuals)

def initial_guess_mom(Sigma_s_list):
    sig_r = np.array([S[0,0] for S in Sigma_s_list])
    sig_f = np.array([S[1,1] for S in Sigma_s_list])
    sig_rf = np.array([S[0,1] for S in Sigma_s_list])
    with np.errstate(divide='ignore', invalid='ignore'):
        br_mom = np.nanmean(np.where(sig_f>0, sig_rf / sig_f, 0.5))
        bf_mom = np.nanmean(np.where(sig_r>0, sig_rf / sig_r, 0.2))
    br_mom = float(np.clip(br_mom, BR_BF_BOUNDS[0]+0.01, BR_BF_BOUNDS[1]-0.01))
    bf_mom = float(np.clip(bf_mom, BR_BF_BOUNDS[0]+0.01, BR_BF_BOUNDS[1]-0.01))
    omega_r_init = [float(max(np.sqrt(max(S[0,0], EPS)), 1e-4)) for S in Sigma_s_list]
    omega_f_init = [float(max(np.sqrt(max(S[1,1], EPS)), 1e-4)) for S in Sigma_s_list]
    return np.array([br_mom, bf_mom] + omega_r_init + omega_f_init)

def solve_ith(Sigma_s_list, x0=None):
    S_local = len(Sigma_s_list)
    if x0 is None:
        x0 = initial_guess_mom(Sigma_s_list)
    low = np.array([BR_BF_BOUNDS[0], BR_BF_BOUNDS[0]] + [EPS]*S_local + [EPS]*S_local)
    high = np.array([BR_BF_BOUNDS[1], BR_BF_BOUNDS[1]] + [OMEGA_UPPER]*S_local + [OMEGA_UPPER]*S_local)
    def fun(x):
        return Sigma_pred_from_params(x, Sigma_s_list)
    try:
        sol = least_squares(fun, x0, bounds=(low,high), xtol=XTOL, ftol=FTOL, max_nfev=MAX_NFEV, verbose=0)
        return {"success": sol.success, "x": sol.x, "cost": float(sol.cost), "message": sol.message, "jac": sol.jac}
    except Exception as e:
        return {"success": False, "x": None, "cost": None, "message": f"solve_failed: {e}", "jac": None}

def select_var_lags(y_df, maxlags=MAX_P):
    model = VAR(y_df)
    try:
        sel = model.select_order(maxlags)
        aic = None
        p = 1
        if hasattr(sel, "aic") and sel.aic is not None:
            try:
                aic_series = sel.aic
                p = int(aic_series.idxmin())
                aic = float(aic_series.min())
            except Exception:
                p = 1
                aic = np.nan
        else:
            p = 1
            aic = np.nan
    except Exception:
        p = 1
        aic = np.nan
    return max(1, p), aic

def fit_var(y_df, p):
    model = VAR(y_df)
    res = model.fit(p)
    return res

def compute_structural_irf(var_res, B, horizon=IRF_HORIZON):
    try:
        red_irf = var_res.irf(horizon)
        rf = red_irf.irfs
        Binv = np.linalg.inv(B)
        hlen = rf.shape[0]
        s_irf = np.empty_like(rf)
        for k in range(hlen):
            s_irf[k] = rf[k] @ Binv
        return s_irf
    except Exception:
        return None

def jacobian_se_approx(jac):
    try:
        JTJ = jac.T @ jac
        cov = np.linalg.pinv(JTJ)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        return se
    except Exception:
        return None

# ---------------- News loader / helper ----------------

def find_news_path():
    for p in CANDIDATE_NEWS_PATHS:
        if p.exists():
            return p
    return None

def load_and_normalize_news():
    """
    Simple, silent and deterministic loader for macro announcements.
    Expects columns:
        indicator
        release_datetime_CT
        actual_raw
        forecast_raw
        negative
    Returns DataFrame with:
        release_time_utc, market_date, series_name, symbol, actual, consensus,
        surprise, neg_surprise
    """
    p = find_news_path()
    if p is None:
        return pd.DataFrame(columns=[
            "release_time_utc","market_date","series_name","symbol",
            "actual","consensus","surprise","neg_surprise"
        ])

    # Load file
    if p.suffix.lower() in [".parquet", ".pq"]:
        raw = pd.read_parquet(p)
    else:
        raw = pd.read_csv(p, dtype=str)

    #--- Extract columns explicitly (no guesswork) ---
    time_col     = "release_datetime_CT"
    actual_col   = "actual_raw"
    forecast_col = "forecast_raw"
    neg_col      = "negative"
    series_col   = "indicator"

    #--- Convert timestamp to UTC ---
    # release_datetime_CT includes offset, so to_datetime(..., utc=True) is correct
    df = pd.DataFrame()
    df["release_time_utc"] = pd.to_datetime(raw[time_col], utc=True, errors="coerce")
    df["market_date"]      = df["release_time_utc"].dt.date

    #--- Clean numeric percentage strings ---
    def clean(s):
        if s is None:
            return np.nan
        s = s.astype(str).fillna("")
        s = s.str.replace(r"[,\s]+", "", regex=True)
        s = s.str.replace(r"\((.*)\)", r"-\1", regex=True)
        s = s.str.replace("%", "", regex=False)
        s = s.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)[0]
        return pd.to_numeric(s, errors="coerce")

    df["actual"]    = clean(raw[actual_col])   if actual_col   in raw else np.nan
    df["consensus"] = clean(raw[forecast_col]) if forecast_col in raw else np.nan
    df["surprise"]  = df["actual"] - df["consensus"]

    #--- neg_surprise ---
    if neg_col in raw:
        # explicit negatives column takes priority
        df["neg_surprise"] = pd.to_numeric(raw[neg_col], errors="coerce").fillna(0).astype(int)
    else:
        df["neg_surprise"] = (df["surprise"] < 0).astype(int).fillna(0) 

    #--- Keep required columns ---
    df["series_name"] = raw[series_col] if series_col in raw else None
    df["symbol"] = None  # no symbol info in your file

    df = df[[
        "release_time_utc", "market_date", "series_name", "symbol",
        "actual", "consensus", "surprise", "neg_surprise"
    ]].sort_values("release_time_utc").reset_index(drop=True)

    return df


# ---------------- Worker initializer ----------------
def worker_init(df1s_serialized, news_serialized):
    """
    initializer for Pool workers; sets global df1s and global news DataFrame.
    df1s_serialized/news_serialized can be DataFrames or path-like (strings).
    """
    global GLOBAL_DF1S, GLOBAL_NEWS
    if isinstance(df1s_serialized, (str, Path)):
        GLOBAL_DF1S = pd.read_parquet(df1s_serialized)
    else:
        GLOBAL_DF1S = df1s_serialized
    if news_serialized is None:
        GLOBAL_NEWS = pd.DataFrame(columns=["release_time_utc","market_date","series_name","symbol","actual","consensus","surprise","neg_surprise"])
    elif isinstance(news_serialized, (str, Path)):
        GLOBAL_NEWS = pd.read_parquet(news_serialized)
    else:
        GLOBAL_NEWS = news_serialized

# ---------------- Worker function ----------------
def process_window(win_tuple):
    """
    Process one window. win_tuple is a tuple of (mdate, sym, ws_iso, we_iso, wid)
    Returns either ("ok", row_dict) or ("fail", fail_tuple)
    """
    try:
        mdate, sym, ws_iso, we_iso, wid = win_tuple
        ws = pd.to_datetime(ws_iso)
        we = pd.to_datetime(we_iso)
        df1s = GLOBAL_DF1S
        news = GLOBAL_NEWS

        # extract seconds inside window
        mask = (df1s["market_date"] == pd.to_datetime(mdate).date()) & (df1s["symbol"] == sym) & (df1s["ts"] >= ws) & (df1s["ts"] < we)
        sub = df1s.loc[mask, ["ts","mid_return_bps","ofi"]].sort_values("ts").reset_index(drop=True)
        if sub.shape[0] < MIN_TOTAL_OBS:
            return ("fail", (mdate, sym, wid, "too_few_total_obs"))
        sub = sub.dropna(subset=["mid_return_bps","ofi"]).reset_index(drop=True)
        n = sub.shape[0]
        if n < MIN_TOTAL_OBS:
            return ("fail", (mdate, sym, wid, "too_few_after_dropna"))

        # News flags for the 15-min window
        ann_flag = 0
        ann_minus_flag = 0
        if not news.empty:
            # Candidate news for same market_date and either same symbol or global (symbol null)
            cand = news[news["market_date"] == pd.to_datetime(mdate).date()]
            if cand.shape[0] > 0:
                # if symbol column present and not null, match either same symbol or null
                if "symbol" in cand.columns:
                    cand_sym = cand[(cand["symbol"].isnull()) | (cand["symbol"] == sym)]
                else:
                    cand_sym = cand
                # check any release_time in [ws,we)
                in_window_mask = (cand_sym["release_time_utc"] >= ws) & (cand_sym["release_time_utc"] < we)
                if in_window_mask.any():
                    ann_flag = int(True)
                    if cand_sym.loc[in_window_mask, "neg_surprise"].any():
                        ann_minus_flag = int(True)

        # split into S substates by time (equal-duration contiguous blocks) for substate-level ANN flags
        sub_ann_flags = [0]*S
        sub_ann_minus_flags = [0]*S
        # compute time splits
        # note: use timestamps dividing the interval ws..we into S equal durations
        t_starts = [ws + (we - ws) * (i / S) for i in range(S)]
        t_ends = [ws + (we - ws) * ((i+1) / S) for i in range(S)]
        if not news.empty:
            cand = news[news["market_date"] == pd.to_datetime(mdate).date()]
            if "symbol" in cand.columns:
                cand = cand[(cand["symbol"].isnull()) | (cand["symbol"] == sym)]
            for k in range(S):
                mask_k = (cand["release_time_utc"] >= t_starts[k]) & (cand["release_time_utc"] < t_ends[k])
                if mask_k.any():
                    sub_ann_flags[k] = 1
                    if cand.loc[mask_k, "neg_surprise"].any():
                        sub_ann_minus_flags[k] = 1

        # estimation: same as before
        idx_bounds = [int(round(i * n / S)) for i in range(S+1)]
        Sigma_s_list = []
        valid_split = True
        var_res = None
        p_selected = 1
        aic_val = np.nan
        try:
            y = sub[["mid_return_bps","ofi"]]
            p_selected, aic_val = select_var_lags(y, maxlags=min(MAX_P, max(1, n//5)))
            var_res = fit_var(y, p_selected)
            resid_arr = np.asarray(var_res.resid)  # shape (T-p, 2)
            resid_start_pos = p_selected
            for s_idx in range(S):
                i0 = idx_bounds[s_idx]
                i1 = idx_bounds[s_idx+1]
                r0 = max(i0, resid_start_pos)
                r1 = max(min(i1, n), resid_start_pos)
                ri0 = r0 - resid_start_pos
                ri1 = r1 - resid_start_pos
                if ri1 - ri0 < MIN_OBS_PER_SUBSTATE:
                    valid_split = False
                    break
                sub_res = resid_arr[ri0:ri1, :].T
                Sigma = np.cov(sub_res, bias=True)
                Sigma_s_list.append(Sigma)
        except Exception as e:
            return ("fail", (mdate, sym, wid, f"var_fit_failed:{str(e)}"))

        if not valid_split:
            Sigma_s_list = []
            ok_alt = True
            for s_idx in range(S):
                i0 = idx_bounds[s_idx]
                i1 = idx_bounds[s_idx+1]
                block = sub.iloc[i0:i1].reset_index(drop=True)
                if len(block) < MIN_OBS_PER_SUBSTATE:
                    ok_alt = False
                    break
                try:
                    p_blk, _ = select_var_lags(block[["mid_return_bps","ofi"]], maxlags=min(MAX_P, max(1, len(block)//5)))
                    res_blk = fit_var(block[["mid_return_bps","ofi"]], p_blk)
                    resid_blk = np.asarray(res_blk.resid)
                    if resid_blk.shape[0] < MIN_OBS_PER_SUBSTATE:
                        ok_alt = False
                        break
                    Sigma = np.cov(resid_blk.T, bias=True)
                    Sigma_s_list.append(Sigma)
                except Exception as e:
                    ok_alt = False
                    break
            if not ok_alt:
                return ("fail", (mdate, sym, wid, "insufficient_substate_obs"))

        # rank check
        if not rank_condition_ok(Sigma_s_list):
            return ("fail", (mdate, sym, wid, "rank_condition_failed"))

        sol = solve_ith(Sigma_s_list)
        if not sol["success"] or sol["x"] is None:
            return ("fail", (mdate, sym, wid, f"ith_solve_failed:{sol.get('message','no_msg')}"))

        x = sol["x"]
        br = float(x[0]); bf = float(x[1])
        omega_r = [float(v) for v in x[2:2+S]]
        omega_f = [float(v) for v in x[2+S:2+2*S]]
        B = build_B_from_params(br, bf)
        s_irf = compute_structural_irf(var_res, B, horizon=IRF_HORIZON)

        # se / tstat approx via jacobian
        jac = sol.get("jac", None)
        se_br = np.nan; se_bf = np.nan; t_br = np.nan; t_bf = np.nan
        if jac is not None:
            se = jacobian_se_approx(jac)
            if se is not None and len(se) >= 2:
                se_br = float(se[0]); se_bf = float(se[1])
                t_br = float(br / se_br) if se_br>0 else np.nan
                t_bf = float(bf / se_bf) if se_bf>0 else np.nan

        # save IRF file from worker (unique name)
        irf_fname = IRF_DIR / f"{mdate}_{sym}_w{wid}.npz"
        try:
            if s_irf is not None:
                np.savez_compressed(str(irf_fname),
                                    s_irf=s_irf,
                                    meta={"market_date": str(mdate), "symbol": sym, "window_id": int(wid),
                                          "br": br, "bf": bf, "omega_r": omega_r, "omega_f": omega_f,
                                          "p_selected": int(p_selected), "aic": float(aic_val)} )
        except Exception:
            return ("fail", (mdate, sym, wid, "irf_save_failed"))

        # build summary row including ANN flags
        row = {
            "market_date": str(mdate),
            "symbol": sym,
            "window_id": int(wid),
            "window_start_utc": ws_iso,
            "window_end_utc": we_iso,
            "n_seconds": int(n),
            "p_selected": int(p_selected),
            "aic": float(aic_val) if not pd.isna(aic_val) else np.nan,
            "br": br,
            "bf": bf,
            "se_br": se_br,
            "se_bf": se_bf,
            "t_br": t_br,
            "t_bf": t_bf,
            "omega_r_1": omega_r[0] if len(omega_r)>0 else np.nan,
            "omega_r_2": omega_r[1] if len(omega_r)>1 else np.nan,
            "omega_r_3": omega_r[2] if len(omega_r)>2 else np.nan,
            "omega_f_1": omega_f[0] if len(omega_f)>0 else np.nan,
            "omega_f_2": omega_f[1] if len(omega_f)>1 else np.nan,
            "omega_f_3": omega_f[2] if len(omega_f)>2 else np.nan,
            # ANN flags for window and for each substate
            "ANN_t": int(ann_flag),
            "ANN_minus_t": int(ann_minus_flag),
            "ANN_sub_1": int(sub_ann_flags[0]),
            "ANN_sub_2": int(sub_ann_flags[1]) if S>1 else 0,
            "ANN_sub_3": int(sub_ann_flags[2]) if S>2 else 0,
            "ANN_minus_sub_1": int(sub_ann_minus_flags[0]),
            "ANN_minus_sub_2": int(sub_ann_minus_flags[1]) if S>1 else 0,
            "ANN_minus_sub_3": int(sub_ann_minus_flags[2]) if S>2 else 0,
            "solver_cost": sol.get("cost", np.nan),
            "solver_message": sol.get("message", "")
        }
        return ("ok", row)
    except Exception as e:
        tb = traceback.format_exc()
        return ("fail", (win_tuple[0], win_tuple[1], win_tuple[4], f"unexpected_error:{str(e)}; tb:{tb}"))

# ---------------- Main ----------------
def main():
    start_time = time.time()
    if not DESC_1S_PQ.exists():
        raise SystemExit("Missing descriptive_1s.parquet. Run script 02 first.")
    if not WINDOWS_INDEX_PQ.exists():
        raise SystemExit("Missing windows_index.parquet. Run script 01 first.")

    print("Loading inputs (main)...")
    df1s = pd.read_parquet(DESC_1S_PQ)
    win_idx = pd.read_parquet(WINDOWS_INDEX_PQ)

    # standardize types
    df1s["ts"] = pd.to_datetime(df1s["ts"], utc=True)
    # ensure market_date is date
    try:
        df1s["market_date"] = pd.to_datetime(df1s["market_date"]).dt.date
    except Exception:
        # if already strings like '2022-01-03'
        df1s["market_date"] = df1s["market_date"].astype(str)
    # ensure required columns
    req_cols = ["market_date","symbol","ts","mid","mid_return_bps","ofi"]
    for c in req_cols:
        if c not in df1s.columns:
            raise SystemExit(f"descriptive_1s missing required column: {c}")

    win_idx["window_start_utc"] = pd.to_datetime(win_idx["window_start_utc"], utc=True)
    win_idx["window_end_utc"] = pd.to_datetime(win_idx["window_end_utc"], utc=True)
    try:
        win_idx["market_date"] = pd.to_datetime(win_idx["market_date"]).dt.date
    except Exception:
        win_idx["market_date"] = win_idx["market_date"].astype(str)

    # load news and normalize
    print("Loading macro announcements (if present)...")
    news = load_and_normalize_news()
    if news.empty:
        print("No news file found or file empty; ANN flags will be zero.")
    else:
        print(f"Loaded news rows: {len(news)} (first rows):")
        print(news.head(5))

    # build list of windows to process
    win_idx_sorted = win_idx.sort_values(["market_date","symbol","window_id"])
    tasks = []
    for _, win in win_idx_sorted.iterrows():
        tasks.append((str(win["market_date"]), win["symbol"], win["window_start_utc"].isoformat(), win["window_end_utc"].isoformat(), int(win["window_id"])))

    print(f"Total windows to process: {len(tasks)}")
    # prepare multiprocessing pool with initializer to set GLOBAL_DF1S and GLOBAL_NEWS in each worker
    pool = None
    try:
        try:
            if POOL_START_METHOD == "fork":
                mp_ctx = mp.get_context("fork")
            else:
                mp_ctx = mp.get_context()
        except Exception:
            mp_ctx = mp.get_context()

        print(f"Starting pool with {N_JOBS} workers (chunksize={CHUNKSIZE})...")
        pool = mp_ctx.Pool(processes=N_JOBS, initializer=worker_init, initargs=(df1s, news))
        it = pool.imap_unordered(process_window, tasks, chunksize=CHUNKSIZE)
        out_rows = []
        failed = []
        pbar = tqdm(total=len(tasks), desc="windows", unit="win")
        for res in it:
            status, payload = res
            if status == "ok":
                out_rows.append(payload)
            else:
                failed.append(payload)
            pbar.update()
        pbar.close()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # write outputs
    if out_rows:
        df_out = pd.DataFrame(out_rows)
        df_out.to_parquet(OUT_SUMMARY_PQ, index=False)
        print("Wrote SVAR ITH summary to:", OUT_SUMMARY_PQ)
    else:
        print("No successful SVAR windows.")

    if failed:
        df_failed = pd.DataFrame(failed, columns=["market_date","symbol","window_id","reason"])
        df_failed.to_csv(FAILED_CSV, index=False)
        print("Some windows failed. See:", FAILED_CSV)
        print(df_failed["reason"].value_counts().head(10))

    print("Done. Elapsed:", round(time.time() - start_time, 2), "s")

if __name__ == "__main__":
    main()
