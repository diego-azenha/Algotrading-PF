#!/usr/bin/env python3
"""
Optimized estimator for bivariate SVAR identified through heteroskedasticity (ITH).
Delivers:
 - models/svar_ith_results.parquet
 - models/irfs/<market_date>_<symbol>_<window_id>.npz
 - models/svar_ith_failed.csv

Design goals:
 - Keep behavior/functionality of original script (ANN flags, S substates, IRFs, solver).
 - Reduce overengineering, clearer flow, modest speed/stability improvements.
"""

import os
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import multiprocessing as mp
from statsmodels.tsa.api import VAR
from scipy.optimize import least_squares

# -------- CONFIG (tune here) ----------
DESC_1S_PQ = Path("windows_parquet/descriptives/descriptive_1s.parquet")
WINDOWS_INDEX_PQ = Path("windows_parquet/windows_index.parquet")

OUT_DIR = Path("models")
IRF_DIR = OUT_DIR / "irfs"
OUT_SUMMARY_PQ = OUT_DIR / "svar_ith_results.parquet"
FAILED_CSV = OUT_DIR / "svar_ith_failed.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
IRF_DIR.mkdir(parents=True, exist_ok=True)

S = 3                       # substates per 15-min window
MAX_P = 6
IRF_HORIZON = 10
MIN_OBS_PER_SUBSTATE = 10
MIN_TOTAL_OBS = 30
SCALE_OFI = 1000.0          # scale factor for OFI units in metadata

BR_BF_BOUNDS = (-5.0, 5.0)
OMEGA_UPPER = 1e6
XTOL = 1e-8
FTOL = 1e-8
MAX_NFEV = 2000

N_JOBS = 4
CHUNKSIZE = 16

# Candidate news file (pick first that exists)
CANDIDATE_NEWS_PATHS = [Path("economic_releases/macro_announcements_2022_final.csv")]

# ---------- small helpers ----------
def find_news_path():
    for p in CANDIDATE_NEWS_PATHS:
        if p.exists():
            return p
    return None

def load_news(path):
    """Return normalized news DataFrame with release_time_utc (tz-aware) and neg_surprise."""
    if path is None:
        return pd.DataFrame(columns=["release_time_utc","market_date","series_name","symbol","actual","consensus","surprise","neg_surprise"])
    p = Path(path)
    if p.suffix.lower() in [".parquet", ".pq"]:
        raw = pd.read_parquet(p)
    else:
        raw = pd.read_csv(p, dtype=str)
    # expected columns in upstream file (will be tolerant)
    time_col = "release_datetime_CT"
    actual_col = "actual_raw"
    forecast_col = "forecast_raw"
    neg_col = "negative"
    series_col = "indicator"
    df = pd.DataFrame()
    df["release_time_utc"] = pd.to_datetime(raw.get(time_col), utc=True, errors="coerce")
    df["market_date"] = df["release_time_utc"].dt.date
    def clean_num(s):
        s = pd.Series(s).astype(str).fillna("")
        s = s.str.replace(r"[,\s]+", "", regex=True)
        s = s.str.replace(r"\((.*)\)", r"-\1", regex=True)
        s = s.str.replace("%", "", regex=False)
        res = s.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)[0]
        return pd.to_numeric(res, errors="coerce")
    df["actual"] = clean_num(raw.get(actual_col)) if actual_col in raw else np.nan
    df["consensus"] = clean_num(raw.get(forecast_col)) if forecast_col in raw else np.nan
    df["surprise"] = df["actual"] - df["consensus"]
    if neg_col in raw:
        df["neg_surprise"] = pd.to_numeric(raw.get(neg_col), errors="coerce").fillna(0).astype(int)
    else:
        df["neg_surprise"] = (df["surprise"] < 0).astype(int).fillna(0)
    df["series_name"] = raw.get(series_col)
    df["symbol"] = raw.get("symbol") if "symbol" in raw else None
    df = df[["release_time_utc","market_date","series_name","symbol","actual","consensus","surprise","neg_surprise"]]
    return df.sort_values("release_time_utc").reset_index(drop=True)

# ---------- ITH solver helpers ----------
def build_B_from_params(br, bf):
    return np.array([[1.0, -br], [-bf, 1.0]])

def Sigma_pred_from_params(params, Sigma_s_list):
    """Residuals vector for least_squares: for each s, (pred00 - w_r^2, pred11 - w_f^2, pred01)."""
    br, bf = params[0], params[1]
    S_local = len(Sigma_s_list)
    omega_r = params[2:2+S_local]
    omega_f = params[2+S_local:2+2*S_local]
    B = build_B_from_params(br, bf)
    out = []
    # scale to roughly unit order to help optimizer
    for s in range(S_local):
        Sigma = Sigma_s_list[s]
        pred = B @ Sigma @ B.T
        scale = float(max(np.sqrt(max(np.max(np.abs(Sigma)), 1.0)), 1.0))
        out.append((pred[0,0] - omega_r[s]**2) / scale)
        out.append((pred[1,1] - omega_f[s]**2) / scale)
        out.append(pred[0,1] / scale)
    return np.array(out)

def initial_guess_mom(Sigma_s_list):
    sig_r = np.array([S[0,0] for S in Sigma_s_list])
    sig_f = np.array([S[1,1] for S in Sigma_s_list])
    sig_rf = np.array([S[0,1] for S in Sigma_s_list])
    # crude MOM
    with np.errstate(divide='ignore', invalid='ignore'):
        br_mom = np.nanmean(np.where(sig_f>0, sig_rf / sig_f, 0.5))
        bf_mom = np.nanmean(np.where(sig_r>0, sig_rf / sig_r, 0.2))
    br_mom = float(np.clip(np.nan_to_num(br_mom, 0.1), BR_BF_BOUNDS[0]+1e-3, BR_BF_BOUNDS[1]-1e-3))
    bf_mom = float(np.clip(np.nan_to_num(bf_mom, 0.1), BR_BF_BOUNDS[0]+1e-3, BR_BF_BOUNDS[1]-1e-3))
    omega_r_init = [float(max(np.sqrt(max(S[0,0], 1e-8)), 1e-4)) for S in Sigma_s_list]
    omega_f_init = [float(max(np.sqrt(max(S[1,1], 1e-8)), 1e-4)) for S in Sigma_s_list]
    return np.array([br_mom, bf_mom] + omega_r_init + omega_f_init)

def solve_ith(Sigma_s_list):
    S_local = len(Sigma_s_list)
    x0 = initial_guess_mom(Sigma_s_list)
    low = np.array([BR_BF_BOUNDS[0], BR_BF_BOUNDS[0]] + [1e-12]*S_local + [1e-12]*S_local)
    high = np.array([BR_BF_BOUNDS[1], BR_BF_BOUNDS[1]] + [OMEGA_UPPER]*S_local + [OMEGA_UPPER]*S_local)
    try:
        sol = least_squares(lambda x: Sigma_pred_from_params(x, Sigma_s_list),
                            x0, bounds=(low, high), xtol=XTOL, ftol=FTOL, max_nfev=MAX_NFEV, verbose=0)
        return {"success": sol.success, "x": sol.x, "cost": float(sol.cost), "message": sol.message, "jac": sol.jac}
    except Exception as e:
        return {"success": False, "x": None, "cost": None, "message": f"solver_exception:{e}", "jac": None}

def jacobian_se_approx(jac):
    try:
        JTJ = jac.T @ jac
        cov = np.linalg.pinv(JTJ)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        return se
    except Exception:
        return None

# ---------- VAR helpers ----------
def select_var_lags(y_df, maxlags=MAX_P):
    try:
        model = VAR(y_df)
        sel = model.select_order(maxlags)
        if hasattr(sel, "aic") and sel.aic is not None:
            aic_series = sel.aic
            p = int(aic_series.idxmin())
            return max(1, p), float(aic_series.min())
    except Exception:
        pass
    return 1, np.nan

def fit_var(y_df, p):
    model = VAR(y_df)
    return model.fit(p)

def compute_structural_irf(var_res, B, horizon=IRF_HORIZON):
    try:
        red_irf = var_res.irf(horizon)
        rf = red_irf.irfs  # (h+1, neqs, neqs)
        Binv = np.linalg.inv(B)
        s_irf = np.empty_like(rf)
        for k in range(rf.shape[0]):
            s_irf[k] = rf[k] @ Binv
        return s_irf
    except Exception:
        return None

# ---------- worker init and work ----------
GLOBAL_DF1S_PATH = None
GLOBAL_NEWS_PATH = None
GLOBAL_DF1S = None
GLOBAL_NEWS = None

def worker_init(df1s_path, news_path):
    global GLOBAL_DF1S_PATH, GLOBAL_NEWS_PATH, GLOBAL_DF1S, GLOBAL_NEWS
    GLOBAL_DF1S_PATH = df1s_path
    GLOBAL_NEWS_PATH = news_path
    # lazy load here to keep pickling light
    try:
        GLOBAL_DF1S = pd.read_parquet(df1s_path)
        if "ts" in GLOBAL_DF1S.columns:
            GLOBAL_DF1S["ts"] = pd.to_datetime(GLOBAL_DF1S["ts"], utc=True, errors="coerce")
        elif "ts_event" in GLOBAL_DF1S.columns:
            GLOBAL_DF1S["ts"] = pd.to_datetime(GLOBAL_DF1S["ts_event"], utc=True, errors="coerce")
        if "market_date" in GLOBAL_DF1S.columns:
            try:
                GLOBAL_DF1S["market_date"] = pd.to_datetime(GLOBAL_DF1S["market_date"]).dt.date
            except Exception:
                pass
    except Exception:
        GLOBAL_DF1S = pd.DataFrame()
    try:
        GLOBAL_NEWS = load_news(news_path) if news_path else pd.DataFrame()
    except Exception:
        GLOBAL_NEWS = pd.DataFrame()

def rank_condition_ok(Sigma_s_list, tol=1e-14):
    sig_r = [S[0,0] for S in Sigma_s_list]
    sig_rf = [S[0,1] for S in Sigma_s_list]
    for i in range(len(Sigma_s_list)):
        for j in range(i+1, len(Sigma_s_list)):
            if abs(sig_r[i]*sig_rf[j] - sig_r[j]*sig_rf[i]) > tol:
                return True
    return False

def process_window(task):
    """task: (market_date_str, symbol, ws_iso, we_iso, window_id)"""
    try:
        mdate, sym, ws_iso, we_iso, wid = task
        ws = pd.to_datetime(ws_iso)
        we = pd.to_datetime(we_iso)
        df1s = GLOBAL_DF1S
        news = GLOBAL_NEWS

        # filter minimal columns
        mask_date = (df1s["market_date"] == (pd.to_datetime(mdate).date() if not isinstance(mdate, pd.Timestamp) else pd.to_datetime(mdate).date()))
        mask_sym = (df1s["symbol"] == sym)
        mask_time = (df1s["ts"] >= ws) & (df1s["ts"] < we)
        sub = df1s.loc[mask_date & mask_sym & mask_time, ["ts","mid_return_bps","ofi"]].sort_values("ts").reset_index(drop=True)
        if sub.shape[0] < MIN_TOTAL_OBS:
            return ("fail", (mdate, sym, wid, "too_few_total_obs"))
        sub = sub.dropna(subset=["mid_return_bps","ofi"]).reset_index(drop=True)
        n = sub.shape[0]
        if n < MIN_TOTAL_OBS:
            return ("fail", (mdate, sym, wid, "too_few_after_dropna"))

        # ANN flags for window and substates
        ann_flag = 0
        ann_minus_flag = 0
        sub_ann_flags = [0]*S
        sub_ann_minus_flags = [0]*S
        if not news.empty:
            cand = news[news["market_date"] == pd.to_datetime(mdate).date()]
            if "symbol" in cand.columns:
                cand = cand[(cand["symbol"].isnull()) | (cand["symbol"] == sym)]
            if not cand.empty:
                in_window = cand[(cand["release_time_utc"] >= ws) & (cand["release_time_utc"] < we)]
                if not in_window.empty:
                    ann_flag = 1
                    ann_minus_flag = int(in_window["neg_surprise"].any())
                # substates
                for k in range(S):
                    t0 = ws + (we - ws) * (k / S)
                    t1 = ws + (we - ws) * ((k+1) / S)
                    ink = cand[(cand["release_time_utc"] >= t0) & (cand["release_time_utc"] < t1)]
                    if not ink.empty:
                        sub_ann_flags[k] = 1
                        if ink["neg_surprise"].any():
                            sub_ann_minus_flags[k] = 1

        # VAR fit on full window
        y = sub[["mid_return_bps","ofi"]]
        p_selected, aic_val = select_var_lags(y, maxlags=min(MAX_P, max(1, n//10)))
        try:
            var_res = fit_var(y, p_selected)
        except Exception as e:
            return ("fail", (mdate, sym, wid, f"var_fit_failed:{e}"))

        # residual-based split into S substates
        resid_arr = np.asarray(var_res.resid)  # (T-p, 2)
        resid_start = p_selected
        T_res = resid_arr.shape[0]
        idx_bounds = [int(round(i * n / S)) for i in range(S+1)]
        idx_bounds = [max(0, min(n, x)) for x in idx_bounds]
        Sigma_s_list = []
        valid = True
        for s_idx in range(S):
            i0 = idx_bounds[s_idx]; i1 = idx_bounds[s_idx+1]
            # map to residual indices conservatively
            r0 = max(i0, resid_start)
            r1 = min(i1, n)
            ri0 = r0 - resid_start
            ri1 = r1 - resid_start
            if ri1 - ri0 < MIN_OBS_PER_SUBSTATE:
                valid = False
                break
            sub_res = resid_arr[ri0:ri1, :].T
            Sigma = np.cov(sub_res, bias=True)
            Sigma_s_list.append(Sigma)
        if not valid:
            # fallback: compute Sigma per-block directly from data residuals by fitting block VARs
            Sigma_s_list = []
            ok_alt = True
            for s_idx in range(S):
                i0 = idx_bounds[s_idx]; i1 = idx_bounds[s_idx+1]
                block = sub.iloc[i0:i1].reset_index(drop=True)
                if len(block) < MIN_OBS_PER_SUBSTATE:
                    ok_alt = False
                    break
                try:
                    pblk, _ = select_var_lags(block[["mid_return_bps","ofi"]], maxlags=min(MAX_P, max(1, len(block)//5)))
                    res_blk = fit_var(block[["mid_return_bps","ofi"]], pblk)
                    resid_blk = np.asarray(res_blk.resid)
                    if resid_blk.shape[0] < MIN_OBS_PER_SUBSTATE:
                        ok_alt = False
                        break
                    Sigma_s_list.append(np.cov(resid_blk.T, bias=True))
                except Exception:
                    ok_alt = False
                    break
            if not ok_alt:
                return ("fail", (mdate, sym, wid, "insufficient_substate_obs"))

        # rank condition
        if not rank_condition_ok(Sigma_s_list):
            return ("fail", (mdate, sym, wid, "rank_condition_failed"))

        # solve ITH
        sol = solve_ith(Sigma_s_list)
        if not sol["success"] or sol["x"] is None:
            return ("fail", (mdate, sym, wid, f"ith_solve_failed:{sol.get('message','no_msg')}"))

        x = sol["x"]
        br = float(x[0]); bf = float(x[1])
        omega_r = [float(v) for v in x[2:2+S]]
        omega_f = [float(v) for v in x[2+S:2+2*S]]
        B = build_B_from_params(br, bf)
        s_irf = compute_structural_irf(var_res, B, horizon=IRF_HORIZON)

        # se/tstat approx
        se_br = se_bf = t_br = t_bf = np.nan
        if sol.get("jac") is not None:
            se = jacobian_se_approx(sol["jac"])
            if se is not None and len(se) >= 2:
                se_br, se_bf = float(se[0]), float(se[1])
                t_br = float(br / se_br) if se_br>0 else np.nan
                t_bf = float(bf / se_bf) if se_bf>0 else np.nan

        # save IRF
        irf_fname = IRF_DIR / f"{mdate}_{sym}_w{wid}.npz"
        try:
            if s_irf is not None:
                np.savez_compressed(str(irf_fname),
                                    s_irf=s_irf,
                                    meta={"market_date": str(mdate), "symbol": sym, "window_id": int(wid),
                                          "br": br, "bf": bf, "omega_r": omega_r, "omega_f": omega_f,
                                          "p_selected": int(p_selected), "aic": float(aic_val) if not pd.isna(aic_val) else np.nan,
                                          "scale_ofi": float(SCALE_OFI)})
        except Exception:
            return ("fail", (mdate, sym, wid, "irf_save_failed"))

        row = {
            "market_date": str(mdate), "symbol": sym, "window_id": int(wid),
            "window_start_utc": ws_iso, "window_end_utc": we_iso,
            "n_seconds": int(n), "p_selected": int(p_selected), "aic": float(aic_val) if not pd.isna(aic_val) else np.nan,
            "br": br, "bf": bf, "se_br": se_br, "se_bf": se_bf, "t_br": t_br, "t_bf": t_bf,
            "omega_r_1": omega_r[0] if len(omega_r)>0 else np.nan,
            "omega_r_2": omega_r[1] if len(omega_r)>1 else np.nan,
            "omega_r_3": omega_r[2] if len(omega_r)>2 else np.nan,
            "omega_f_1": omega_f[0] if len(omega_f)>0 else np.nan,
            "omega_f_2": omega_f[1] if len(omega_f)>1 else np.nan,
            "omega_f_3": omega_f[2] if len(omega_f)>2 else np.nan,
            "ANN_t": int(ann_flag), "ANN_minus_t": int(ann_minus_flag),
            "ANN_sub_1": int(sub_ann_flags[0]), "ANN_sub_2": int(sub_ann_flags[1]) if S>1 else 0, "ANN_sub_3": int(sub_ann_flags[2]) if S>2 else 0,
            "ANN_minus_sub_1": int(sub_ann_minus_flags[0]), "ANN_minus_sub_2": int(sub_ann_minus_flags[1]) if S>1 else 0, "ANN_minus_sub_3": int(sub_ann_minus_flags[2]) if S>2 else 0,
            "solver_cost": sol.get("cost", np.nan), "solver_message": sol.get("message", ""), "scale_ofi": float(SCALE_OFI)
        }
        return ("ok", row)
    except Exception as e:
        tb = traceback.format_exc()
        return ("fail", (task[0], task[1], task[4], f"unexpected_error:{e}; tb:{tb}"))

# ---------- main ----------
def main():
    t0 = time.time()
    if not DESC_1S_PQ.exists():
        raise SystemExit("Missing descriptive_1s.parquet. Run previous scripts.")
    if not WINDOWS_INDEX_PQ.exists():
        raise SystemExit("Missing windows_index.parquet. Run previous scripts.")

    print("Loading descriptors (head)...")
    df1s = pd.read_parquet(DESC_1S_PQ)
    win_idx = pd.read_parquet(WINDOWS_INDEX_PQ)

    # minimal normalization
    if "ts" in df1s.columns:
        df1s["ts"] = pd.to_datetime(df1s["ts"], utc=True)
    else:
        if "ts_event" in df1s.columns:
            df1s["ts"] = pd.to_datetime(df1s["ts_event"], utc=True)
    try:
        df1s["market_date"] = pd.to_datetime(df1s["market_date"]).dt.date
    except Exception:
        df1s["market_date"] = df1s["market_date"].astype(str)

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

    news_path = find_news_path()
    if news_path is None:
        print("No news file found; ANN flags will be zero.")
    else:
        print("Found news file:", news_path)

    # prepare tasks
    win_idx_sorted = win_idx.sort_values(["market_date","symbol","window_id"])
    tasks = [(str(r["market_date"]), r["symbol"], r["window_start_utc"].isoformat(), r["window_end_utc"].isoformat(), int(r["window_id"])) for _, r in win_idx_sorted.iterrows()]
    print(f"Total windows: {len(tasks)}")

    # write small temp copies for worker to read (to avoid pickling entire DF)
    # we already have DESC_1S_PQ path; pass that + news_path to workers
    df1s_arg = str(DESC_1S_PQ)
    news_arg = str(news_path) if news_path is not None else None

    out_rows = []
    failed = []
    # start pool
    # Multiprocessing context: choose fork on *nix if available, otherwise spawn (Windows).
    try:
        # Prefer 'fork' when available (fast, avoids pickling). If not available (Windows), fall back to 'spawn'.
        if os.name == "nt":
            ctx = mp.get_context("spawn")
        else:
            try:
                ctx = mp.get_context("fork")
            except (ValueError, AttributeError):
                # fallback to default context
                ctx = mp.get_context()
    except Exception:
        # ultimate fallback
        ctx = mp.get_context()

    pool = ctx.Pool(processes=N_JOBS, initializer=worker_init, initargs=(df1s_arg, news_arg))
    try:
        it = pool.imap_unordered(process_window, tasks, chunksize=CHUNKSIZE)
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
        try:
            pool.close(); pool.join()
        except Exception:
            try:
                pool.terminate()
            except Exception:
                pass

    # write outputs
    if out_rows:
        df_out = pd.DataFrame(out_rows)
        df_out.to_parquet(OUT_SUMMARY_PQ, index=False)
        print("Wrote summary:", OUT_SUMMARY_PQ)
    else:
        print("No successful windows.")

    if failed:
        df_failed = pd.DataFrame(failed, columns=["market_date","symbol","window_id","reason"])
        df_failed.to_csv(FAILED_CSV, index=False)
        print("Wrote failures:", FAILED_CSV)
        try:
            print(df_failed["reason"].value_counts().head(10))
        except Exception:
            pass

    print("Done. Elapsed: {:.2f}s".format(time.time() - t0))

if __name__ == "__main__":
    main()
