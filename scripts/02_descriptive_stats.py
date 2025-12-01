# scripts/02_descriptive_from_raw.py
"""
Cria a base descritiva diretamente a partir dos parquets em clean_data/ (sem usar janelas).
Outputs (windows_parquet/descriptives/):
 - descriptive_1s.parquet   : série por segundo (market_date, symbol, ts, mid, mid_return_bps, ofi, n_events, avg_event_size, avg_spread, depth, is_most_active)
 - descriptive_5min.parquet : intradiário agregado em 5 minutos (std/mean as in paper)
 - descriptive_15min.parquet: intradiário agregado em 15 minutes (para referência)
 - daily_stats.csv          : estatísticas diárias (Tabela 1 style)
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------- config ----------
CLEAN_DIR = Path("clean_data")
OUT_DIR = Path("windows_parquet") / "descriptives"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_1S_PQ = OUT_DIR / "descriptive_1s.parquet"
OUT_5MIN_PQ = OUT_DIR / "descriptive_5min.parquet"
OUT_15MIN_PQ = OUT_DIR / "descriptive_15min.parquet"
OUT_DAILY_CSV = OUT_DIR / "daily_stats.csv"
DAILY_WINNERS_PQ = Path("windows_parquet/daily_most_active.parquet")  # optional existing file

MARKET_TZ = "America/Chicago"
START_TIME = "08:30:00"
END_TIME = "15:00:00"

# safety params (not strict)
MIN_EVENTS_PER_DAY = 10

# ---------- helpers ----------
def list_clean_parquets(clean_dir: Path):
    return sorted(clean_dir.glob("*.parquet"))

def detect_winners_from_clean(clean_dir: Path):
    """Count events by (market_date, symbol) across clean parquets and return winners_map."""
    files = list_clean_parquets(clean_dir)
    volume = defaultdict(int)
    for p in files:
        try:
            # read minimal columns (speed)
            df = pd.read_parquet(p, columns=["ts_event", "symbol"])
        except Exception:
            try:
                df = pd.read_parquet(p)
            except Exception as e:
                print(f"  failed reading {p}: {e}")
                continue
        if df.empty:
            continue
        # normalize timestamp to UTC
        if "ts_event" in df.columns:
            df["ts"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        elif "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        else:
            continue
        df = df.dropna(subset=["ts"])
        if df.empty or "symbol" not in df.columns:
            continue
        # market_date in market tz
        df["market_date"] = df["ts"].dt.tz_convert(MARKET_TZ).dt.date
        counts = df.groupby(["market_date", "symbol"]).size().reset_index(name="n")
        for _, row in counts.iterrows():
            volume[(row["market_date"], row["symbol"])] += int(row["n"])
    winners = {}
    for (d, s), vol in volume.items():
        if d not in winners or vol > volume[(d, winners[d])]:
            winners[d] = s
    print(f"Detected winners for {len(winners)} market_dates.")
    return winners

def load_or_compute_winners():
    if DAILY_WINNERS_PQ.exists():
        try:
            dfw = pd.read_parquet(DAILY_WINNERS_PQ)
            winners = {}
            for _, r in dfw.iterrows():
                try:
                    d = pd.to_datetime(r["market_date"]).date()
                except Exception:
                    d = pd.to_datetime(str(r["market_date"])).date()
                winners[d] = r.get("most_active_contract") or r.get("most_active") or r.get("symbol")
            print(f"Loaded winners_map from {DAILY_WINNERS_PQ}")
            return winners
        except Exception as e:
            print("Failed to load daily_most_active.parquet:", e)
    return detect_winners_from_clean(CLEAN_DIR)

def time_window_for_date(date):
    """Return (start_utc, end_utc) for market_date in UTC given MARKET_TZ, START_TIME, END_TIME."""
    day_start_market = pd.Timestamp(f"{date} {START_TIME}", tz=MARKET_TZ)
    day_end_market = pd.Timestamp(f"{date} {END_TIME}", tz=MARKET_TZ)
    return day_start_market.tz_convert("UTC"), day_end_market.tz_convert("UTC")

# ---------- core processing ----------
def process_file_to_1s(df_raw, winners_map, writer_holder):
    """
    df_raw: raw parquet DataFrame for a single contract file (may contain multiple days).
    winners_map: map date->winning_symbol (only process days where this file's symbol is the winner)
    writer_holder: dict to hold ParquetWriter to stream-write the 1s table
    Returns: list of per-day stats to be combined later
    """
    # standardize ts
    if "ts" not in df_raw.columns:
        if "ts_event" in df_raw.columns:
            df_raw["ts"] = pd.to_datetime(df_raw["ts_event"], utc=True, errors="coerce")
        else:
            raise ValueError("Input parquet has no ts/ts_event column")
    else:
        df_raw["ts"] = pd.to_datetime(df_raw["ts"], utc=True, errors="coerce")

    df_raw = df_raw.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    if df_raw.empty:
        return []

    # determine file symbol
    if "symbol" not in df_raw.columns:
        raise ValueError("Input parquet missing 'symbol' column")
    file_symbol = df_raw["symbol"].mode().iloc[0] if not df_raw["symbol"].mode().empty else None

    # numeric enforcement
    for col in ["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
        else:
            df_raw[col] = np.nan

    # create event-level mid/spread
    df_raw["mid"] = (df_raw["bid_px_00"] + df_raw["ask_px_00"]) / 2.0
    df_raw["spread"] = df_raw["ask_px_00"] - df_raw["bid_px_00"]

    # aliases for OFI (use best-level)
    df_raw["Pb"] = df_raw["bid_px_00"]
    df_raw["Pa"] = df_raw["ask_px_00"]
    df_raw["qb"] = df_raw["bid_sz_00"]
    df_raw["qa"] = df_raw["ask_sz_00"]

    # ensure market_date column (in market tz)
    df_raw["market_date"] = df_raw["ts"].dt.tz_convert(MARKET_TZ).dt.date

    # group by market_date and symbol to compute lags used in OFI
    out_daily_stats = []
    grouped_days = df_raw.groupby(["market_date", "symbol"], sort=True)
    for (mdate, sym), grp in grouped_days:
        print(f"    -> Processando dia {mdate} ({sym})")
        # process only if this file is the winner for that day (if winners_map has an entry)
        winner = winners_map.get(mdate)
        if winner is not None and winner != file_symbol:
            continue

        # restrict to official trading hours (UTC)
        day_start_utc, day_end_utc = time_window_for_date(mdate)
        grp_day = grp[(grp["ts"] >= day_start_utc) & (grp["ts"] < day_end_utc)].copy()
        if grp_day.empty or len(grp_day) < MIN_EVENTS_PER_DAY:
            continue

        # compute lagged book state (shift in chronological order within group)
        grp_day = grp_day.sort_values("ts").reset_index(drop=True)
        grp_day["Pb_lag"] = grp_day["Pb"].shift(1).fillna(method="ffill").fillna(0)
        grp_day["Pa_lag"] = grp_day["Pa"].shift(1).fillna(method="ffill").fillna(0)
        grp_day["qb_lag"] = grp_day["qb"].shift(1).fillna(0)
        grp_day["qa_lag"] = grp_day["qa"].shift(1).fillna(0)

        # event-level OFI as in Cont et al. (strict comparisons)
        # en = qb * 1{Pb > Pb_lag} - qb_lag * 1{Pb < Pb_lag} - qa * 1{Pa < Pa_lag} + qa_lag * 1{Pa > Pa_lag}
        # uses strict > / < to detect price moves
        pb_up = (grp_day["Pb"] > grp_day["Pb_lag"]).astype(int)
        pb_down = (grp_day["Pb"] < grp_day["Pb_lag"]).astype(int)
        pa_up = (grp_day["Pa"] > grp_day["Pa_lag"]).astype(int)
        pa_down = (grp_day["Pa"] < grp_day["Pa_lag"]).astype(int)

        qb = grp_day["qb"].fillna(0)
        qa = grp_day["qa"].fillna(0)
        qb_lag = grp_day["qb_lag"].fillna(0)
        qa_lag = grp_day["qa_lag"].fillna(0)

        grp_day["en"] = (
            qb * pb_up
            - qb_lag * pb_down
            - qa * pa_down
            + qa_lag * pa_up
        )

        # floor to second
        grp_day["ts_1s"] = grp_day["ts"].dt.floor("s")

        # aggregate per second
        agg = grp_day.groupby("ts_1s", sort=True).agg(
            n_events = ("ts", "count"),
            ofi = ("en", "sum"),
            avg_event_size = ("qb", lambda x: float(x.dropna().mean()) if len(x.dropna())>0 else np.nan),
            mid = ("mid", "mean"),
            avg_spread = ("spread", "mean"),
            depth_qb = ("qb", lambda x: float(x.dropna().mean()) if len(x.dropna())>0 else np.nan),
            depth_qa = ("qa", lambda x: float(x.dropna().mean()) if len(x.dropna())>0 else np.nan)
        ).reset_index().rename(columns={"ts_1s": "ts"})

        # depth: mean((qb+qa)/2) per second
        agg["depth"] = agg[["depth_qb","depth_qa"]].mean(axis=1)
        agg = agg.drop(columns=["depth_qb","depth_qa"])

        # mid_return in bps, safe (avoid division by zero)
        agg = agg.sort_values("ts").reset_index(drop=True)
        agg["mid_prev"] = agg["mid"].shift(1)
        # avoid divide by zero: where mid_prev is null or zero, produce NaN
        agg["mid_return_bps"] = np.where(
            (agg["mid_prev"].notna()) & (agg["mid_prev"] != 0),
            (agg["mid"] - agg["mid_prev"]) / agg["mid_prev"] * 10000.0,
            np.nan
        )

        # keep meta columns
        agg["market_date"] = mdate
        agg["symbol"] = sym
        # is_most_active flag (true if winners_map says this symbol won)
        agg["is_most_active"] = bool(winner == sym)

        # reorder columns in output
        out_cols = ["market_date","symbol","ts","mid","mid_return_bps","ofi","n_events","avg_event_size","avg_spread","depth","is_most_active"]
        for c in out_cols:
            if c not in agg.columns:
                agg[c] = pd.NA
        agg = agg[out_cols]

        # write to parquet streaming (first+append via pyarrow writer)
        table = pa.Table.from_pandas(agg, preserve_index=False)
        # init writer if needed
        if writer_holder.get("writer") is None:
            writer_holder["schema"] = table.schema
            writer_holder["writer"] = pq.ParquetWriter(str(OUT_1S_PQ), table.schema, compression="SNAPPY")
        else:
            # reconcile schema differences if any
            if not table.schema.equals(writer_holder["schema"]):
                df_tmp = table.to_pandas()
                for f in writer_holder["schema"]:
                    if f.name not in df_tmp.columns:
                        df_tmp[f.name] = pd.NA
                df_tmp = df_tmp[[f.name for f in writer_holder["schema"]]]
                table = pa.Table.from_pandas(df_tmp, schema=writer_holder["schema"], preserve_index=False)
        writer_holder["writer"].write_table(table)

        # compute daily simple stats from agg to report later
        day_stats = {
            "market_date": str(mdate),
            "symbol": sym,
            "n_seconds": int(len(agg)),
            "n_events_total": int(grp_day.shape[0]),
            "mean_mid": float(agg["mid"].mean()) if agg["mid"].notna().any() else np.nan,
            "std_mid_return_bps": float(agg["mid_return_bps"].std(skipna=True)),
            "std_ofi": float(agg["ofi"].std(skipna=True)),
            "mean_n_events_per_second": float(agg["n_events"].mean())
        }
        out_daily_stats.append(day_stats)

    return out_daily_stats

# ---------- entrypoint ----------
def main():
    parquet_files = list_clean_parquets(CLEAN_DIR)
    if not parquet_files:
        raise SystemExit("No parquet files found in clean_data/. Abort.")

    winners_map = load_or_compute_winners()

    # prepare writer holder
    writer_holder = {"writer": None, "schema": None}
    all_daily_stats = []

    # iterate files (stream per-file)
    for p in parquet_files:
        print("Processing file:", p.name)
        try:
            df_raw = pd.read_parquet(p)
        except Exception as e:
            print("  failed to read:", e)
            continue
        try:
            daily_stats = process_file_to_1s(df_raw, winners_map, writer_holder)
            all_daily_stats.extend(daily_stats)
        except Exception as e:
            print("  error processing file:", e)
            continue

    # close writer
    if writer_holder.get("writer") is not None:
        writer_holder["writer"].close()
        print("Wrote 1-second parquet to:", OUT_1S_PQ)
    else:
        print("No data written to 1s parquet. Exiting.")
        return

    # ---- post-processing: read the descriptive_1s and compute intraday aggregates ----
    print("Reading back 1s parquet for intraday aggregation (may use memory).")
    df1s = pd.read_parquet(OUT_1S_PQ)
    # ensure ts is timezone-aware UTC
    df1s["ts"] = pd.to_datetime(df1s["ts"], utc=True)
    # sort
    df1s = df1s.sort_values(["market_date","symbol","ts"]).reset_index(drop=True)

    # compute 5min and 15min intraday series per market_date+symbol
    intraday_5min_rows = []
    intraday_15min_rows = []
    grouped = df1s.groupby(["market_date","symbol"], sort=True)
    for (mdate, sym), g in grouped:
        if g.empty:
            continue
        g = g.set_index("ts")
        # resample 5T
        r5 = g.resample("5T").agg({
            "mid_return_bps": "std",
            "ofi": "std",
            "n_events": "mean",
            "avg_event_size": "mean",
            "avg_spread": "mean",
            "depth": "mean"
        }).rename(columns={
            "mid_return_bps":"std_mid_return_bps",
            "ofi":"std_ofi",
            "n_events":"mean_n_events",
            "avg_event_size":"mean_event_size",
            "avg_spread":"mean_spread",
            "depth":"mean_depth"
        }).reset_index()
        r5["market_date"] = mdate
        r5["symbol"] = sym
        intraday_5min_rows.append(r5)

        # resample 15T
        r15 = g.resample("15T").agg({
            "mid_return_bps": "std",
            "ofi": "std",
            "n_events": "mean",
            "avg_event_size": "mean",
            "avg_spread": "mean",
            "depth": "mean"
        }).rename(columns={
            "mid_return_bps":"std_mid_return_bps",
            "ofi":"std_ofi",
            "n_events":"mean_n_events",
            "avg_event_size":"mean_event_size",
            "avg_spread":"mean_spread",
            "depth":"mean_depth"
        }).reset_index()
        r15["market_date"] = mdate
        r15["symbol"] = sym
        intraday_15min_rows.append(r15)

    if intraday_5min_rows:
        df5 = pd.concat(intraday_5min_rows, ignore_index=True)
        df5.to_parquet(OUT_5MIN_PQ, index=False)
        print("Wrote 5-min intraday parquet:", OUT_5MIN_PQ)
    else:
        print("No 5-min rows generated.")

    if intraday_15min_rows:
        df15 = pd.concat(intraday_15min_rows, ignore_index=True)
        df15.to_parquet(OUT_15MIN_PQ, index=False)
        print("Wrote 15-min intraday parquet:", OUT_15MIN_PQ)
    else:
        print("No 15-min rows generated.")

    # write daily stats summary
    if all_daily_stats:
        df_daily = pd.DataFrame(all_daily_stats)
        df_daily.to_csv(OUT_DAILY_CSV, index=False)
        print("Wrote daily stats CSV:", OUT_DAILY_CSV)
    else:
        print("No daily stats computed.")

if __name__ == "__main__":
    main()
