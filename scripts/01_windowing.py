# scripts/01_prepare_and_window_final.py
"""
Final pipeline (versão ajustada):
 - detect winners per market_date across all clean_data/*.parquet
 - build 15-min windows per file (only for days where this file's symbol is the day's winner)
 - normalize schema for all windows
 - stream-write windows events into a single parquet (with harmonização de schema)
 - write a small lookup daily_most_active.parquet and windows_index.parquet and summary_by_day.csv
 - remove days with fewer than 26 windows from both index and final parquet (post-filter)
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CLEAN_DIR = "clean_data"
OUT_DIR = Path("windows_parquet")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_PQ = OUT_DIR / "windows_all_days.parquet"
DAILY_WINNERS_PQ = OUT_DIR / "daily_most_active.parquet"
SUMMARY_CSV = OUT_DIR / "summary_by_day.csv"
WINDOWS_INDEX_PQ = OUT_DIR / "windows_index.parquet"

MARKET_TZ = "America/Chicago"
WINDOW_MIN = 15
START_TIME = "08:30:00"
END_TIME = "15:00:00"
S = 3  # substates, stored if needed (we keep raw rows; substates not written)

# thresholds
MIN_EVENTS_PER_WINDOW = 10      # mínimo de eventos para persistir a janela
MIN_SECONDS_WITH_DATA = 30      # pelo menos N segundos com dados na janela
EXPECTED_WINDOWS_PER_DAY = 26   # 6h30 / 15min

# ---------- helpers ----------
def _parquet_files_in_dir(clean_dir: str) -> List[str]:
    return [os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith(".parquet")]

def _detect_winners_across_files(clean_dir: str = CLEAN_DIR, market_tz: str = MARKET_TZ) -> Tuple[Dict, Dict]:
    """
    Conta eventos por (market_date, symbol) agregando sobre todos os parquets
    e devolve um mapa winners[date] = symbol mais ativo nesse dia.
    """
    parquet_files = [f for f in os.listdir(clean_dir) if f.endswith(".parquet")]
    volume = defaultdict(int)
    for fname in parquet_files:
        path = os.path.join(clean_dir, fname)
        try:
            # ler apenas colunas mínimas se existirem (mais rápido)
            df = pd.read_parquet(path, columns=["ts_event", "symbol"])
        except Exception:
            df = pd.read_parquet(path)
        if df.empty:
            continue
        # normalize timestamp to UTC
        if "ts_event" in df.columns:
            df["ts"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        else:
            # fallback: try ts column
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            else:
                continue
        df = df.dropna(subset=["ts"])
        if df.empty:
            continue
        df["market_date"] = df["ts"].dt.tz_convert(market_tz).dt.date
        # If symbol column missing, skip file
        if "symbol" not in df.columns:
            continue
        counts = df.groupby(["market_date", "symbol"]).size().reset_index(name="n")
        for _, row in counts.iterrows():
            d = row["market_date"]
            s = row["symbol"]
            volume[(d, s)] += int(row["n"])
    winners = {}
    stats_days_per_symbol = defaultdict(int)
    for (d, s), vol in volume.items():
        if d not in winners or vol > volume[(d, winners[d])]:
            winners[d] = s
    for d, s in winners.items():
        stats_days_per_symbol[s] += 1
    return winners, dict(stats_days_per_symbol)

def _split_into_substates_by_utc(win_df: pd.DataFrame, win_start_utc, win_end_utc, S=3):
    total = (win_end_utc - win_start_utc) / S
    substates = []
    for i in range(S):
        s0 = win_start_utc + i * total
        s1 = win_start_utc + (i + 1) * total
        mask = (win_df["ts"] >= s0) & (win_df["ts"] < s1)
        substates.append(win_df.loc[mask])
    return substates

def _infer_event_columns_from_parquet(parquet_path: str) -> List[str]:
    """
    Usa pyarrow para ler o schema rapidamente e inferir 'event columns'
    (exclui colunas meta reservadas).
    """
    pf = pq.ParquetFile(parquet_path)
    names = [f.name for f in pf.schema.to_arrow_schema()]
    reserved = {"ts", "market_date", "window_start_utc", "window_end_utc",
                "symbol", "window_id", "is_most_active"}
    event_columns = [c for c in names if c not in reserved]
    # ensure ts_event first if present
    if "ts_event" in event_columns:
        event_columns = ["ts_event"] + [c for c in event_columns if c != "ts_event"]
    return event_columns

def prepare_and_window(parquet_path: str,
                       market_tz: str = MARKET_TZ,
                       start_time: str = START_TIME,
                       end_time: str = END_TIME,
                       window_min: int = WINDOW_MIN,
                       winners_map: Dict = None,
                       symbol_override: str = None) -> List[Dict[str, Any]]:
    """
    Lê um parquet (pode conter múltiplos símbolos), detecta símbolo mais comum no arquivo,
    e cria janelas de 15-min dentro do intervalo de trading entre start_time e end_time.
    Retorna lista de dicionários com meta + dataframe por janela (raw events).
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(parquet_path)
    df = pd.read_parquet(parquet_path)
    if "ts_event" not in df.columns and "ts" not in df.columns:
        raise ValueError("Arquivo precisa conter a coluna 'ts_event' ou 'ts'")
    df = df.copy()
    # prefer ts_event -> ts; parse to UTC
    if "ts_event" in df.columns:
        df["ts"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    else:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    if df.empty:
        return []
    # determine file symbol: choose most frequent symbol in file (more robust than first row)
    file_symbol = symbol_override
    if file_symbol is None and "symbol" in df.columns:
        file_symbol = df["symbol"].mode().iloc[0] if not df["symbol"].mode().empty else None

    if winners_map is None:
        # local detection fallback (per day choose most active symbol from this file)
        local_map = {}
        df_local = df.copy()
        df_local["market_date"] = df_local["ts"].dt.tz_convert(market_tz).dt.date
        counts = df_local.groupby(["market_date", "symbol"]).size().reset_index(name="n")
        for date, group in counts.groupby("market_date"):
            best = group.sort_values("n", ascending=False).iloc[0]
            local_map[date] = best["symbol"]
        winners_map = local_map

    min_ts = df["ts"].min()
    max_ts = df["ts"].max()
    min_market_date = min_ts.tz_convert(market_tz).date()
    max_market_date = max_ts.tz_convert(market_tz).date()
    results = []
    delta = pd.Timedelta(minutes=window_min)
    current_date = min_market_date
    while current_date <= max_market_date:
        # filter by global winners_map: only generate windows for this file if it's the winner that day
        winner_for_date = winners_map.get(current_date)
        if winner_for_date is not None and file_symbol is not None:
            if winner_for_date != file_symbol:
                current_date = (pd.Timestamp(current_date) + pd.Timedelta(days=1)).date()
                continue
        day_start_market = pd.Timestamp(f"{current_date} {start_time}", tz=market_tz)
        day_end_market = pd.Timestamp(f"{current_date} {end_time}", tz=market_tz)
        day_start_utc = day_start_market.tz_convert("UTC")
        day_end_utc = day_end_market.tz_convert("UTC")
        mask_day = (df["ts"] >= day_start_utc) & (df["ts"] < day_end_utc)
        df_day = df.loc[mask_day]
        if not df_day.empty:
            win_start = day_start_utc
            win_id = 0
            while win_start < day_end_utc:
                win_end = win_start + delta
                mask_win = (df_day["ts"] >= win_start) & (df_day["ts"] < win_end)
                win_df = df_day.loc[mask_win]
                if not win_df.empty:
                    # quick quality checks for minimal data
                    n_events = len(win_df)
                    n_seconds = win_df['ts'].dt.floor('S').nunique()
                    if n_events >= MIN_EVENTS_PER_WINDOW and n_seconds >= MIN_SECONDS_WITH_DATA:
                        results.append({
                            "market_date": current_date,
                            "symbol": file_symbol,
                            "window_start_utc": win_start,
                            "window_end_utc": win_end,
                            "window_df": win_df.reset_index(drop=True),
                            "window_id": int(win_id),
                            "is_most_active": winners_map.get(current_date) == file_symbol,
                            "n_events": int(n_events),
                            "n_seconds": int(n_seconds)
                        })
                win_start = win_end
                win_id += 1
        current_date = (pd.Timestamp(current_date) + pd.Timedelta(days=1)).date()
    return results

def _normalize_window_df(win_df: pd.DataFrame, meta: Dict[str, Any], out_event_columns: List[str]) -> pd.DataFrame:
    """
    Ensure fixed columns and types. Return a DataFrame ready for conversion to pyarrow Table.
    - market_date: YYYY-MM-DD str
    - window_start_utc, window_end_utc: ISO str with timezone
    - symbol: str
    - window_id: int
    - is_most_active: bool
    - keep event cols (ts_event, bid_px_00, ...) preserving their names (fill missing)
    """
    df = win_df.copy()
    # ensure event columns exist
    for c in out_event_columns:
        if c not in df.columns:
            df[c] = pd.NA
    # preserve original ts_event as string if present
    if "ts_event" in df.columns:
        df["ts_event"] = df["ts_event"].astype(str)
    # meta columns
    df["market_date"] = pd.Timestamp(meta["market_date"]).strftime("%Y-%m-%d")
    # store ISO strings with timezone for stability
    df["window_start_utc"] = pd.Timestamp(meta["window_start_utc"]).tz_convert("UTC").isoformat()
    df["window_end_utc"] = pd.Timestamp(meta["window_end_utc"]).tz_convert("UTC").isoformat()
    df["symbol"] = str(meta["symbol"])
    df["window_id"] = int(meta["window_id"])
    df["is_most_active"] = bool(meta.get("is_most_active", False))
    # reorder columns: event columns first (in the same order), then the meta columns we define
    out_cols = out_event_columns + ["market_date", "window_start_utc", "window_end_utc",
                                    "symbol", "window_id", "is_most_active"]
    # if any out_cols missing due to unexpected schema issues, add them as NA
    for c in out_cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[out_cols]
    return df

# ---------- main flow ----------
def main():
    print(">>> Detectando contrato mais ativo por dia (todos parquets)...")
    winners_map, days_per_symbol = _detect_winners_across_files(CLEAN_DIR, MARKET_TZ)
    # save lookup
    dfw = pd.DataFrame([{"market_date": d.strftime("%Y-%m-%d"), "most_active_contract": s}
                        for d, s in sorted(winners_map.items())])
    if not dfw.empty:
        dfw.to_parquet(DAILY_WINNERS_PQ, index=False)
        print(f"Saved daily winners to {DAILY_WINNERS_PQ}")
    print(">>> Dias vencidos por contrato (resumo):")
    for s, cnt in sorted(days_per_symbol.items()):
        print(f"  {s}: {cnt} dias")
    print(f"\nTotal market_dates: {len(winners_map)}\n")

    parquet_files = _parquet_files_in_dir(CLEAN_DIR)
    if not parquet_files:
        print("Nenhum parquet em clean_data/ encontrado. Abortando.")
        return

    # infer event columns from first parquet (pyarrow, cheap)
    event_columns = _infer_event_columns_from_parquet(parquet_files[0])
    print(">>> Event columns que serão persistidas (em ordem):", event_columns)

    writer = None
    writer_schema = None
    n_windows_total = 0
    n_windows_written = 0
    windows_written_per_file = defaultdict(int)
    meta_index_rows = []

    print(">>> Gerando janelas e escrevendo em", FINAL_PQ)
    try:
        for p in parquet_files:
            fname = os.path.basename(p)
            print(f"\nProcessando arquivo: {fname}")
            try:
                win_list = prepare_and_window(p, MARKET_TZ, START_TIME, END_TIME, WINDOW_MIN, winners_map)
            except Exception as e:
                print("  Erro ao gerar janelas:", e)
                continue
            if not win_list:
                print("  Nenhuma janela válida.")
                continue
            n_windows_total += len(win_list)
            # process windows sequentially and write with ParquetWriter
            for w in win_list:
                meta = {
                    "market_date": w["market_date"],
                    "window_start_utc": w["window_start_utc"],
                    "window_end_utc": w["window_end_utc"],
                    "symbol": w["symbol"],
                    "window_id": w.get("window_id", 0),
                    "is_most_active": w.get("is_most_active", False)
                }
                try:
                    out_df = _normalize_window_df(w["window_df"], meta, event_columns)
                    # convert to pyarrow Table
                    # ensure deterministic dtypes by letting pyarrow infer, but we will harmonize to writer.schema if needed
                    table = pa.Table.from_pandas(out_df, preserve_index=False)
                    if writer is None:
                        # create writer with this schema
                        writer = pq.ParquetWriter(str(FINAL_PQ), table.schema, compression="SNAPPY")
                        writer_schema = writer.schema
                    else:
                        # if schemas differ, attempt to harmonize by ensuring writer_schema columns exist in table
                        if not table.schema.equals(writer_schema):
                            # convert table to pandas, add missing cols (NA), reorder to target, then rebuild table
                            df_cast = table.to_pandas()
                            for f in writer_schema:
                                if f.name not in df_cast.columns:
                                    df_cast[f.name] = pd.NA
                            # ensure column order as writer_schema
                            df_cast = df_cast[[f.name for f in writer_schema]]
                            # build new table using writer_schema (pyarrow will attempt casts)
                            try:
                                table = pa.Table.from_pandas(df_cast, schema=writer_schema, preserve_index=False)
                            except Exception as ex:
                                # as fallback, coerce without schema and log + skip this window
                                print(f"  Schema mismatch irreconciliável para janela {meta}, pulando janela. Erro: {ex}")
                                continue
                    writer.write_table(table)
                    n_windows_written += 1
                    windows_written_per_file[fname] += 1
                    # append to index
                    meta_index_rows.append({
                        "market_date": meta["market_date"],
                        "symbol": meta["symbol"],
                        "window_id": meta["window_id"],
                        "window_start_utc": pd.Timestamp(meta["window_start_utc"]).isoformat(),
                        "window_end_utc": pd.Timestamp(meta["window_end_utc"]).isoformat(),
                        "is_most_active": meta["is_most_active"],
                        "n_events": int(w.get("n_events", -1)),
                        "n_seconds": int(w.get("n_seconds", -1)),
                        "source_file": fname
                    })
                except Exception as e:
                    print(f"  Erro ao escrever janela {meta}: {e}")
            print(f"  Processadas {len(win_list)} janelas do arquivo — gravadas {windows_written_per_file[fname]}.")
    finally:
        if writer is not None:
            writer.close()

    print("\n================ RESUMO FINAL ================")
    print(f"Arquivos processados: {len(parquet_files)}")
    print(f"Janelas detectadas (total): {n_windows_total}")
    print(f"Janelas escritas no {FINAL_PQ.name}: {n_windows_written}")
    print("==============================================\n")

    # write windows_index.parquet with meta info
    if meta_index_rows:
        df_index = pd.DataFrame(meta_index_rows)
        df_index.to_parquet(WINDOWS_INDEX_PQ, index=False)
        print(f"Index de janelas salvo em: {WINDOWS_INDEX_PQ}")

        # now drop days with fewer than EXPECTED_WINDOWS_PER_DAY windows
        per_day = df_index.groupby("market_date").agg(n_windows=("window_id", "nunique")).reset_index()
        short_days = per_day[per_day["n_windows"] < EXPECTED_WINDOWS_PER_DAY]["market_date"].tolist()
        if short_days:
            print(f"\nRemovendo {len(short_days)} dia(s) com menos de {EXPECTED_WINDOWS_PER_DAY} janelas:"
                  f" {short_days[:10]}{'...' if len(short_days) > 10 else ''}")
            # filter index to keep only good days
            df_index_filtered = df_index[~df_index["market_date"].isin(short_days)].copy()
            # overwrite windows_index.parquet
            df_index_filtered.to_parquet(WINDOWS_INDEX_PQ, index=False)
            print(f"Index atualizado salvo em: {WINDOWS_INDEX_PQ} (dias removidos: {len(short_days)})")

            # rewrite FINAL_PQ removing rows with these market_dates
            if FINAL_PQ.exists():
                try:
                    print("Reescrevendo arquivo final (removendo janelas de dias curtos). Isso pode levar algum tempo...")
                    # read full parquet into pandas (warning: memory intensive for very large datasets)
                    df_full = pd.read_parquet(FINAL_PQ)
                    before_rows = len(df_full)
                    df_full = df_full[~df_full["market_date"].isin(short_days)].reset_index(drop=True)
                    after_rows = len(df_full)
                    df_full.to_parquet(FINAL_PQ, index=False)
                    removed_rows = before_rows - after_rows
                    print(f"Arquivo {FINAL_PQ.name} reescrito. Linhas removidas: {removed_rows}")
                    # update summary_by_day from filtered index
                    summary = df_index_filtered.groupby("market_date").agg(
                        symbol=("symbol", "first"),
                        n_windows=("window_id", "nunique")
                    ).reset_index()
                    summary.to_csv(SUMMARY_CSV, index=False)
                    print(f"Resumo por dia (após filtro) salvo em: {SUMMARY_CSV}")
                except Exception as e:
                    print("Falha ao reescrever FINAL_PQ removendo dias curtos:", e)
                    print("O index foi atualizado, mas o arquivo windows_all_days.parquet permaneceu inalterado.")
        else:
            # no short days -> normal summary
            try:
                summary = df_index.groupby("market_date").agg(
                    symbol=("symbol", "first"),
                    n_windows=("window_id", "nunique")
                ).reset_index()
                summary.to_csv(SUMMARY_CSV, index=False)
                print(f"Resumo por dia salvo em: {SUMMARY_CSV}")
            except Exception as e:
                print("Falha ao gerar summary_by_day.csv:", e)

if __name__ == "__main__":
    main()
