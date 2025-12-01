#!/usr/bin/env python3
import re
import pandas as pd
from datetime import datetime
import pytz
from pathlib import Path

# --- Config ---
base_path = Path(__file__).parent
input_txt = base_path / "calendar.txt"
out_full = base_path / "macro_announcements_2022_final.csv"
out_minimal = base_path / "macro_announcements_2022_minimal.csv"
market_tz = "America/Chicago"

# Indicators to look for
indicators = [
    "Construction Spending",
    "Consumer Confidence",
    "Existing Home Sales",
    "Factory Orders",
    "ISM Manufacturing",
    "ISM Non-Manufacturing",   # ISM Services
    "Leading Indicators",
    "Wholesale Inventories"
]

# Regex patterns
date_header_re = re.compile(
    r"^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+([A-Za-z]+ \d{1,2}, \d{4})$",
    re.IGNORECASE,
)
event_re = re.compile(
    r"""^(?:(\d{1,2}:\d{2})\s+)?   # optional time e.g. 08:30
       [A-Z]{3}\s+                # currency code like USD
       (.+?)\s+                   # event name (lazy)
       ((?:-?\d+(?:\.\d+)?%?)\s+)?  # optional actual
       ((?:-?\d+(?:\.\d+)?%?)\s+)?  # optional forecast
       ((?:-?\d+(?:\.\d+)?%?)\s*)?$ # optional previous
    """,
    re.VERBOSE,
)

# helper: parse numeric-ish strings into floats (handles %, K, M)
def parse_num(s):
    if not s or pd.isna(s):
        return None
    s = str(s).strip().replace(",", "")
    # handle percent sign
    is_pct = s.endswith("%")
    if is_pct:
        s = s[:-1]
    # handle K / M suffix
    mult = 1.0
    if s.endswith("K") or s.endswith("M"):
        if s.endswith("K"):
            mult = 1_000.0
        else:
            mult = 1_000_000.0
        s = s[:-1]
    try:
        val = float(s)
        if is_pct:
            # keep percent numeric (as raw percent, not divided by 100) so comparisons still meaningful
            return val
        return val * mult
    except Exception:
        return None

# Read input
if not input_txt.exists():
    print(f"ERROR: {input_txt} not found. Coloque calendar.txt na mesma pasta que este script.")
    raise SystemExit(1)

txt = input_txt.read_text(encoding="utf-8")
lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]

rows = []
current_date = None

for ln in lines:
    # update day context
    mdate = date_header_re.match(ln)
    if mdate:
        try:
            current_date = datetime.strptime(mdate.group(1), "%B %d, %Y").date()
        except Exception:
            current_date = None
        continue

    # try to parse event-like line
    m = event_re.match(ln)
    if not m:
        # fallback split by 2+ spaces (some lines are like the pasted calendar)
        parts = re.split(r"\s{2,}", ln)
        if len(parts) >= 3 and re.match(r"\d{1,2}:\d{2}", parts[0]):
            timepart = parts[0].strip()
            evt_name = parts[2].strip()
            rest = parts[3:] if len(parts) > 3 else []
            actual = rest[0].strip() if len(rest) >= 1 else None
            forecast = rest[1].strip() if len(rest) >= 2 else None
        else:
            continue
    else:
        timepart = m.group(1)  # may be None
        evt_raw = m.group(2).strip()
        actual = (m.group(3) or "").strip() or None
        forecast = (m.group(4) or "").strip() or None
        evt_name = re.sub(r"\s+", " ", evt_raw).strip()

    # if no current_date from header, try extract date in-line
    if current_date is None:
        m2 = re.search(r"([A-Za-z]+ \d{1,2}, \d{4})", ln)
        if m2:
            try:
                current_date = datetime.strptime(m2.group(1), "%B %d, %Y").date()
            except:
                current_date = None

    # match event name to one of the indicators
    evt_key = None
    for ind in indicators:
        # broad matching: substring match or ISM special case
        if ind.lower() in evt_name.lower() or (
            ind == "ISM Manufacturing" and "ISM Manufacturing" in evt_name
        ) or (
            ind == "ISM Non-Manufacturing"
            and ("Non-Manufacturing" in evt_name or "Services" in evt_name)
        ):
            evt_key = ind
            break

    if evt_key and current_date:
        # construct datetime (default 09:00 if missing)
        time_str = timepart or "09:00"
        try:
            dt_local = datetime.strptime(f"{current_date} {time_str}", "%Y-%m-%d %H:%M")
        except Exception:
            # if parsing fails, skip
            continue
        tz = pytz.timezone(market_tz)
        dt_local = tz.localize(dt_local)

        # numeric parse
        a = parse_num(actual)
        fct = parse_num(forecast)

        # decide negative:
        # - if both numeric: negative = 1 if actual < forecast else 0
        # - if not numeric (e.g. missing forecast/actual): default negative = 0 (user asked to mark non-negatives as 0)
        if a is not None and fct is not None:
            negative = 1 if a < fct else 0
        else:
            negative = 0

        rows.append(
            {
                "indicator": evt_key,
                "release_datetime_CT": dt_local.isoformat(),
                "actual_raw": actual or "",
                "forecast_raw": forecast or "",
                "negative": int(negative),
                "source_line": ln,
            }
        )

# Create DataFrame
df = pd.DataFrame(rows)

if df.empty:
    print("Nenhum indicador encontrado. Verifique calendar.txt.")
    raise SystemExit(0)

# drop duplicates (Indicator + datetime) and sort
df = df.drop_duplicates(subset=["indicator", "release_datetime_CT"])
df = df.sort_values(["indicator", "release_datetime_CT"]).reset_index(drop=True)

# save full CSV
df.to_csv(out_full, index=False)

# save minimal CSV
df_min = df[["indicator", "release_datetime_CT", "negative"]].copy()
df_min.to_csv(out_minimal, index=False)

# Summary counts
total_events = len(df_min)
negatives = int(df_min["negative"].sum())
non_negatives = total_events - negatives

# counts per indicator
counts_by_indicator = df_min.groupby("indicator")["negative"].agg(["count", "sum"])
counts_by_indicator = counts_by_indicator.rename(columns={"count": "total", "sum": "n_negative"}).reset_index()

# Print summary
print("✅ Extração concluída.")
print(f"Arquivo completo salvo em: {out_full}")
print(f"Arquivo minimal salvo em: {out_minimal}")
print()
print(f"Resumo geral:")
print(f"  Total de eventos extraídos: {total_events}")
print(f"  Eventos com negative == 1 (actual < forecast): {negatives}")
print(f"  Eventos com negative == 0 (non-negative / default): {non_negatives}")
print()
print("Contagem por indicador:")
print(counts_by_indicator.to_string(index=False))

# Also print a quick sample of the first few rows
print("\nAmostra (primeiras 10 linhas):")
print(df_min.head(10).to_string(index=False))
