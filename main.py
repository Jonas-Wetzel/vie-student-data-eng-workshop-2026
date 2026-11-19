"""Load raw 1-minute CSVs from data/raw into SQLite and expose a clean 1m bar view.

Tables / views:
  raw_market_data  every raw row as received (plus normalisation + validity flag)
  bars_1m          one row per (ticker, minute): latest valid version of each bar

Re-running is idempotent: each file's rows are replaced, never duplicated.
"""
import csv
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = "market_data.db"
RAW_DIR = "data/raw"

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_market_data (
    id            INTEGER PRIMARY KEY,
    source_file   TEXT    NOT NULL,
    source_row    INTEGER NOT NULL,
    ticker        TEXT    NOT NULL,
    ts            INTEGER NOT NULL,   -- bar start, unix epoch seconds (UTC)
    open          REAL,
    high          REAL,
    low           REAL,
    close         REAL,
    volume        INTEGER,
    received_at   INTEGER,            -- unix epoch seconds, NULL if not supplied
    clean         INTEGER NOT NULL,   -- 1 = passed sanity checks
    reject_reason TEXT,
    UNIQUE (source_file, source_row)
);
CREATE INDEX IF NOT EXISTS idx_raw_ticker_ts ON raw_market_data (ticker, ts);

DROP VIEW IF EXISTS bars_1m;

-- Latest valid observation per (ticker, minute). Later received_at wins
-- (upstream corrections); rows without received_at count as oldest.
CREATE VIEW bars_1m AS
SELECT ticker, ts, datetime(ts, 'unixepoch') AS ts_utc,
       open, high, low, close, volume, received_at, source_file
FROM (
    SELECT *, ROW_NUMBER() OVER (
               PARTITION BY ticker, ts
               ORDER BY COALESCE(received_at, -1) DESC, source_file DESC, source_row DESC
           ) AS rn
    FROM raw_market_data
    WHERE clean = 1
)
WHERE rn = 1;
"""


def parse_ts(value):
    """Parse ISO timestamps with any offset ('2026-10-08 09:30:00-04:00' or
    '2026-09-29T13:30:00+00:00') into UTC epoch seconds."""
    dt = datetime.fromisoformat(value.strip())
    if dt.tzinfo is None:
        raise ValueError(f"timestamp without timezone: {value!r}")
    return int(dt.astimezone(timezone.utc).timestamp())


def normalise_ticker(raw):
    """Strip stray whitespace and upper-case, e.g. ' aapl ' -> 'AAPL'."""
    return raw.strip().upper()


def check_row(o, h, lo, c, v):
    """Return None if the bar is sane, else a reason string."""
    if min(o, h, lo, c) <= 0:
        return "non-positive price"
    if h < lo:
        return "high < low"
    if h < max(o, c) or lo > min(o, c):
        return "open/close outside high-low range"
    if v < 0:
        return "negative volume"
    return None


def load_file(cursor, path):
    name = os.path.basename(path)
    cursor.execute("DELETE FROM raw_market_data WHERE source_file = ?", (name,))
    rows, skipped = [], 0
    with open(path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):  # line 1 = header
            try:
                o, h, lo, c = (float(row[k]) for k in ("open", "high", "low", "close"))
                v = int(float(row["volume"]))
                ts = parse_ts(row["timestamp"])
                rec = row.get("received_at")
                rec = parse_ts(rec) if rec else None
                ticker = normalise_ticker(row["ticker"])
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            reason = check_row(o, h, lo, c, v)
            rows.append((name, i, ticker, ts, o, h, lo, c, v, rec,
                         int(reason is None), reason))
    cursor.executemany(
        """INSERT INTO raw_market_data
           (source_file, source_row, ticker, ts, open, high, low, close,
            volume, received_at, clean, reject_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    return len(rows), skipped


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        with conn:  # one transaction: all-or-nothing
            conn.executescript(SCHEMA)
            cur = conn.cursor()
            for file in sorted(os.listdir(RAW_DIR)):
                if not file.endswith(".csv"):
                    continue
                n, skipped = load_file(cur, os.path.join(RAW_DIR, file))
                print(f"{file}: {n} rows loaded, {skipped} unparseable")
        print(f"Database written to {DB_PATH}. Explore it with: streamlit run app.py")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
