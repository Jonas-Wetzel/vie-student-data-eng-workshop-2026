"""Streamlit research dashboard for the bars built by main.py.

Run:  streamlit run app.py

Reads the cleaned 1-minute bars (bars_1m) for one ticker and day and resamples
them to 1m / 5m / 15m / 30m / 1h in pandas. Switching frequency doesn't query
the database again: the day's 1m data is cached, and resampling ~400 rows
takes about a millisecond.
"""
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DB_PATH = "market_data.db"
FREQUENCIES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}
RV_WINDOW = 30  # number of consecutive 1-minute returns
# Exchange timezone for display, by ticker suffix; anything else is treated as US.
EXCHANGE_TZ = {".PA": "Europe/Paris"}
DEFAULT_TZ = "America/New_York"


def query(sql, params=()):
    # sqlite3's own context manager only commits; closing() actually closes the connection
    with closing(sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)) as conn:
        return pd.read_sql_query(sql, conn, params=params)


@st.cache_data
def load_tickers():
    return query("SELECT DISTINCT ticker FROM bars_1m ORDER BY ticker")["ticker"].tolist()


@st.cache_data
def load_dates(ticker):
    df = query(
        "SELECT DISTINCT date(ts, 'unixepoch') AS d FROM bars_1m WHERE ticker = ? ORDER BY d DESC",
        (ticker,),
    )
    return df["d"].tolist()


@st.cache_data
def load_day(ticker, date):
    """1-minute bars for one ticker and one (UTC) date, indexed by UTC timestamp."""
    df = query(
        """SELECT ts, open, high, low, close, volume FROM bars_1m
           WHERE ticker = ? AND date(ts, 'unixepoch') = ? ORDER BY ts""",
        (ticker, date),
    )
    df.index = pd.to_datetime(df.pop("ts"), unit="s", utc=True)
    return df


def resample(df_1m, minutes):
    """Aggregate 1m bars into OHLCV candles labelled by their start time.

    Buckets are anchored to the session open (first bar floored to 30 min), so
    US hourly candles are 09:30-10:29, 10:30-11:29, ... instead of starting at
    09:00 with a half-empty first bar.
    """
    if minutes == 1:
        out = df_1m.copy()
        out["n_minutes"] = 1
        return out
    origin = df_1m.index[0].floor("30min")
    out = df_1m.resample(f"{minutes}min", origin=origin, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        n_minutes=("close", "count"),
    )
    return out[out["n_minutes"] > 0]  # drop buckets with no data at all


def realized_vol(df_1m, window=RV_WINDOW):
    """RV_t = 100 * sqrt(sum of the last `window` squared 1m log returns).

    Only returns within the selected day are used (no overnight return), so
    the first value appears at the 31st bar of the session.
    """
    r = np.log(df_1m["close"] / df_1m["close"].shift(1))
    return 100 * np.sqrt((r**2).rolling(window, min_periods=window).sum())


def exchange_tz(ticker):
    return next((tz for sfx, tz in EXCHANGE_TZ.items() if ticker.endswith(sfx)), DEFAULT_TZ)


def to_display_tz(index, tz):
    """Plotly ignores tz-aware datetimes, so convert to naive wall-clock time."""
    return index.tz_convert(tz).tz_localize(None)


def build_chart(bars, rv, tz, freq):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
        subplot_titles=(
            f"Price ({freq})",
            f"Rolling realized volatility ({RV_WINDOW} x 1m returns, %)",
        ),
    )
    hover = [
        f"n_minutes: {n}" + ("  (incomplete)" if n < FREQUENCIES[freq] else "")
        + f"<br>volume: {v:,.0f}"
        for n, v in zip(bars["n_minutes"], bars["volume"])
    ]
    fig.add_trace(
        go.Candlestick(
            x=to_display_tz(bars.index, tz),
            open=bars["open"], high=bars["high"], low=bars["low"], close=bars["close"],
            text=hover, name="OHLC", showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=to_display_tz(rv.index, tz), y=rv, mode="lines", name="RV",
                   line=dict(width=1.5), showlegend=False,
                   hovertemplate="%{x|%H:%M}<br>RV: %{y:.3f}%<extra></extra>"),
        row=2, col=1,
    )
    fig.update_layout(height=720, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_rangeslider_visible=False, hovermode="x unified")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RV %", rangemode="tozero", row=2, col=1)
    fig.update_xaxes(title_text=f"Time ({tz})", row=2, col=1)
    return fig


def main():
    st.set_page_config(page_title="Intraday bars", layout="wide")
    st.title("Intraday bars explorer")

    try:
        tickers = load_tickers()
    except Exception as e:  # missing DB / views
        st.error(f"Could not read {DB_PATH}: {e}. Run `python main.py` first.")
        st.stop()

    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    ticker = c1.selectbox("Ticker", tickers)
    date = c2.selectbox("Trading date (UTC)", load_dates(ticker))
    freq = c3.segmented_control("Frequency", list(FREQUENCIES), default="5m") or "5m"
    use_utc = c4.radio("Time zone", ["Exchange", "UTC"], horizontal=True) == "UTC"
    tz = "UTC" if use_utc else exchange_tz(ticker)

    df_1m = load_day(ticker, date)
    if df_1m.empty:
        st.warning("No data for this selection.")
        st.stop()

    bars = resample(df_1m, FREQUENCIES[freq])
    rv = realized_vol(df_1m)

    day_ret = 100 * (df_1m["close"].iloc[-1] / df_1m["open"].iloc[0] - 1)
    m = st.columns(5)
    m[0].metric("Close", f"{df_1m['close'].iloc[-1]:.2f}", f"{day_ret:+.2f}% on day")
    m[1].metric("Day range", f"{df_1m['low'].min():.2f} - {df_1m['high'].max():.2f}")
    m[2].metric("Volume", f"{df_1m['volume'].sum():,.0f}")
    m[3].metric("Latest RV", "n/a" if rv.dropna().empty else f"{rv.dropna().iloc[-1]:.3f}%")
    incomplete = int((bars["n_minutes"] < FREQUENCIES[freq]).sum())
    m[4].metric(f"{freq} bars", f"{len(bars)}", f"{incomplete} incomplete" if incomplete else None,
                delta_color="off")

    st.plotly_chart(build_chart(bars, rv, tz, freq), width="stretch")

    with st.expander("Bar data"):
        table = bars.copy()
        table.index = to_display_tz(table.index, tz)
        table.index.name = f"start ({tz})"
        st.dataframe(table, width="stretch")


if __name__ == "__main__":
    main()
