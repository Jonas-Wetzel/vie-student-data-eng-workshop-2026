"""Streamlit research dashboard for the bars built by main.py.

Run:  streamlit run app.py

Loads a ticker's cleaned 1-minute bars (bars_1m) once, then slices the chosen
range of trading dates and resamples it to 1m / 5m / 15m / 30m / 1h in pandas.
All times are UTC. Several trading days are shown on one continuous chart with
the overnight/weekend gaps collapsed.
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
DEFAULT_DAYS = 5  # trading days shown when a ticker is first selected
GAP = pd.Timedelta("1h")  # gaps longer than this between bars are collapsed on the chart


def query(sql, params=()):
    # sqlite3's own context manager only commits; closing() actually closes the connection
    with closing(sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)) as conn:
        return pd.read_sql_query(sql, conn, params=params)


@st.cache_data
def load_tickers():
    return query("SELECT DISTINCT ticker FROM bars_1m ORDER BY ticker")["ticker"].tolist()


@st.cache_data
def load_ticker(ticker):
    """All 1-minute bars for one ticker, indexed by UTC timestamp."""
    df = query(
        "SELECT ts, open, high, low, close, volume FROM bars_1m WHERE ticker = ? ORDER BY ts",
        (ticker,),
    )
    df.index = pd.to_datetime(df.pop("ts"), unit="s", utc=True)
    return df


def resample_day(day_1m, minutes):
    """Aggregate one day's 1m bars into OHLCV candles labelled by their start time.

    Buckets are anchored to the day's first bar floored to 30 min, so a session
    opening at 13:30 UTC gets hourly candles 13:30, 14:30, ... rather than a
    half-empty 13:00 candle.
    """
    origin = day_1m.index[0].floor("30min")
    out = day_1m.resample(f"{minutes}min", origin=origin, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        n_minutes=("close", "count"),
    )
    return out[out["n_minutes"] > 0]  # drop buckets with no data at all


def resample(df_1m, minutes):
    if minutes == 1:
        return df_1m.assign(n_minutes=1)
    return pd.concat(resample_day(day, minutes) for _, day in df_1m.groupby(df_1m.index.date))


def realized_vol(df_1m, window=RV_WINDOW):
    """RV_t = 100 * sqrt(sum of the last `window` squared 1m log returns).

    Computed separately for each trading day (no overnight returns), so each
    day's line starts at its 31st bar.
    """
    def one_day(close):
        r = np.log(close / close.shift(1))
        return 100 * np.sqrt((r**2).rolling(window, min_periods=window).sum())

    close = df_1m["close"]
    return close.groupby(close.index.date, group_keys=False).apply(one_day)


def naive(index):
    """Plotly ignores tz-aware datetimes; drop the tz so it plots UTC as-is."""
    return index.tz_localize(None)


def build_chart(bars, rv, df_1m, freq):
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
            x=naive(bars.index),
            open=bars["open"], high=bars["high"], low=bars["low"], close=bars["close"],
            text=hover, name="OHLC", showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=naive(rv.index), y=rv, mode="lines", name="RV",
                   line=dict(width=1.5), showlegend=False,
                   hovertemplate="%{x|%Y-%m-%d %H:%M}<br>RV: %{y:.3f}%<extra></extra>"),
        row=2, col=1,
    )

    # Collapse overnight / weekend gaps so consecutive trading days sit side by side,
    # and mark where each new day starts.
    ts = naive(df_1m.index).to_series()
    gaps = ts.diff()
    breaks = [
        dict(values=[(start + pd.Timedelta("1min")).isoformat()],
             dvalue=(gap - pd.Timedelta("1min")).total_seconds() * 1000)
        for start, gap in zip(ts.shift(1)[gaps > GAP], gaps[gaps > GAP])
    ]
    fig.update_xaxes(rangebreaks=breaks)
    for day_start in ts[gaps > GAP]:
        fig.add_vline(x=day_start, line=dict(width=1, dash="dot", color="grey"), opacity=0.5)

    fig.update_layout(height=720, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_rangeslider_visible=False, hovermode="x unified")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RV %", rangemode="tozero", row=2, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)
    return fig


def main():
    st.set_page_config(page_title="Intraday bars", layout="wide")
    st.title("Intraday bars explorer")

    try:
        tickers = load_tickers()
    except Exception as e:  # missing DB / views
        st.error(f"Could not read {DB_PATH}: {e}. Run `python main.py` first.")
        st.stop()

    c1, c2 = st.columns([1, 2])
    ticker = c1.selectbox("Ticker", tickers)
    freq = c2.segmented_control("Frequency", list(FREQUENCIES), default="5m") or "5m"

    all_1m = load_ticker(ticker)
    if all_1m.empty:
        st.warning("No data for this ticker.")
        st.stop()

    dates = sorted(set(all_1m.index.date))
    if len(dates) > 1:
        start, end = st.select_slider(
            "Trading dates (UTC)", options=dates,
            value=(dates[max(0, len(dates) - DEFAULT_DAYS)], dates[-1]),
            format_func=lambda d: d.strftime("%a %d %b"), key=f"dates_{ticker}",
        )
    else:
        start = end = dates[0]
        st.caption(f"Only one trading date for {ticker}: {start:%a %d %b %Y}")

    in_range = (all_1m.index.date >= start) & (all_1m.index.date <= end)
    df_1m = all_1m[in_range]
    bars = resample(df_1m, FREQUENCIES[freq])
    rv = realized_vol(df_1m)

    ret = 100 * (df_1m["close"].iloc[-1] / df_1m["open"].iloc[0] - 1)
    n_days = len(set(df_1m.index.date))
    m = st.columns(5)
    m[0].metric("Last close", f"{df_1m['close'].iloc[-1]:.2f}",
                f"{ret:+.2f}% over {n_days} day{'s' if n_days > 1 else ''}")
    m[1].metric("Range", f"{df_1m['low'].min():.2f} - {df_1m['high'].max():.2f}")
    m[2].metric("Volume", f"{df_1m['volume'].sum():,.0f}")
    m[3].metric("Latest RV", "n/a" if rv.dropna().empty else f"{rv.dropna().iloc[-1]:.3f}%")
    incomplete = int((bars["n_minutes"] < FREQUENCIES[freq]).sum())
    m[4].metric(f"{freq} bars", f"{len(bars)}", f"{incomplete} incomplete" if incomplete else None,
                delta_color="off")

    st.plotly_chart(build_chart(bars, rv, df_1m, freq), width="stretch")

    with st.expander("Bar data"):
        table = bars.copy()
        table.index = naive(table.index)
        table.index.name = "start (UTC)"
        st.dataframe(table, width="stretch")


if __name__ == "__main__":
    main()
