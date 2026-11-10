"""Data helpers: Yahoo Finance download with a local cache, returns, annual statistics and plot style."""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Price cache. Set RESEARCH_DATA_DIR to share one cache folder across projects.
DATA_DIR = Path(os.environ.get("RESEARCH_DATA_DIR", Path(__file__).resolve().parent / "data"))
START, END = "2007-01-01", None

STOCKS = {
    "Tech": ["AAPL", "MSFT", "NVDA", "ADBE", "CRM", "ORCL", "CSCO", "INTC", "TXN", "QCOM", "AVGO", "IBM"],
    "Comm": ["GOOGL", "META", "NFLX", "DIS", "VZ", "T", "CMCSA", "TMUS"],
    "Discretionary": ["AMZN", "HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "TSLA", "BKNG"],
    "Staples": ["PG", "KO", "PEP", "WMT", "COST", "MDLZ", "CL", "MO", "KMB", "GIS"],
    "Health": ["JNJ", "UNH", "PFE", "MRK", "ABT", "LLY", "TMO", "AMGN", "MDT", "BMY", "GILD"],
    "Financials": ["JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "BLK", "USB", "SCHW", "MET"],
    "Industrials": ["CAT", "BA", "HON", "UPS", "GE", "MMM", "LMT", "DE", "RTX", "UNP"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY", "PSX", "MPC"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC"],
    "Materials": ["LIN", "APD", "SHW", "FCX", "NEM", "DOW"],
}
# Tradable factor proxies (all trade since mid-2007 or earlier) and sector funds for exposure reports.
FACTOR_ETFS = {
    "SPY": "US market", "IWM": "US small caps", "VTV": "US value", "VUG": "US growth",
    "IEF": "7-10y Treasuries", "HYG": "High-yield credit", "GLD": "Gold", "USO": "Crude oil", "UUP": "US dollar",
}
SECTOR_ETFS = {"XLK": "Tech", "XLF": "Financials", "XLE": "Energy", "XLV": "Health", "XLY": "Discretionary",
               "XLP": "Staples", "XLI": "Industrials", "XLU": "Utilities", "XLB": "Materials"}


def sector_map():
    return {t: s for s, ts in STOCKS.items() for t in ts}


def etf_map():
    return {**FACTOR_ETFS, **SECTOR_ETFS}


def _from_yfinance(tickers, chunk=25, pause=2.0):
    import logging
    import yfinance as yf

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    out = []
    for i in range(0, len(tickers), chunk):
        try:
            df = yf.download(tickers[i:i + chunk], start=START, end=END, auto_adjust=True, progress=False, threads=False)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            close = df["Close"]
            out.append(close.to_frame(tickers[i]) if isinstance(close, pd.Series) else close)
        time.sleep(pause)
    px = pd.concat(out, axis=1) if out else pd.DataFrame()
    return px.loc[:, px.notna().sum() > 250] if not px.empty else px


_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def _from_yahoo_chart(ticker):
    """Yahoo chart API directly: dividend- and split-adjusted closes."""
    import requests

    t0 = int(pd.Timestamp(START).timestamp())
    t1 = int((pd.Timestamp(END) if END else pd.Timestamp.now()).timestamp())
    for host in ("query2", "query1"):
        try:
            r = requests.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{ticker}", headers=_UA, timeout=20,
                             params={"period1": t0, "period2": t1, "interval": "1d", "events": "div,split",
                                     "includeAdjustedClose": "true"})
            res = r.json()["chart"]["result"][0]
            idx = pd.to_datetime(res["timestamp"], unit="s").normalize()
            adj = res["indicators"]["adjclose"][0]["adjclose"]
            s = pd.Series(adj, index=idx, name=ticker, dtype=float).dropna()
            if len(s) > 250:
                return s[~s.index.duplicated(keep="last")]
        except Exception:
            pass
    return None


def _from_stooq(ticker):
    """Stooq daily closes (adjusted for splits and dividends), no key needed."""
    import io
    import requests

    d1 = pd.Timestamp(START).strftime("%Y%m%d")
    d2 = (pd.Timestamp(END) if END else pd.Timestamp.now()).strftime("%Y%m%d")
    try:
        r = requests.get("https://stooq.com/q/d/l/", headers=_UA, timeout=20,
                         params={"s": f"{ticker.lower().replace('.', '-')}.us", "i": "d", "d1": d1, "d2": d2})
        df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"], index_col="Date")
        s = df["Close"].rename(ticker).astype(float).dropna()
        return s if len(s) > 250 else None
    except Exception:
        return None


def _download(tickers):
    """Adjusted closes from yfinance; tickers it misses are fetched from the Yahoo chart API, then from Stooq."""
    tickers = list(tickers)
    try:
        px = _from_yfinance(tickers)
    except ImportError:
        px = pd.DataFrame()
    source = {t: "yfinance" for t in px.columns}
    for fetch, name in ((_from_yahoo_chart, "Yahoo chart API"), (_from_stooq, "Stooq")):
        missing = [t for t in tickers if t not in px.columns]
        if not missing:
            break
        got = {}
        for t in missing:
            s = fetch(t)
            if s is not None:
                got[t] = s
                source[t] = name
            time.sleep(0.3)
        if got:
            px = pd.concat([px, pd.DataFrame(got)], axis=1)
    if not px.empty:
        counts = pd.Series(source).value_counts().to_dict()
        print(f"Downloaded {px.shape[1]} of {len(tickers)} tickers: {counts}")
    return px.sort_index()


def load_prices(universe="stocks", refresh=False):
    """Adjusted close prices, downloaded on first use and kept in <DATA_DIR>/<universe>.csv so later runs start instantly.
    refresh=True forces a new download."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{universe}_from{START[:4]}.csv"   # start year in the name, so caches from other projects are not reused
    if path.exists() and not refresh:
        px = pd.read_csv(path, index_col=0, parse_dates=True)
        if px.shape[0] > 250 and px.shape[1] > 0:
            return px
        path.unlink()  # empty or broken cache from a failed download
    tickers = {"stocks": list(sector_map()), "etfs": list(etf_map())}[universe]
    px = _download(tickers)
    if px.empty or px.shape[1] < 0.8 * len(tickers):
        raise RuntimeError(
            f"Price download returned {px.shape[1]} of {len(tickers)} tickers for '{universe}'. Nothing was cached. "
            "All three sources (yfinance, Yahoo chart API, Stooq) failed or were throttled. "
            "Run `pip install -U yfinance requests`, check the internet connection and retry in a few minutes.")
    # Keep every ticker: stocks listed after 2007 stay empty before their first trade instead of being dropped.
    px = px.loc[px.notna().sum(axis=1) >= 0.5 * px.shape[1]]      # drop non-trading days
    px = px.ffill(limit=5)
    px.index.name = "Date"
    px.to_csv(path)
    return px


def load_returns(universe="stocks", log=False, **kw):
    """Daily returns; NaN before a ticker's first trade."""
    px = load_prices(universe, **kw)
    r = np.log(px).diff() if log else px.pct_change(fill_method=None)
    return r.iloc[1:]


def annual_stats(r, periods=252):
    """Annualised return, volatility, Sharpe and max drawdown of a daily simple-return series.
    Sharpe here is the geometric annual return divided by annual volatility, with no risk-free rate."""
    r = pd.Series(r).dropna()
    ann_ret = (1 + r).prod() ** (periods / len(r)) - 1
    vol = r.std() * np.sqrt(periods)
    wealth = (1 + r).cumprod()
    mdd = (wealth / wealth.cummax() - 1).min()
    return pd.Series({"ann_return": ann_ret, "ann_vol": vol, "sharpe": ann_ret / vol if vol else np.nan, "max_dd": mdd})


def set_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.figsize": (9, 4.5), "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "figure.dpi": 100,
    })


# ---------- Windows guard ----------
def _guard_threadpoolctl():
    """scikit-learn asks threadpoolctl to cap BLAS threads before k-means runs. On Windows, when the
    environment sits in a folder with non-ASCII characters (e.g. OneDrive\\文件), threadpoolctl can read the
    OpenBLAS path back garbled and fail to load it. Skip any library it cannot load instead of crashing;
    the only effect is that BLAS thread counts are left at their defaults."""
    try:
        import threadpoolctl as tpc
    except ImportError:
        return
    ctrl = getattr(tpc, "ThreadpoolController", None)
    if ctrl is None or getattr(ctrl, "_path_guard", False) or not hasattr(ctrl, "_make_controller_from_path"):
        return
    original = ctrl._make_controller_from_path

    def safe(self, filepath):
        try:
            return original(self, filepath)
        except OSError:
            return None

    ctrl._make_controller_from_path = safe
    ctrl._path_guard = True


_guard_threadpoolctl()
