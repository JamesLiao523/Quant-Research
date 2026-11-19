"""Local macro and market inputs for the statistical-learning allocation study.

Reads the frozen provider exports (BIS/OECD/Eurostat/ECB) and the latest
gathered FRED/Yahoo snapshot, applies approximate publication lags, and builds
country-level factor categories from fixed recipes using the closest local
inputs. All histories are current revisions, not
point-in-time vintages.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

ECON = Path.home() / "OneDrive" / "Investment" / "Market Data" / "Econ"
PROVIDERS = ("BIS", "OECD", "EUROSTAT", "ECB")
QDF_COLS = ["cid", "xcat", "real_date", "value"]

SLEEVES = {
    "USEQ": ("SPY", "US equities"),
    "DXEQ": ("EFA", "Developed ex-US equities"),
    "EMEQ": ("EEM", "Emerging equities"),
    "USDU": ("IEI", "US intermediate Treasuries"),
    "DXDU": ("BWX", "Unhedged developed government bonds"),
    "DXFX": ("UDN", "Short USD futures ETF; NOT FX forwards"),
    "G2CR": ("LQD", "US investment-grade corporate bonds; NOT CDS"),
    "EMCR": ("EMB", "USD emerging sovereign bonds"),
    "EMLC": ("EMLC", "Emerging local-currency bonds"),
    "MTL": ("DBB", "Industrial-metals futures ETF"),
    "GLD": ("GLD", "Physical gold ETF; NOT futures"),
}

# Static current official inflation targets (midpoints), in percent. These are
# an approximation: real target histories changed over time and are not PIT.
INFLATION_TARGETS = {
    "USD": 2.0, "EUR": 2.0, "JPY": 2.0, "GBP": 2.0, "CHF": 1.0, "SEK": 2.0,
    "NOK": 2.0, "AUD": 2.5, "CAD": 2.0, "NZD": 2.0, "DEM": 2.0, "FRF": 2.0,
    "ITL": 2.0, "NLG": 2.0, "ESP": 2.0, "CZK": 2.0, "HUF": 3.0, "PLN": 2.5,
    "RON": 2.5, "ILS": 2.0, "KRW": 2.0, "CNY": 3.0, "INR": 4.0, "IDR": 2.5,
    "THB": 2.0, "PHP": 3.0, "MYR": 2.5, "SGD": 2.0, "TWD": 2.0, "BRL": 3.0,
    "MXN": 3.0, "CLP": 3.0, "COP": 3.0, "PEN": 2.0, "ZAR": 4.5, "TRY": 5.0,
    "RUB": 4.0, "HKD": 2.0, "DKK": 2.0,
}

# FRED series -> (cid, family, native frequency, publication lag in days).
FRED_MAP = {
    "INDPRO": ("USD", "IP", "M", 16),
    "GDPC1": ("USD", "RGDP", "Q", 90),
    "GDPNOW": ("USD", "GDPNOW", "D", 1),
    "RRSFS": ("USD", "RSALES", "M", 16),
    "PAYEMS": ("USD", "EMPL", "M", 7),
    "CLF16OV": ("USD", "WFORCE", "M", 7),
    "UNRATE": ("USD", "UNEMP", "M", 7),
    "CPIAUCSL": ("USD", "CPIH", "M", 14),
    "CPILFESL": ("USD", "CPIC", "M", 14),
    "PCEPILFE": ("USD", "PCECORE", "M", 30),
    "PPIACO": ("USD", "PPI", "M", 16),
    "CSUSHPINSA": ("USD", "HPI", "M", 60),
    "M2SL": ("USD", "MONEY", "M", 28),
    "MABMM301EZM189S": ("EUR", "MONEYALT", "M", 60),
    "MABMM301JPM189S": ("JPY", "MONEY", "M", 60),
    "MABMM301GBM189S": ("GBP", "MONEY", "M", 60),
    "DFF": ("USD", "CASHRATE", "D", 1),
    "DFII5": ("USD", "REALYLD5Y", "D", 1),
    "DGS2": ("USD", "GB02Y_YLD", "D", 1),
    "DGS5": ("USD", "GB05Y_YLD", "D", 1),
    "DGS10": ("USD", "GB10Y_YLD", "D", 1),
    "THREEFYTP5": ("USD", "DU05YETP", "D", 1),
    "T5YIE": ("USD", "INFE5Y", "D", 1),
    "T5YIFR": ("USD", "INFE5Y5Y", "D", 1),
    "MICH": ("USD", "HHINFE", "M", 14),
    "UMCSENT": ("USD", "CSENT", "M", 14),
    "NFCI": ("USD", "FCI", "W", 7),
    "T10Y2Y": ("USD", "SLOPE10Y2Y", "D", 1),
    "T10Y3M": ("USD", "SLOPE10Y3M", "D", 1),
    "BAA10Y": ("USD", "CRSPRDBAA", "D", 1),
    "DTWEXBGS": ("USD", "DXYBROAD", "D", 1),
    "IEABC": ("USD", "CABAL", "Q", 90),
    "MKTGDPUSA646NWDB": ("USD", "NGDPUSD", "A", 365),
    "BOPGSTB": ("USD", "TRADEBAL", "M", 35),
    "IIPUSNETIQ": ("USD", "NIIP", "Q", 90),
    "BUSINV": ("USD", "BINV", "M", 45),
    "ISRATIO": ("USD", "ISRATIO", "M", 45),
    "CHNB6BLTT02STSAQ": ("CNY", "CABGDP", "Q", 90),
    "INDB6BLTT02STSAQ": ("INR", "CABGDP", "Q", 90),
    "PCOPPUSDM": ("USD", "PXCOPPER", "M", 30),
    "PALUMUSDM": ("USD", "PXALUM", "M", 30),
    "PZINCUSDM": ("USD", "PXZINC", "M", 30),
    "PNICKUSDM": ("USD", "PXNICKEL", "M", 30),
    # Requested series (Macro Data Extract/requests.csv); used when gathered.
    "DRTSCILM": ("USD", "BLSTIGHT", "Q", 35),
    "DRTSCIS": ("USD", "BLSTIGHTS", "Q", 35),
    "TDSP": ("USD", "HHDSR", "Q", 100),
    "PERMIT": ("USD", "PERMITS", "M", 18),
    "HOUST": ("USD", "STARTS", "M", 18),
}
OECD_RATE_CODES = {"AU": "AUD", "CA": "CAD", "CH": "CHF", "DE": "DEM", "GB": "GBP",
                   "JP": "JPY", "NZ": "NZD", "SE": "SEK", "NO": "NOK", "FR": "FRF",
                   "IT": "ITL", "ES": "ESP", "NL": "NLG"}
# Nominal GDP in USD (World Bank, annual) for GDP-weighted country baskets.
GDP_CODES = {"AU": "AUD", "CA": "CAD", "CH": "CHF", "DE": "DEM", "FR": "FRF",
             "IT": "ITL", "ES": "ESP", "NL": "NLG", "GB": "GBP", "JP": "JPY",
             "NZ": "NZD", "SE": "SEK", "NO": "NOK", "US": "USD", "BR": "BRL",
             "CL": "CLP", "CN": "CNY", "CO": "COP", "CZ": "CZK", "HU": "HUF",
             "ID": "IDR", "IL": "ILS", "IN": "INR", "KR": "KRW", "MX": "MXN",
             "MY": "MYR", "PE": "PEN", "PH": "PHP", "PL": "PLN", "RO": "RON",
             "SG": "SGD", "TH": "THB", "TR": "TRY", "ZA": "ZAR"}
for _code, _cid in GDP_CODES.items():
    FRED_MAP[f"MKTGDP{_code}A646NWDB"] = (_cid, "NGDPUSD", "A", 545)
for _code, _cid in OECD_RATE_CODES.items():
    FRED_MAP[f"IRLTLT01{_code}M156N"] = (_cid, "LTYLD", "M", 30)
    FRED_MAP[f"IR3TIB01{_code}M156N"] = (_cid, "STYLD", "M", 30)


def find_gatherer() -> Path:
    """Locate Macro Data Extract/gather_econ_data.py.

    Order: the MACRO_GATHERER environment variable, the repository this file
    sits in, then the default local clone under OneDrive.
    """
    if os.environ.get("MACRO_GATHERER"):
        return Path(os.environ["MACRO_GATHERER"])
    rel = Path("Macro Data Extract") / "gather_econ_data.py"
    for parent in Path(__file__).resolve().parents:
        if (parent / rel).exists():
            return parent / rel
    return (Path.home() / "OneDrive" / "文件" / "GitHub" / "Quant-Research" / rel)


def refresh_data(econ: Path = ECON, force: bool = False) -> Path:
    """Pull any series listed in requests.csv that the latest snapshot lacks.

    Runs the repository gatherer in-process, so API keys are read from the
    local API Key.xlsx (or environment variables) and never printed. A series
    the providers reject is recorded as skipped and not retried on every run.
    """
    path = find_gatherer()
    if not path.exists():
        raise FileNotFoundError(f"gatherer not found at {path}; set MACRO_GATHERER")
    spec = importlib.util.spec_from_file_location("gather_econ_data", path)
    gatherer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gatherer)
    wanted = gatherer.load_requests()

    have: set[str] = set()
    try:
        manifest = json.loads((latest_gathered(econ) / "manifest.json").read_text())
        for output in manifest.get("outputs", []):
            have.update(output.get("series", []))
        have.update(item["id"] for item in manifest.get("requested_skipped", []))
    except FileNotFoundError:
        manifest = None
    missing = [s for ids in wanted.values() for s in ids if s not in have]
    if manifest is not None and not missing and not force:
        print(f"snapshot up to date: {latest_gathered(econ).name}")
        return latest_gathered(econ)

    print(f"pulling a new snapshot: {len(missing)} requested series missing")
    # httpx logs request URLs at INFO, and those URLs carry the API keys.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    folder = gatherer.gather(econ.parent, set(gatherer.PROVIDERS), [])
    new = json.loads((folder / "manifest.json").read_text())
    skipped = [item["id"] for item in new.get("requested_skipped", [])]
    print(f"published {folder.name}; requested series skipped: {skipped or 'none'}")
    return folder


def latest_gathered(econ: Path = ECON) -> Path:
    """Return the newest gathered observation snapshot."""
    snapshots = sorted((econ / "Gathered").glob("20*/manifest.json"))
    if not snapshots:
        raise FileNotFoundError(
            "No gathered snapshot; run the Macro Data Extract gatherer"
        )
    return snapshots[-1].parent


def business_days(
    start: str = "1990-01-01", end: str | None = None
) -> pd.DatetimeIndex:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    return pd.bdate_range(start, end)


FILL_LIMIT = {"D": 15, "W": 15, "M": 70, "Q": 145, "A": 400}


def to_daily(series: pd.Series, lag_days: int, bdays: pd.DatetimeIndex,
             freq: str = "M") -> pd.Series:
    """Shift observations by an approximate publication lag and forward fill.

    The fill horizon is capped by the native frequency so discontinued series
    (for example legacy euro policy rates ending in 1998) stop instead of
    extending stale values indefinitely.
    """
    shifted = series.copy()
    shifted.index = shifted.index + pd.Timedelta(days=int(lag_days))
    shifted = shifted[~shifted.index.duplicated(keep="last")].sort_index()
    return shifted.reindex(bdays, method="ffill", limit=FILL_LIMIT[freq])


def yoy(series: pd.Series, periods: int) -> pd.Series:
    return 100.0 * (series / series.shift(periods) - 1.0)


def ar_6m6m(series: pd.Series) -> pd.Series:
    """Annualized growth of the last six months over the prior six, monthly data."""
    mean6 = series.rolling(6).mean()
    return 100.0 * ((mean6 / mean6.shift(6)) ** 2 - 1.0)


def d3m3ml3(series: pd.Series) -> pd.Series:
    """Average of the last three months minus the prior three, monthly data."""
    mean3 = series.rolling(3).mean()
    return mean3 - mean3.shift(3)


def trend_20q(quarterly_yoy: pd.Series) -> pd.Series:
    """Twenty-quarter rolling mean of quarterly year-on-year growth."""
    return quarterly_yoy.rolling(20, min_periods=12).mean()


def vs_5y_median(series: pd.Series, periods_per_year: int) -> pd.Series:
    window = 5 * periods_per_year
    return series - series.rolling(window, min_periods=window // 2).median()


def load_provider_raw(econ: Path = ECON) -> pd.DataFrame:
    """Load every provider raw series: one row per observation with family metadata."""
    frames = []
    for provider in PROVIDERS:
        folder = econ / f"{provider}_Current_Revision"
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["series"]:
            if entry.get("status") != "ok":
                continue
            field = entry["fields"][0]
            family = field["xcat"].split("_")[0]
            # BIS publishes three different credit measures. They must stay apart:
            # merging them interleaves a gap (~0), a ratio (~150) and a level (~5,000).
            credit = {"CREDIT_GDP_GAP": "CREDITGAP", "CREDIT_GDP": "CREDITGDP",
                      "CREDIT_YOY": "CREDITLVL"}
            matched = next((v for k, v in credit.items()
                            if field["xcat"].startswith(k)), None)
            for known in ("RGDP", "RSALES", "UNEMP", "IP", "PPI", "CPIH", "M2",
                          "POLRATE", "CREDIT", "REER", "NEER", "RPP", "BCONF",
                          "CCONF", "CLI"):
                if field["xcat"].startswith(known):
                    family = known
                    break
            if matched:
                family = matched
            raw = pd.read_csv(folder / entry["filename"])
            raw["obs_date"] = pd.to_datetime(raw["obs_date"])
            raw = raw.dropna(subset=["value"])
            # BIS publishes real and nominal property files under one family.
            if family == "RPP" and "real" in str(field.get("notes", "")).lower():
                family = "RPPREAL"
            if family == "RPP" and field["xcat"].startswith("RPP_R"):
                family = "RPPREAL"
            frames.append(pd.DataFrame({
                "provider": provider,
                "series": entry["filename"],
                "cid": field["cid"],
                "family": family,
                "freq": field["frequency"],
                "lag": int(field["publication_lag_days"]),
                "obs_date": raw["obs_date"],
                "value": pd.to_numeric(raw["value"], errors="raise"),
            }))
    return pd.concat(frames, ignore_index=True)


def load_fred_raw(econ: Path = ECON) -> pd.DataFrame:
    """Load the gathered FRED snapshot restricted to the mapped research series."""
    table = pd.read_parquet(latest_gathered(econ) / "fred.parquet")
    table = table[table["series_id"].isin(FRED_MAP)].copy()
    meta = table["series_id"].map(FRED_MAP)
    table["cid"] = meta.str[0]
    table["family"] = meta.str[1]
    table["freq"] = meta.str[2]
    table["lag"] = meta.str[3]
    table = table.rename(columns={"as_of_date": "obs_date"})
    table["provider"] = "FRED"
    table["series"] = table["series_id"]
    return table[["provider", "series", "cid", "family", "freq", "lag", "obs_date",
                  "value"]]


EM_CIDS = {"BRL", "CLP", "CNY", "COP", "CZK", "HUF", "IDR", "ILS", "INR", "KRW",
           "MXN", "MYR", "PEN", "PHP", "PLN", "RON", "SGD", "THB", "TRY", "TWD",
           "ZAR", "RUB"}
EM_START = pd.Timestamp("1996-01-01")

INDEX_FAMILIES = {"CPIH", "CPIC", "RPP", "RPPREAL", "REER", "NEER", "IP", "RSALES",
                  "RGDP", "MONEY", "M2", "CREDITLVL", "PPI"}


def series_map(raw: pd.DataFrame) -> dict[tuple[str, str], tuple[pd.Series, int, str]]:
    """Index raw observations as (cid, family) -> (series, lag, freq)."""
    out: dict[tuple[str, str], tuple[pd.Series, int, str]] = {}
    for (cid, family), group in raw.groupby(["cid", "family"]):
        # One source series per (cid, family): the longest history leads and the
        # others only fill dates it does not cover. Taking the last value per
        # date across sources would splice different bases or measures.
        parts = [
            g.set_index("obs_date")["value"].sort_index().groupby(level=0).last()
            for _, g in group.groupby("series")
        ]
        parts.sort(key=len, reverse=True)
        series = parts[0]
        for extra in parts[1:]:
            head = extra.loc[extra.index < series.index.min()]
            if head.empty:
                continue
            if family in INDEX_FAMILIES:
                # rebase the older index onto the leading one at their first overlap
                common = series.index.intersection(extra.index)
                if common.empty:
                    continue
                head = head * series[common[0]] / extra[common[0]]
            series = series.combine_first(head)
        out[(cid, family)] = (
            series, int(group["lag"].iloc[0]), str(group["freq"].iloc[0])
        )
    return out


def _emit(rows: list, cid: str, xcat: str, series: pd.Series, lag: int,
          bdays: pd.DatetimeIndex, freq: str = "M") -> None:
    daily = to_daily(series.dropna(), lag, bdays, freq).dropna()
    if len(daily) < 260:
        return
    rows.append(pd.DataFrame({
        "cid": cid, "xcat": xcat, "real_date": daily.index, "value": daily.values,
    }))


def build_categories(econ: Path = ECON) -> pd.DataFrame:
    """Build the country-level factor categories as a long-format DataFrame (cid, xcat, real_date, value).

    Every category follows a fixed recipe built from the closest local
    input; mapping caveats are documented in the study notebook.
    """
    bdays = business_days()
    raw = pd.concat([load_provider_raw(econ), load_fred_raw(econ)], ignore_index=True)
    data = series_map(raw)
    cids = sorted({cid for cid, _ in data})
    rows: list[pd.DataFrame] = []

    def get(cid: str, family: str) -> tuple[pd.Series, int] | None:
        item = data.get((cid, family))
        return (item[0], item[1]) if item else None

    # US nominal GDP in billions for ratio denominators (annual, interpolated).
    ngdp_usd = None
    if (item := get("USD", "NGDPUSD")) is not None:
        ngdp_usd = (item[0] / 1e9).resample("QE").last().interpolate(limit=8)

    for cid in cids:
        target = INFLATION_TARGETS.get(cid, 2.0)

        # Real GDP trend: 20-quarter mean of quarterly YoY growth.
        gdp_trend = None
        if (item := get(cid, "RGDP")) is not None:
            level, lag = item
            quarterly = level.resample("QE").last()
            gdp_yoy = yoy(quarterly, 4)
            gdp_trend = to_daily(trend_20q(gdp_yoy).dropna(), lag, bdays, "Q")
            _emit(rows, cid, "RGDP_YOY", gdp_yoy.dropna(), lag, bdays, "Q")

        # Industrial production: YoY 3mma, and the excess over the GDP trend.
        if (item := get(cid, "IP")) is not None:
            level, lag = item
            ip_yoy3 = yoy(level, 12).rolling(3).mean()
            daily_ip = to_daily(ip_yoy3.dropna(), lag, bdays)
            if gdp_trend is not None:
                xip = (daily_ip - gdp_trend).dropna()
                if len(xip) > 260:
                    rows.append(pd.DataFrame({
                        "cid": cid, "xcat": "XIP",
                        "real_date": xip.index, "value": xip.values}))
            _emit(rows, cid, "IP_YOY3M", ip_yoy3.dropna(), lag, bdays)

        # Real retail sales (private-consumption proxy) excess over GDP trend.
        if (item := get(cid, "RSALES")) is not None:
            level, lag = item
            rs_yoy3 = yoy(level, 12).rolling(3).mean()
            daily_rs = to_daily(rs_yoy3.dropna(), lag, bdays)
            if gdp_trend is not None:
                xcons = (daily_rs - gdp_trend).dropna()
                if len(xcons) > 260:
                    rows.append(pd.DataFrame({
                        "cid": cid, "xcat": "XRPCONS",
                        "real_date": xcons.index, "value": xcons.values}))

        # Broad excess CPI: headline (+core where available) YoY and 6m/6m
        # annualized, averaged, minus the static target.
        cpi_parts = []
        for family in ("CPIH", "CPIC"):
            if (item := get(cid, family)) is not None:
                level, lag = item
                monthly = level.resample("ME").last().dropna()
                cpi_parts.append((yoy(monthly, 12), lag))
                cpi_parts.append((ar_6m6m(monthly), lag))
        if cpi_parts:
            joined = pd.concat([p for p, _ in cpi_parts], axis=1).mean(axis=1)
            lag = cpi_parts[0][1]
            _emit(rows, cid, "XCPIB_PAR", (joined - target).dropna(), lag, bdays)
            _emit(rows, cid, "CPIH_YOY", cpi_parts[0][0].dropna(), lag, bdays)

        # Excess house-price growth: BIS nominal property (US: Case-Shiller).
        if (item := get(cid, "RPP")) is not None or (
            item := get(cid, "HPI")
        ) is not None:
            level, lag = item
            hp_yoy = yoy(level.resample("QE").last().dropna(), 4)
            if hp_yoy.dropna().empty:
                hp_yoy = yoy(level.resample("ME").last().dropna(), 12)
            _emit(rows, cid, "XHPI", (hp_yoy - target).dropna(), lag, bdays, "Q")

        # Real policy rate, and the composite real-rate measure.
        pol = get(cid, "POLRATE") or get(cid, "CASHRATE")
        cpi_yoy_daily = None
        if cpi_parts:
            cpi_yoy_daily = to_daily(cpi_parts[0][0].dropna(), cpi_parts[0][1], bdays)
        if pol is not None and cpi_yoy_daily is not None:
            pol_daily = to_daily(pol[0].dropna(), pol[1], bdays, "D")
            rir = (pol_daily - cpi_yoy_daily).dropna()
            parts = [rir]
            if (item := get(cid, "REALYLD5Y")) is not None:
                parts.append(to_daily(item[0].dropna(), item[1], bdays, "D"))
            riry = pd.concat(parts, axis=1).mean(axis=1).dropna()
            if len(riry) > 260:
                rows.append(pd.DataFrame({
                    "cid": cid, "xcat": "RIRY",
                    "real_date": riry.index, "value": riry.values}))

        # Excess money growth: broad money YoY minus target minus GDP trend.
        money = get(cid, "MONEY") or get(cid, "MONEYALT")
        if money is not None:
            level, lag = money
            m_yoy = yoy(level.resample("ME").last().dropna(), 12)
            daily_m = to_daily(m_yoy.dropna(), lag, bdays) - target
            if gdp_trend is not None:
                daily_m = daily_m - gdp_trend
            elif cid == "USD":
                pass  # handled below with the US GDP trend present
            daily_m = daily_m.dropna()
            if len(daily_m) > 260:
                rows.append(pd.DataFrame({
                    "cid": cid, "xcat": "XMONGROWTH",
                    "real_date": daily_m.index, "value": daily_m.values}))

        # Survey levels and changes; OECD standardized indices center at 100.
        for family, stem in (("BCONF", "BCONF"), ("CCONF", "CCONF"), ("CLI", "CLI")):
            if (item := get(cid, family)) is not None:
                level, lag = item
                monthly = level.resample("ME").last().dropna()
                _emit(rows, cid, f"{stem}_LVL",
                      (monthly - 100.0).rolling(3).mean().dropna(), lag, bdays)
                _emit(rows, cid, f"{stem}_D3M", d3m3ml3(monthly).dropna(), lag, bdays)

        # Unemployment dynamics.
        if (item := get(cid, "UNEMP")) is not None:
            level, lag = item
            monthly = level.resample("ME").last().dropna()
            u3 = monthly.rolling(3).mean()
            _emit(rows, cid, "UNEMP_D12M", (u3 - u3.shift(12)).dropna(), lag, bdays)
            _emit(rows, cid, "UNEMP_V5YMM", vs_5y_median(u3, 12).dropna(), lag, bdays)

        # BIS private credit: the annual change in credit/GDP and the credit gap.
        if (item := get(cid, "CREDITGDP")) is not None:
            level, lag = item
            quarterly = level.resample("QE").last().dropna()
            _emit(rows, cid, "CREDITGDP_D4Q",
                  (quarterly - quarterly.shift(4)).dropna(), lag, bdays, "Q")
        if (item := get(cid, "CREDITGAP")) is not None:
            level, lag = item
            _emit(rows, cid, "CREDITGAP",
                  level.resample("QE").last().dropna(), lag, bdays, "Q")

        # Real effective exchange rate momentum and annual appreciation.
        if (item := get(cid, "REER")) is not None:
            level, lag = item
            monthly = level.resample("ME").last().dropna()
            _emit(rows, cid, "REER_YOY", yoy(monthly, 12).dropna(), lag, bdays)
            _emit(rows, cid, "REERAPPR_1M",
                  (100.0 * (monthly / monthly.shift(1) - 1.0)).dropna(), lag, bdays)
            mean5 = monthly.rolling(60, min_periods=48).mean()
            _emit(rows, cid, "REER_V5Y",
                  (100.0 * (monthly / mean5 - 1.0)).dropna(), lag, bdays)

        # Nominal GDP in USD, the basket weight; a year's figure is used once
        # published, about six months after year end (obs dated 1 January).
        if (item := get(cid, "NGDPUSD")) is not None:
            _emit(rows, cid, "NGDPUSD", item[0].dropna(), item[1], bdays, "A")

        # Term spread (bond carry proxy): 10-year yield minus 3-month rate.
        lt, st = get(cid, "LTYLD"), get(cid, "STYLD")
        if lt is not None and st is not None:
            spread = (lt[0].resample("ME").last() - st[0].resample("ME").last())
            _emit(rows, cid, "TERMSPRD", spread.dropna(), lt[1], bdays)

        # Current-account balance ratios: direct ratios, or US flow over GDP.
        if (item := get(cid, "CABGDP")) is not None:
            _emit(rows, cid, "CABGDP", item[0].dropna(), item[1], bdays, "Q")
        elif cid == "USD" and ngdp_usd is not None and (
            item := get(cid, "CABAL")
        ) is not None:
            level, lag = item
            quarterly_ca = level.resample("QE").last() / 1e3  # millions -> billions
            gdp_q = ngdp_usd.reindex(quarterly_ca.index).ffill()
            ratio = (100.0 * 4.0 * quarterly_ca / gdp_q).dropna()
            _emit(rows, cid, "CABGDP", ratio, lag, bdays, "Q")

    # US-only extras, emitted once.
    us_simple = {
        "DU05YETP": "DU05YETP", "INFE5Y": "INFE5YRAW",
        "INFE5Y5Y": "INFE5Y5YRAW", "FCI": "FCI", "SLOPE10Y2Y": "SLOPE10Y2Y",
        "SLOPE10Y3M": "SLOPE10Y3M", "CRSPRDBAA": "CRSPRDBAA",
        "GB02Y_YLD": "GB02Y_YLD", "GB05Y_YLD": "GB05Y_YLD", "GB10Y_YLD": "GB10Y_YLD",
    }
    for family, xcat in us_simple.items():
        if (item := get("USD", family)) is not None:
            _emit(rows, "USD", xcat, item[0].dropna(), item[1], bdays, "D")

    # Excess GDP nowcast: the Atlanta Fed nowcast minus the trailing GDP trend.
    us_gdp = get("USD", "RGDP")
    if us_gdp is not None and (item := get("USD", "GDPNOW")) is not None:
        trend = to_daily(
            trend_20q(yoy(us_gdp[0].resample("QE").last(), 4)).dropna(),
            us_gdp[1], bdays, "Q",
        )
        nowcast = to_daily(item[0].dropna(), item[1], bdays, "D")
        xnow = (nowcast - trend).dropna()
        if len(xnow) > 260:
            rows.append(pd.DataFrame({
                "cid": "USD", "xcat": "XGDPNOW",
                "real_date": xnow.index, "value": xnow.values}))

    target_us = INFLATION_TARGETS["USD"]
    for family, xcat in (("INFE5Y", "XINFE5Y"), ("INFE5Y5Y", "XINFE5Y5Y"),
                         ("HHINFE", "XHHINFE")):
        if (item := get("USD", family)) is not None:
            freq = "M" if family == "HHINFE" else "D"
            _emit(rows, "USD", xcat, (item[0] - target_us).dropna(),
                  item[1], bdays, freq)

    if (item := get("USD", "PCECORE")) is not None:
        level, lag = item
        _emit(rows, "USD", "XPCECORE",
              (yoy(level.resample("ME").last().dropna(), 12) - target_us).dropna(),
              lag, bdays)

    # US labor-tightness components.
    empl = get("USD", "EMPL")
    wforce = get("USD", "WFORCE")
    if empl is not None and wforce is not None:
        e_yoy3 = yoy(empl[0].resample("ME").last().dropna(), 12).rolling(3).mean()
        w_trend = yoy(wforce[0].resample("ME").last().dropna(), 12).rolling(
            60, min_periods=36).median()
        _emit(rows, "USD", "XEMPL", (e_yoy3 - w_trend).dropna(), empl[1], bdays)

    # US consumer sentiment dynamics (level is not zero-centered, changes are).
    if (item := get("USD", "CSENT")) is not None:
        level, lag = item
        monthly = level.resample("ME").last().dropna()
        _emit(rows, "USD", "CSENT_D3M", d3m3ml3(monthly).dropna(), lag, bdays)
        _emit(rows, "USD", "CSENT_V5YMM", vs_5y_median(monthly, 12).dropna(),
              lag, bdays)

    # Bank lending survey: net share tightening C&I standards (higher = tighter).
    for family, xcat in (("BLSTIGHT", "BLSTIGHT"), ("BLSTIGHTS", "BLSTIGHTS")):
        if (item := get("USD", family)) is not None:
            _emit(rows, "USD", xcat, item[0].dropna(), item[1], bdays, "Q")
    # Household debt service ratio, change over a year.
    if (item := get("USD", "HHDSR")) is not None:
        quarterly = item[0].resample("QE").last().dropna()
        _emit(rows, "USD", "HHDSR_D4Q", (quarterly - quarterly.shift(4)).dropna(),
              item[1], bdays, "Q")
    # Construction: permits and starts growth, 3m average over a year earlier.
    for family, xcat in (("PERMITS", "XPERMITS"), ("STARTS", "XSTARTS")):
        if (item := get("USD", family)) is not None:
            monthly = item[0].resample("ME").last().dropna()
            _emit(rows, "USD", xcat, yoy(monthly.rolling(3).mean(), 12).dropna(),
                  item[1], bdays)

    # US inventory cycle proxy: inventories-to-sales ratio change.
    if (item := get("USD", "ISRATIO")) is not None:
        level, lag = item
        monthly = level.resample("ME").last().dropna()
        _emit(rows, "USD", "ISRATIO_D3M", d3m3ml3(monthly).dropna(), lag, bdays)

    # US trade-balance trend over nominal GDP.
    if ngdp_usd is not None and (item := get("USD", "TRADEBAL")) is not None:
        level, lag = item
        monthly_tb = (12.0 * level.resample("ME").last() / 1e3).dropna()
        gdp_m = ngdp_usd.reindex(monthly_tb.index, method="ffill")
        ratio = (100.0 * monthly_tb / gdp_m).dropna()
        _emit(rows, "USD", "XBTREND", d3m3ml3(ratio).dropna(), lag, bdays)

    # US net international investment position trend over GDP.
    if ngdp_usd is not None and (item := get("USD", "NIIP")) is not None:
        level, lag = item
        quarterly = (level.resample("QE").last() / 1e3).dropna()
        gdp_q = ngdp_usd.reindex(quarterly.index).ffill()
        ratio = (100.0 * quarterly / gdp_q).dropna()
        _emit(rows, "USD", "NIIPTREND",
              (ratio - ratio.rolling(8, min_periods=4).mean()).dropna(),
              item[1], bdays, "Q")

    # US dollar broad index momentum.
    if (item := get("USD", "DXYBROAD")) is not None:
        level, lag = item
        monthly = level.resample("ME").last().dropna()
        _emit(rows, "USD", "DXY_MOM3M",
              (100.0 * (monthly / monthly.shift(3) - 1.0)).dropna(), lag, bdays)

    # Industrial metal price momentum: equal-weighted 3-month log momentum.
    metal_parts = []
    metal_lag = 30
    for family in ("PXCOPPER", "PXALUM", "PXZINC", "PXNICKEL"):
        if (item := get("USD", family)) is not None:
            monthly = item[0].resample("ME").last().dropna()
            metal_parts.append(np.log(monthly / monthly.shift(3)))
            metal_lag = item[1]
    if metal_parts:
        metal_mom = 100.0 * pd.concat(metal_parts, axis=1).mean(axis=1).dropna()
        _emit(rows, "USD", "METALPX_MOM3M", metal_mom, metal_lag, bdays)

    # FX carry proxies: covered interest parity makes a currency forward earn
    # roughly the short-rate gap to the dollar, so the policy-rate gap stands in
    # for forward carry; the real gap subtracts the inflation differential.
    tmp = pd.concat(rows, ignore_index=True)
    usd_pol = get("USD", "CASHRATE")
    if usd_pol is not None:
        us_rate = to_daily(usd_pol[0].dropna(), usd_pol[1], bdays, "D")
        wide_riry = tmp[tmp["xcat"] == "RIRY"].pivot(
            index="real_date", columns="cid", values="value")
        for cid in cids:
            pol = get(cid, "POLRATE")
            if cid == "USD" or pol is None:
                continue
            gap = (to_daily(pol[0].dropna(), pol[1], bdays, "D") - us_rate).dropna()
            if len(gap) > 260:
                rows.append(pd.DataFrame({
                    "cid": cid, "xcat": "FXCARRY",
                    "real_date": gap.index, "value": gap.values}))
            if cid in wide_riry and "USD" in wide_riry:
                rgap = (wide_riry[cid] - wide_riry["USD"]).dropna()
                if len(rgap) > 260:
                    rows.append(pd.DataFrame({
                        "cid": cid, "xcat": "RFXCARRY",
                        "real_date": rgap.index, "value": rgap.values}))

    out = pd.concat(rows, ignore_index=True)
    # Hyperinflation-era EM prints (Brazil before the 1994 Real plan, Turkey,
    # Romania, Peru) would dominate the expanding scoring scale for decades.
    early_em = out["cid"].isin(EM_CIDS) & (out["real_date"] < EM_START)
    out = out.loc[~early_em].reset_index(drop=True)
    assert not out.duplicated(["cid", "xcat", "real_date"]).any()
    assert np.isfinite(out["value"]).all()
    return out[QDF_COLS]


def bond_excess_from_yield(yld: pd.Series, cash: pd.Series, tenor: float = 5.0
                           ) -> pd.Series:
    """Daily excess return (%) of a constant-maturity par bond rebuilt from its yield.

    Return = yield accrual + price change from the yield move (modified duration
    and convexity of a semi-annual par bond at the prior yield) - cash accrual.
    This is the construction behind generic 5-year government bond returns.
    """
    y = yld.dropna() / 100.0
    days = y.index.to_series().diff().dt.days
    y0 = y.shift(1)
    n = 2 * tenor
    dur = (1.0 - (1.0 + y0 / 2.0) ** (-n)) / y0
    conv = dur ** 2 + dur / (1.0 + y0 / 2.0) / 2.0
    dy = y - y0
    carry = y0 * days / 365.0
    price = -dur * dy + 0.5 * conv * dy ** 2
    rate = cash.reindex(y.index, method="ffill").shift(1) / 100.0
    return (100.0 * (carry + price - rate * days / 360.0)).dropna()


def credit_excess_from_spread(spread: pd.Series, spread_duration: float = 5.0
                              ) -> pd.Series:
    """Daily excess return (%) of a credit spread position over duration-matched
    government bonds: spread accrual minus spread duration times the spread move."""
    s = spread.dropna() / 100.0
    days = s.index.to_series().diff().dt.days
    return (100.0 * (s.shift(1) * days / 365.0
                     - spread_duration * (s - s.shift(1)))).dropna()


# FRED daily spot rates for the nine developed currencies of the FX
# sleeve. Quotes are either USD per foreign unit (invert=False) or foreign
# units per USD (invert=True).
FX_SPOT = {
    "AUD": ("DEXUSAL", False), "CAD": ("DEXCAUS", True), "CHF": ("DEXSZUS", True),
    "EUR": ("DEXUSEU", False), "GBP": ("DEXUSUK", False), "JPY": ("DEXJPUS", True),
    "NOK": ("DEXNOUS", True), "NZD": ("DEXUSNZ", False), "SEK": ("DEXSDUS", True),
}
# Country ETFs for the equal-weighted equity index baskets.
COUNTRY_ETFS = {
    "DXEQ": ["EWA", "EWC", "EWL", "EZU", "EWU", "EWJ", "EWD"],
    "EMEQ": ["EWZ", "FXI", "INDA", "EWY", "EWW", "EWM", "EPOL", "EWS", "THD",
             "TUR", "EWT", "EZA"],
}


def fx_forward_excess(fred_wide: pd.DataFrame, policy: dict[str, pd.Series],
                      dff: pd.Series) -> pd.Series | None:
    """Equal-weighted long-DM-currency forward excess return (%), or None.

    A forward earns the spot move plus the short-rate gap (covered interest
    parity), so each leg is the spot return plus (foreign - US policy rate) accrual.
    """
    legs = []
    for cid, (code, invert) in FX_SPOT.items():
        if code not in fred_wide or cid not in policy:
            continue
        spot = fred_wide[code].dropna()
        usd_per_fcy = 1.0 / spot if invert else spot
        days = usd_per_fcy.index.to_series().diff().dt.days
        gap = (policy[cid].reindex(usd_per_fcy.index, method="ffill")
               - dff.reindex(usd_per_fcy.index, method="ffill")).shift(1)
        legs.append((100.0 * usd_per_fcy.pct_change() + gap * days / 360.0)
                    .rename(cid))
    if len(legs) < 3:
        return None
    panel = pd.concat(legs, axis=1)
    return panel.mean(axis=1)[panel.notna().sum(axis=1) >= 3].dropna()


def etf_basket_excess(closes: pd.DataFrame, symbols: list[str], dff: pd.Series
                      ) -> pd.Series | None:
    """Equal-weighted excess return (%) of the listed ETFs, live from three members."""
    have = [s for s in symbols if s in closes]
    if len(have) < 3:
        return None
    rets = 100.0 * closes[have].pct_change(fill_method=None)
    days = closes.index.to_series().diff().dt.days
    rate = dff.reindex(closes.index, method="ffill").shift(1)
    excess = rets.sub(rate * days / 360.0, axis=0)
    return excess.mean(axis=1)[excess.notna().sum(axis=1) >= 3].dropna()


# Sleeves built from yields as generic bond returns:
# USDU is the 5-year Treasury excess return (GB05YXR) and G2CR is credit excess
# on top of it (CRR = CRXR + GB05YXR). Both then start decades before the ETFs.
SYNTHETIC_SLEEVES = ("USDU", "G2CR")
BUILT_SLEEVES: list[str] = list(SYNTHETIC_SLEEVES)


def load_market(econ: Path = ECON, sleeves: str = "matched"
                ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (daily excess-return qdf XR_NSA, daily cash accrual, adjusted closes).

    sleeves="etf" uses the ETF proxy for every sleeve. sleeves="matched" (the
    default) builds USDU and G2CR from yields and spreads.
    """
    snapshot = latest_gathered(econ)
    yahoo = pd.read_parquet(snapshot / "yahoo.parquet")
    closes = (
        yahoo.pivot(index="as_of_date", columns="series_id", values="value")
        .sort_index()
    )
    fred = pd.read_parquet(snapshot / "fred.parquet")
    dff = (
        fred[fred["series_id"] == "DFF"]
        .set_index("as_of_date")["value"]
        .sort_index()
        .dropna()
    )
    frames = []
    cash_by_date = {}
    if sleeves == "matched":
        fred_wide = fred.pivot(index="as_of_date", columns="series_id",
                               values="value").sort_index()
        usdu = bond_excess_from_yield(fred_wide["DGS5"], dff)
        g2cr = usdu.add(credit_excess_from_spread(fred_wide["BAA10Y"]),
                        fill_value=np.nan).dropna()
        built = {"USDU": usdu, "G2CR": g2cr}
        # Optional sleeves, used only when the gatherer has pulled their inputs.
        policy = {}
        if any(code in fred_wide for code, _ in FX_SPOT.values()):
            raw = load_provider_raw(econ)
            pol = raw[raw["family"] == "POLRATE"]
            for cid, group in pol.groupby("cid"):
                policy[cid] = group.set_index("obs_date")["value"].sort_index(
                ).groupby(level=0).last()
        fx = fx_forward_excess(fred_wide, policy, dff)
        if fx is not None:
            built["DXFX"] = fx
        for sleeve, symbols in COUNTRY_ETFS.items():
            basket_xr = etf_basket_excess(closes, symbols, dff)
            if basket_xr is not None:
                built[sleeve] = basket_xr
        global BUILT_SLEEVES
        BUILT_SLEEVES = sorted(built)
        for sleeve, series in built.items():
            series = series[series.index >= "1990-01-01"]
            frames.append(pd.DataFrame({
                "cid": sleeve, "xcat": "XR_NSA",
                "real_date": series.index, "value": series.values}))
    for sleeve, (symbol, _) in SLEEVES.items():
        if sleeves == "matched" and sleeve in BUILT_SLEEVES:
            continue
        prices = closes[symbol].dropna()
        returns = 100.0 * prices.pct_change()
        # Cash accrual between consecutive trading dates: act/360 on prior-day DFF.
        dates = prices.index
        gaps = dates.to_series().diff().dt.days
        rate = dff.reindex(dates, method="ffill").shift(1)
        accrual = (rate * gaps / 360.0).fillna(0.0)
        excess = (returns - accrual).dropna()
        cash_by_date.update(accrual.to_dict())
        frames.append(pd.DataFrame({
            "cid": sleeve, "xcat": "XR_NSA",
            "real_date": excess.index, "value": excess.values,
        }))
    xr = pd.concat(frames, ignore_index=True)
    all_days = pd.DatetimeIndex(sorted(xr["real_date"].unique()))
    gaps = all_days.to_series().diff().dt.days
    cash = (dff.reindex(all_days, method="ffill").shift(1) * gaps / 360.0).fillna(0.0)
    return xr, cash, closes
