"""Gather allocation research inputs without changing frozen notebook data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import pandas as pd
import yfinance as yf
from openpyxl import Workbook, load_workbook
from yfinance.exceptions import YFException

MARKET = Path.home() / "OneDrive" / "Investment" / "Market Data"
FRED_IDS = """
CES0500000003 CPIAUCNS CPIAUCSL CPILFESL CSUSHPINSA DFF DFII5 DGS10 DGS2 DGS5
DTB3 GDPC1 GDPNOW INDPRO M2SL MICH NFCI PAYEMS PCEPILFE PPIACO RRSFS T10YIE
T5YIE THREEFYTP10 UMCSENT UNRATE BAMLC0A0CM BAMLH0A0HYM2 BAMLEMCBPIOAS BAA10Y
AAA10Y THREEFYTP5 T5YIFR T10Y3M T10Y2Y DTWEXBGS IIPUSNETIQ IIPUSASSQ IEABC
BOPGSTB FYFSGDA188S GGNLBAUSA188N GDPPOT CLF16OV USSTHPI BUSINV ISRATIO PCOPPUSDM
PALUMUSDM PZINCUSDM PNICKUSDM MKTGDPUSA646NWDB MKTGDPCNA646NWDB MABMM301JPM189S
MABMM301EZM189S MABMM301GBM189S BRABCAGDPBP6 CHNB6BLTT02STSAQ INDB6BLTT02STSAQ
GGNLBAEZA188N GGNLBAJPA188N
""".split()
SYMBOLS = "SPY EFA EEM IEI BWX UDN LQD EMB EMLC DBB GLD".split()
MANUAL_COLUMNS = (
    "series_id",
    "as_of_date",
    "available_date",
    "value",
    "unit",
    "source",
)
PROVIDERS = ("fred", "eia", "yahoo", "manual")
REQUESTS = Path(__file__).with_name("requests.csv")


def load_requests(path: Path = REQUESTS) -> dict[str, list[str]]:
    """Extra series to pull, one per row: provider,identifier,purpose.

    Core series above must succeed; a requested series that fails is skipped and
    listed in the manifest, so one bad identifier never blocks a refresh.
    """
    out: dict[str, list[str]] = {"fred": [], "yahoo": []}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            provider = row["provider"].strip().lower()
            identifier = row["identifier"].strip()
            if provider not in out:
                raise GatherError(f"{path.name}: unsupported provider {provider!r}")
            if not re.fullmatch(r"[A-Za-z0-9_.=^-]+", identifier):
                raise GatherError(f"{path.name}: invalid identifier {identifier!r}")
            if identifier not in out[provider]:
                out[provider].append(identifier)
    return out


class GatherError(Exception):
    """Public diagnostic; never contains an API key or API response body."""


def api_key(provider: str, workbook: Path) -> str:
    env = os.getenv(f"{provider.upper()}_API_KEY", "").strip()
    if env:
        return env
    with workbook.open("rb") as stream:
        book = load_workbook(stream, read_only=True, data_only=True)
        try:
            found = {
                row[1].strip()
                for sheet in book
                for row in sheet.iter_rows(min_col=1, max_col=2, values_only=True)
                if str(row[0]).strip().casefold() == provider
                and isinstance(row[1], str)
                and row[1].strip()
            }
        finally:
            book.close()
    if len(found) != 1:
        raise GatherError(f"{provider}: expected one credential in the key workbook")
    return found.pop()


def api_json(client: httpx.Client, url: str, label: str, params: dict) -> dict:
    for attempt in range(3):
        try:
            response = client.get(url, params=params)
        except httpx.RequestError as exc:
            if attempt == 2:
                raise GatherError(f"{label}: {type(exc).__name__}") from None
        else:
            if response.is_success:
                try:
                    return response.json()
                except ValueError:
                    raise GatherError(f"{label}: response was not JSON") from None
            if response.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise GatherError(f"{label}: HTTP {response.status_code}")
        time.sleep(2**attempt)
    raise AssertionError("retry loop did not return")


def fred(client: httpx.Client, key: str, sid: str) -> pd.DataFrame:
    base = "https://api.stlouisfed.org/fred/series"
    params = {"api_key": key, "file_type": "json", "series_id": sid}
    label = f"FRED {sid}"
    meta = api_json(client, base, label, params)["seriess"][0]
    time.sleep(0.55)
    offset, rows = 0, []
    while True:
        page = api_json(
            client, f"{base}/observations", label, {**params, "offset": offset}
        )
        batch, total = page["observations"], int(page["count"])
        if not batch:
            raise GatherError(f"{label}: empty observations page at {offset}/{total}")
        rows.extend(batch)
        offset += len(batch)
        time.sleep(0.55)
        if offset >= total:
            break
    frame = pd.DataFrame(rows).rename(columns={"date": "as_of_date"})
    frame["value"] = frame["value"].replace(".", None)
    frame["unit"] = meta["units"]
    frame["frequency"] = meta["frequency_short"]
    frame["available_date"] = pd.NaT
    return frame


def eia_plan(catalog: Path) -> tuple[list[tuple[str, str]], list[str]]:
    plan, skipped = [], []
    with catalog.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            agency = (
                "NG"
                if row["group"] == "ng_weekly"
                or row["name"].startswith(("ng_", "henryhub"))
                else "PET"
            )
            sid = f"{agency}.{row['series_id']}.{row['freq']}"
            if not re.fullmatch(r"[A-Z0-9._-]+", sid):
                raise GatherError("EIA catalog has an invalid series ID")
            if row["verified_id"].lower() == "true":
                plan.append((sid, row["name"]))
            else:
                skipped.append(sid)
    if not plan or len({sid for sid, _ in plan}) != len(plan):
        raise GatherError("EIA catalog has no verified IDs or has duplicate IDs")
    return plan, skipped


def eia(client: httpx.Client, key: str, sid: str) -> pd.DataFrame:
    rows, offset = [], 0
    while True:
        response = api_json(
            client,
            f"https://api.eia.gov/v2/seriesid/{sid}",
            f"EIA {sid}",
            {"api_key": key, "length": 5000, "offset": offset},
        )["response"]
        batch, total = response["data"], int(response["total"])
        if not batch:
            raise GatherError(f"EIA {sid}: empty observations page at {offset}/{total}")
        rows.extend(batch)
        offset += len(batch)
        if offset >= total:
            break
    frame = pd.DataFrame(rows)[["period", "value", "units"]].rename(
        columns={"period": "as_of_date", "units": "unit"}
    )
    frame["available_date"] = pd.NaT
    return frame


_YAHOO_SESSION = None


def yahoo_session():
    """curl_cffi session whose CA bundle sits on an ASCII path.

    libcurl on Windows cannot open a certificate file whose path holds
    non-ASCII characters (a virtual environment under OneDrive/文件 does), and
    yfinance then fails every request with UnicodeEncodeError. The certifi
    bundle is copied once to an ASCII folder and passed explicitly.
    """
    global _YAHOO_SESSION
    if _YAHOO_SESSION is None:
        import shutil
        import tempfile

        import certifi
        from curl_cffi import requests as curl_requests

        bundle = Path(certifi.where())
        if not str(bundle).isascii():
            folder = Path(tempfile.gettempdir())
            if not str(folder).isascii():
                folder = Path(os.environ.get("PUBLIC", "C:/Users/Public"))
            target = folder / "yf_cacert.pem"
            if not target.exists() or target.stat().st_size != bundle.stat().st_size:
                shutil.copyfile(bundle, target)
            bundle = target
        _YAHOO_SESSION = curl_requests.Session(impersonate="chrome",
                                               verify=str(bundle))
    return _YAHOO_SESSION


def yahoo(symbol: str) -> pd.DataFrame:
    try:
        data = yf.Ticker(symbol, session=yahoo_session()).history(
            period="max", auto_adjust=False, actions=True, raise_errors=True
        )
    except (YFException, OSError, ValueError) as exc:
        # Library errors can include internal URLs; report the symbol and failure class.
        raise GatherError(f"Yahoo {symbol}: {type(exc).__name__}") from None
    required = ["Close", "Adj Close", "Dividends", "Stock Splits"]
    if data.empty or not set(required).issubset(data.columns):
        raise GatherError(
            f"Yahoo {symbol}: missing adjusted prices or corporate actions"
        )
    frame = data[required].rename(
        columns={
            "Close": "unadjusted_close",
            "Adj Close": "value",
            "Dividends": "dividend",
            "Stock Splits": "split_ratio",
        }
    )
    if "Capital Gains" in data.columns:
        frame["capital_gain"] = data["Capital Gains"]
    frame["as_of_date"] = data.index.strftime("%Y-%m-%d")
    frame["unit"] = "USD/share"
    frame["available_date"] = pd.NaT
    # Do not publish a possibly incomplete current-session bar.
    frame = frame[frame["as_of_date"] < datetime.now(UTC).date().isoformat()]
    # yfinance can return a row with no prices at all, for example a dividend
    # or split date that falls outside a trading session. Such rows carry no
    # price, so they are dropped; a priced row with a bad value still fails.
    no_price = frame["value"].isna() & frame["unadjusted_close"].isna()
    frame = frame.loc[~no_price]
    bad = frame["value"].isna() | frame["value"].le(0)
    if frame.empty or bad.any():
        first = frame.loc[bad, "as_of_date"].head(3).tolist() if bad.any() else []
        raise GatherError(
            f"Yahoo {symbol}: invalid adjusted prices on {int(bad.sum())} rows "
            f"(first: {first})"
        )
    return frame.copy()


def validate(frame: pd.DataFrame, keys: list[str], label: str) -> pd.DataFrame:
    frame = frame.dropna(subset=["value"]).copy()
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    frame["as_of_date"] = pd.to_datetime(
        frame["as_of_date"], errors="raise"
    ).dt.normalize()
    frame["available_date"] = pd.to_datetime(
        frame["available_date"], errors="raise"
    ).dt.normalize()
    if (
        frame.empty
        or frame["as_of_date"].isna().any()
        or frame.duplicated(keys).any()
        or not frame["value"].map(math.isfinite).all()
    ):
        raise GatherError(f"{label}: empty, duplicate or invalid observations")
    early = frame["available_date"].notna() & frame["available_date"].lt(
        frame["as_of_date"]
    )
    if early.any():
        raise GatherError(f"{label}: available_date before as_of_date")
    return frame.sort_values(keys).reset_index(drop=True)


def manual_inputs(directory: Path) -> tuple[pd.DataFrame, list[dict]]:
    directory.mkdir(parents=True, exist_ok=True)
    template = directory / "Template.xlsx"
    if not template.exists():
        book = Workbook()
        sheet = book.active
        assert sheet is not None
        sheet.title = "Observations"
        sheet.append(MANUAL_COLUMNS)
        book.save(template)
        book.close()
    frames, files = [], []
    for path in sorted(directory.glob("*.xlsx")):
        if path.name == template.name or path.name.startswith("~$"):
            continue
        try:
            frame = pd.read_excel(path, sheet_name="Observations", engine="openpyxl")
        except ValueError as exc:
            raise GatherError(f"{path.name}: cannot read Observations sheet") from exc
        if not set(MANUAL_COLUMNS).issubset(frame.columns):
            raise GatherError(f"{path.name}: missing columns; use Template.xlsx")
        frame = frame[list(MANUAL_COLUMNS)].dropna(how="all")
        for column in ("series_id", "unit", "source"):
            frame[column] = frame[column].astype("string").str.strip()
            if frame[column].isna().any() or frame[column].eq("").any():
                raise GatherError(f"{path.name}: blank {column}")
        if frame["value"].isna().any():
            raise GatherError(f"{path.name}: blank value")
        if not frame.empty:
            frame = validate(
                frame, ["series_id", "as_of_date", "available_date"], path.name
            )
            frame["workbook"] = path.name
            frames.append(frame)
        files.append({"file": path.name, "sha256": sha256(path), "rows": len(frame)})
    if not frames:
        empty = pd.DataFrame(
            {
                name: pd.Series(
                    dtype="datetime64[ns]"
                    if name.endswith("_date")
                    else "float64"
                    if name == "value"
                    else "string"
                )
                for name in (*MANUAL_COLUMNS, "workbook")
            }
        )
        return empty, files
    data = validate(
        pd.concat(frames, ignore_index=True),
        ["series_id", "as_of_date", "available_date"],
        "Manual workbooks",
    )
    if data.groupby("series_id")["unit"].nunique().gt(1).any():
        raise GatherError("Manual workbooks: inconsistent units within a series_id")
    return data, files


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def gather(market: Path, providers: set[str], extra_fred: list[str],
           extra_yahoo: list[str] | None = None) -> Path:
    econ = market / "Econ"
    parent = econ / "Gathered"
    parent.mkdir(parents=True, exist_ok=True)
    run_at = datetime.now(UTC)
    final = parent / run_at.strftime("%Y%m%dT%H%M%S%fZ")
    manifest: dict = {
        "retrieved_utc": run_at.isoformat(),
        "providers": sorted(providers),
        "point_in_time_safe": False,
        "outputs": [],
        "requested_skipped": [],
    }
    requested = load_requests()

    def save(provider: str, frame: pd.DataFrame) -> None:
        path = stage / f"{provider}.parquet"
        frame.to_parquet(path, index=False)
        manifest["outputs"].append(
            {
                "file": path.name,
                "rows": len(frame),
                "sha256": sha256(path),
                "series": sorted(frame["series_id"].unique().tolist()),
            }
        )
        print(f"{provider}: {len(frame):,} rows", flush=True)

    with TemporaryDirectory(prefix=".staging-", dir=parent) as temp:
        stage = Path(temp)
        if "manual" in providers:
            data, files = manual_inputs(econ / "Manual")
            manifest["manual_workbooks"] = files
            save("manual", data)
        with httpx.Client(timeout=45, follow_redirects=False) as client:
            for provider in ("fred", "eia"):
                if provider not in providers:
                    continue
                key = api_key(provider, market / "API Key.xlsx")
                optional: set[str] = set()
                if provider == "fred":
                    core = list(dict.fromkeys(FRED_IDS + extra_fred))
                    optional = {s for s in requested["fred"] if s not in core}
                    plan = [(sid, sid) for sid in core + sorted(optional)]
                    fetch = fred
                else:
                    plan, skipped = eia_plan(
                        market / "Commodity" / "EIA" / "eia_catalog.csv"
                    )
                    manifest["eia_unverified_ids_not_requested"] = skipped
                    fetch = eia
                frames = []
                for sid, name in plan:
                    try:
                        frames.append(
                            validate(
                                fetch(client, key, sid), ["as_of_date"],
                                f"{provider} {sid}",
                            ).assign(series_id=sid, series_name=name)
                        )
                    except GatherError as exc:
                        if sid not in optional:
                            raise
                        manifest["requested_skipped"].append(
                            {"provider": provider, "id": sid, "reason": str(exc)})
                        print(f"skipped requested {provider} {sid}: {exc}", flush=True)
                save(provider, pd.concat(frames, ignore_index=True))
        if "yahoo" in providers:
            optional = [s for s in dict.fromkeys(requested["yahoo"]
                                                 + (extra_yahoo or []))
                        if s not in SYMBOLS]
            frames = []
            for symbol in SYMBOLS + optional:
                try:
                    frames.append(
                        validate(yahoo(symbol), ["as_of_date"], f"Yahoo {symbol}")
                        .assign(series_id=symbol))
                except GatherError as exc:
                    if symbol in SYMBOLS:
                        raise
                    manifest["requested_skipped"].append(
                        {"provider": "yahoo", "id": symbol, "reason": str(exc)})
                    print(f"skipped requested Yahoo {symbol}: {exc}", flush=True)
            save("yahoo", pd.concat(frames, ignore_index=True))
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        stage.rename(final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-root", type=Path, default=MARKET)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--catalog", action="store_true", help="Build discovery metadata"
    )
    modes.add_argument("--find", help="Search catalog using a feature description")
    modes.add_argument(
        "--match-notebook", type=Path, help="Shortlist for notebook recipes"
    )
    parser.add_argument(
        "--fred-catalog-db", type=Path, help="FRED metadata SQLite snapshot"
    )
    parser.add_argument("--match-output", type=Path, help="Save matches to Parquet")
    parser.add_argument("--only", choices=PROVIDERS, nargs="+")
    parser.add_argument(
        "--fred-id", action="append", default=[], help="Additional FRED ID"
    )
    parser.add_argument(
        "--symbol", action="append", default=[],
        help="Additional Yahoo Finance symbol (also see requests.csv)",
    )
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9_.=^-]+", sym) for sym in args.symbol):
        parser.error("--symbol must be a Yahoo Finance symbol")
    if args.only and "yahoo" not in args.only and args.symbol:
        parser.error("--symbol requires the Yahoo stage")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", sid) for sid in args.fred_id):
        parser.error("--fred-id must be a FRED series identifier")
    if args.only and "fred" not in args.only and args.fred_id:
        parser.error("--fred-id requires the FRED stage")
    if (args.catalog or args.find is not None or args.match_notebook) and (
        args.only or args.fred_id or args.symbol
    ):
        parser.error("--only and --fred-id are observation-gathering options")
    if args.fred_catalog_db and not args.catalog:
        parser.error("--fred-catalog-db requires --catalog")
    if args.match_output and not (args.find is not None or args.match_notebook):
        parser.error("--match-output requires a matching mode")
    # httpx logs query parameters at INFO, including provider API keys.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        if args.catalog or args.find is not None or args.match_notebook:
            import econ_catalog

            econ = args.market_root / "Econ"
            if args.catalog:
                folder = econ_catalog.build_catalog(
                    econ, args.fred_catalog_db, manual_inputs
                )
            else:
                features = (
                    econ_catalog.extract_features(args.match_notebook)
                    if args.match_notebook
                    else [("query", args.find)]
                )
                catalog = econ_catalog.latest_catalog(econ)
                found = econ_catalog.match_catalog(catalog, features)
                if args.match_output:
                    found.to_parquet(args.match_output, index=False)
                    print(f"Saved {len(found)} rows to {args.match_output}")
                else:
                    print(
                        found[
                            ["feature", "provider", "identifier", "name", "status"]
                        ].to_string(index=False)
                    )
                return
        else:
            folder = gather(args.market_root, set(args.only or PROVIDERS),
                            args.fred_id, args.symbol)
    except (GatherError, OSError, ValueError, httpx.HTTPError) as exc:
        parser.exit(1, f"Gather failed: {exc}\n")
    print(f"Published {folder}")


if __name__ == "__main__":
    main()
