"""Metadata sources and offline matching for gather_econ_data."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import re
import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from urllib.parse import urljoin, urlsplit
from zipfile import ZipFile

import duckdb
import httpx
import pandas as pd

EIA_MANIFEST = "https://www.eia.gov/opendata/bulk/manifest.txt"
JPX = "https://www.jpx.co.jp/english/markets/statistics-equities/misc/01.html"
HKEX = (
    "https://www.hkex.com.hk/-/media/HKEX-Market/Services/Trading/Securities/"
    "Securities-Lists/Securities-Using-Standard-Transfer-Form-(including-GEM)"
    "-By-Stock-Code-Order/secstkorder.xls"
)
COLUMNS = [
    "provider",
    "identifier",
    "name",
    "description",
    "kind",
    "frequency",
    "unit",
    "observation_start",
    "observation_end",
    "source_updated",
    "source",
    "exchange",
    "exchange_symbol",
    "availability",
]


def frame(provider: str, rows) -> pd.DataFrame:
    data = pd.DataFrame(rows).assign(provider=provider).reindex(columns=COLUMNS)
    if data.empty or data[["identifier", "name"]].isna().any().any():
        raise ValueError(f"{provider}: empty catalog or missing identifier/name")
    return data.astype("string")


def fred_metadata(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = pd.read_sql_query(
            """SELECT id AS identifier, title AS name, frequency, units AS unit,
                      observation_start, observation_end, last_updated AS source_updated
               FROM series""",
            db,
        )
        expected, releases = db.execute(
            "SELECT count(DISTINCT series_id), count(DISTINCT release_id) "
            "FROM release_series"
        ).fetchone()
        if (
            len(rows) != expected
            or releases != db.execute("SELECT count(*) FROM releases").fetchone()[0]
        ):
            raise ValueError("FRED release membership does not reconcile")
    return frame(
        "fred",
        rows.assign(
            kind="macro_series",
            source="FRED public release inventory",
            availability="release_listed",
        ),
    )


def download(client: httpx.Client, url: str) -> httpx.Response:
    for attempt in range(3):
        try:
            response = client.get(url)
            response.raise_for_status()
            return response
        except httpx.HTTPError:
            if attempt == 2:
                raise
    raise AssertionError("unreachable")


def eia_archive(client: httpx.Client, dataset: str, url: str) -> pd.DataFrame:
    if urlsplit(url).hostname != "www.eia.gov":
        raise ValueError(f"EIA {dataset}: unexpected bulk host")
    for attempt in range(3):
        with TemporaryFile() as staging:
            try:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        staging.write(chunk)
            except httpx.HTTPError:
                if attempt == 2:
                    raise
                continue
            with ZipFile(staging) as archive:
                rows = []
                for member in archive.namelist():
                    if not member.endswith(".txt"):
                        continue
                    with archive.open(member) as lines:
                        for line in lines:
                            if b'"series_id"' not in line:
                                continue
                            item = json.loads(line)
                            if "series_id" not in item:
                                continue
                            rows.append(
                                {
                                    "identifier": item["series_id"],
                                    "name": item["name"],
                                    "description": item.get("description"),
                                    "frequency": item.get("f"),
                                    "unit": item.get("units"),
                                    "observation_start": item.get("start"),
                                    "observation_end": item.get("end"),
                                    "source_updated": item.get("last_updated"),
                                    "source": url,
                                    "kind": f"bulk_series:{dataset}",
                                    "availability": "eia_bulk_listed_not_api_verified",
                                }
                            )
            out = frame("eia", rows)
            print(f"EIA {dataset}: {len(out):,} identifiers", flush=True)
            return out
    raise AssertionError("unreachable")


def eia_metadata(client: httpx.Client) -> tuple[pd.DataFrame, dict]:
    datasets = download(client, EIA_MANIFEST).json()["dataset"]
    if not datasets:
        raise ValueError("Empty EIA bulk manifest")
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {
            name: pool.submit(eia_archive, client, name, item["accessURL"])
            for name, item in sorted(datasets.items())
        }
        sources = {name: job.result() for name, job in jobs.items()}
    data = pd.concat(sources.values(), ignore_index=True)
    return (
        data.sort_values(
            "source_updated", ascending=False, na_position="last"
        ).drop_duplicates("identifier"),
        {name: len(df) for name, df in sources.items()},
    )


def exchange_rows(items, exchange: str, url: str, suffix: str):
    for symbol, name, kind in items:
        if pd.isna(symbol) or pd.isna(name):
            continue
        symbol = str(symbol).strip()
        if not symbol or not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]*", symbol):
            continue
        yield {
            "identifier": symbol.replace(".", "-") + suffix,
            "exchange_symbol": symbol,
            "name": str(name).strip(),
            "kind": kind,
            "source": url,
            "exchange": exchange,
            "availability": "exchange_derived_yahoo_candidate",
        }


def exchange_directory(client: httpx.Client, venue: str):
    if venue in {"Nasdaq", "Other US"}:
        file = "nasdaqlisted.txt" if venue == "Nasdaq" else "otherlisted.txt"
        url = f"https://www.nasdaqtrader.com/dynamic/SymDir/{file}"
        rows = csv.DictReader(io.StringIO(download(client, url).text), delimiter="|")
        cols = "Symbol" if venue == "Nasdaq" else "ACT Symbol"
        items = (
            (
                r[cols],
                r["Security Name"],
                "etf" if r["ETF"] == "Y" else "listed_security",
            )
            for r in rows
            if r["Test Issue"] == "N"
        )
        return list(exchange_rows(items, venue, url, ""))
    if venue == "NSE":
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        data = pd.read_csv(io.BytesIO(download(client, url).content), dtype=str)
        return list(
            exchange_rows(
                zip(
                    data["SYMBOL"],
                    data["NAME OF COMPANY"],
                    ["equity"] * len(data),
                    strict=True,
                ),
                venue,
                url,
                ".NS",
            )
        )
    if venue == "ASX":
        url = "https://www.asx.com.au/asx/research/ASXListedCompanies.csv"
        data = pd.read_csv(
            io.BytesIO(download(client, url).content),
            skiprows=1,
            dtype=str,
        )
        return list(
            exchange_rows(
                zip(
                    data["ASX code"],
                    data["Company name"],
                    ["listed_company"] * len(data),
                    strict=True,
                ),
                venue,
                url,
                ".AX",
            )
        )
    if venue == "JPX":
        link = re.search(r'href="([^"]+/data_e\.xlsx)"', download(client, JPX).text)
        if link is None:
            raise ValueError("JPX: listed issues workbook was not found")
        url = urljoin(JPX, link.group(1))
        data = pd.read_excel(io.BytesIO(download(client, url).content), dtype=str)
        return list(
            exchange_rows(
                zip(
                    data["Local Code"],
                    data["Name (English)"],
                    data["Section/Products"],
                    strict=True,
                ),
                venue,
                url,
                ".T",
            )
        )
    if venue == "HKEX":
        raw = pd.read_excel(io.BytesIO(download(client, HKEX).content), header=None)
        headings = raw.index[
            raw.iloc[:, 0].astype(str).str.strip().str.casefold().eq("stock code")
        ]
        if len(headings) != 1:
            raise ValueError("HKEX: stock code header not found")
        items = (
            (str(code).strip().zfill(4), name, "listed_security")
            for code, name, *_ in raw.iloc[headings[0] + 1 :].itertuples(
                index=False, name=None
            )
            if re.fullmatch(r"\d{1,5}", str(code).strip())
        )
        return list(exchange_rows(items, venue, HKEX, ".HK"))
    raise ValueError(f"Unsupported exchange directory: {venue}")


MARKET_SYMBOLS = {
    "^GSPC": ("S&P 500", "index"),
    "^VIX": ("Cboe Volatility Index", "index"),
    "^FTSE": ("FTSE 100", "index"),
    "^GDAXI": ("DAX", "index"),
    "^FCHI": ("CAC 40", "index"),
    "^N225": ("Nikkei 225", "index"),
    "^HSI": ("Hang Seng Index", "index"),
    "^STOXX50E": ("EURO STOXX 50", "index"),
    "^GSPTSE": ("S&P/TSX Composite", "index"),
    "^AXJO": ("S&P/ASX 200", "index"),
    "^NSEI": ("Nifty 50", "index"),
    "^BVSP": ("Ibovespa", "index"),
    "EURUSD=X": ("Euro to US Dollar", "fx"),
    "JPY=X": ("US Dollar to Japanese Yen", "fx"),
    "GBPUSD=X": ("British Pound to US Dollar", "fx"),
    "AUDUSD=X": ("Australian Dollar to US Dollar", "fx"),
    "CL=F": ("WTI Crude Oil futures", "futures"),
    "BZ=F": ("Brent Crude Oil futures", "futures"),
    "NG=F": ("Natural Gas futures", "futures"),
    "HG=F": ("Copper futures", "futures"),
    "GC=F": ("Gold futures", "futures"),
    "SI=F": ("Silver futures", "futures"),
    "DX-Y.NYB": ("US Dollar Index", "index"),
}


def yahoo_metadata(
    client: httpx.Client, observed: set[str]
) -> tuple[pd.DataFrame, dict]:
    venues = ("Nasdaq", "Other US", "NSE", "ASX", "JPX", "HKEX")
    sources = {}
    with ThreadPoolExecutor(max_workers=len(venues)) as pool:
        jobs = {
            venue: pool.submit(exchange_directory, client, venue) for venue in venues
        }
        for venue, job in jobs.items():
            rows = job.result()
            if not rows:
                raise ValueError(f"{venue}: directory returned no identifiers")
            sources[venue] = rows
    data = frame(
        "yahoo",
        [row for rows in sources.values() for row in rows]
        + [
            {
                "identifier": sid,
                "name": title,
                "kind": kind,
                "source": "Research symbol list",
                "availability": "reference_symbol_not_rechecked",
            }
            for sid, (title, kind) in MARKET_SYMBOLS.items()
        ],
    ).drop_duplicates("identifier")
    data.loc[data["identifier"].isin(observed), "availability"] = (
        "yahoo_prices_downloaded"
    )
    return data, {venue: len(rows) for venue, rows in sources.items()}


def extract_features(path: Path) -> list[tuple[str, str]]:
    notebook = json.loads(path.read_text(encoding="utf-8-sig"))
    recipes = {}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        code = "".join(cell.get("source", []))
        try:
            tree = ast.parse(code)
        except SyntaxError:
            if "dict_recipes.update" in code or "theory_recipes =" in code:
                raise
            continue
        for node in ast.walk(tree):
            registry = None
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name)
                and target.id in {"dict_recipes", "theory_recipes"}
                for target in node.targets
            ):
                registry = node.value
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "dict_recipes"
                and node.args
            ):
                registry = node.args[0]
            if not isinstance(registry, ast.Dict):
                continue
            for key, value in zip(registry.keys, registry.values, strict=True):
                if not (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and isinstance(value, ast.Dict)
                ):
                    continue
                for field, text in zip(value.keys, value.values, strict=True):
                    if (
                        isinstance(field, ast.Constant)
                        and field.value == "concept"
                        and isinstance(text, ast.Constant)
                        and isinstance(text.value, str)
                    ):
                        recipes.setdefault(key.value, text.value)
    if not recipes:
        raise ValueError(
            "No literal Macrosynergy recipes found; use --find with a feature name"
        )
    return list(recipes.items())


def match_catalog(path: Path, features: list[tuple[str, str]]) -> pd.DataFrame:
    targets = pd.DataFrame(features, columns=["feature", "concept"])
    with duckdb.connect() as db:
        db.register("targets", targets)
        return db.execute(
            """
            WITH target_terms AS (
              SELECT DISTINCT feature, unnest(regexp_extract_all(
                  lower(concept), '[a-z][a-z0-9]{2,}')) AS term FROM targets
            ), clean_terms AS (
              SELECT * FROM target_terms WHERE term NOT IN
                  ('and','the','for','from','with','relative','excess','composite',
                   'change','score','level','growth','trend')
              UNION
              SELECT feature, '#' || lower(trim(concept)) FROM targets
            ), weights AS (
              SELECT feature, greatest(count_if(NOT starts_with(term, '#')), 1) AS terms
              FROM clean_terms GROUP BY feature
            ), instruments AS (
              SELECT provider, identifier, name, kind, unit, frequency,
                  local_observations,
                  observation_start, observation_end, availability,
                  unnest(list_distinct(list_append(regexp_extract_all(lower(name),
                      '[a-z][a-z0-9]{2,}'), '#' || lower(identifier)))) AS term
              FROM read_parquet(?)
            ), scores AS (
              SELECT i.* EXCLUDE(term), t.feature,
                count_if(NOT starts_with(i.term, '#')) AS matched_terms,
                bool_or(starts_with(i.term, '#')) AS exact_identifier,
                string_agg(i.term, ', ' ORDER BY i.term)
                    FILTER (WHERE NOT starts_with(i.term, '#')) AS matching_words
              FROM clean_terms t JOIN instruments i USING(term) GROUP BY ALL
            ), ranked AS (
              SELECT s.*, row_number() OVER (PARTITION BY feature, provider
                  ORDER BY exact_identifier DESC, matched_terms DESC,
                      local_observations DESC,
                      regexp_matches(kind, '^bulk_series:(AEO|IEO|STEO)'),
                      length(name), identifier) AS rank
              FROM scores s
            )
            SELECT t.*, r.* EXCLUDE(feature),
                coalesce(matched_terms * 1.0 / w.terms, 0) AS term_coverage,
                CASE WHEN identifier IS NULL THEN 'no_candidate'
                     ELSE 'candidate_requires_review' END AS status
            FROM targets t LEFT JOIN weights w USING(feature)
            LEFT JOIN ranked r ON t.feature = r.feature AND r.rank <= 3
            ORDER BY t.feature, exact_identifier DESC NULLS LAST, term_coverage DESC,
                local_observations DESC, provider, rank
            """,
            [str(path)],
        ).df()


def latest_catalog(econ: Path) -> Path:
    paths = sorted((econ / "Catalog").glob("20*/series_catalog.parquet"))
    if not paths:
        raise ValueError("No compatible catalog found; run with --catalog first")
    return paths[-1]


def observed_series(econ: Path) -> dict[str, set[str]]:
    known: dict[str, set[str]] = {}
    for path in sorted((econ / "Gathered").glob("20*/manifest.json")):
        for item in json.loads(path.read_text(encoding="utf-8"))["outputs"]:
            if (path.parent / item["file"]).is_file():
                known.setdefault(Path(item["file"]).stem, set()).update(item["series"])
    return known


def build_catalog(
    econ: Path,
    fred_db: Path | None,
    manual_inputs: Callable[[Path], tuple[pd.DataFrame, list[dict]]],
) -> Path:
    parent = econ / "Catalog"
    parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    final = parent / now.strftime("%Y%m%dT%H%M%S%fZ")
    if fred_db is None:
        copies = sorted((econ / "FRED_Catalog").glob("*/catalog.sqlite"))
        if not copies:
            raise ValueError("No FRED release inventory; supply --fred-catalog-db")
        fred_db = copies[-1]
    with fred_db.open("rb") as stream:
        fred_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    fred = fred_metadata(fred_db)
    known = observed_series(econ)
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        yahoo, exchanges = yahoo_metadata(client, known.get("yahoo", set()))
        eia, datasets = eia_metadata(client)
    manual, files = manual_inputs(econ / "Manual")
    parts = [fred, eia, yahoo]
    if not manual.empty:
        manual_rows = []
        for identifier, rows in manual.groupby("series_id"):
            manual_rows.append(
                {
                    "identifier": identifier,
                    "name": identifier.replace("_", " "),
                    "unit": rows["unit"].iloc[0],
                    "kind": "manual_series",
                    "observation_start": rows["as_of_date"].min().date().isoformat(),
                    "observation_end": rows["as_of_date"].max().date().isoformat(),
                    "source": "; ".join(sorted(set(rows["source"]))),
                    "availability": "excel_observations_present",
                }
            )
        parts.append(frame("manual", manual_rows))
        known["manual"] = {row["identifier"] for row in manual_rows}
    catalog = pd.concat(parts, ignore_index=True)
    catalog["local_observations"] = False
    for provider, ids in known.items():
        catalog.loc[
            catalog["provider"].eq(provider) & catalog["identifier"].isin(ids),
            "local_observations",
        ] = True
    if catalog.duplicated(["provider", "identifier"]).any():
        raise ValueError("Nonunique provider and identifier in the catalog")
    with TemporaryDirectory(prefix=".staging-", dir=parent) as tmp:
        stage = Path(tmp)
        path = stage / "series_catalog.parquet"
        catalog.sort_values(["provider", "identifier"]).to_parquet(path, index=False)
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        counts = catalog["provider"].value_counts().to_dict()
        manifest = {
            "generated_utc": now.isoformat(),
            "rows_by_provider": counts,
            "file": path.name,
            "sha256": sha,
            "fred_snapshot": str(fred_db),
            "fred_snapshot_sha256": fred_sha,
            "yahoo_exchange_directory_counts_before_dedup": exchanges,
            "eia_bulk_archive_counts_before_dedup": datasets,
            "manual_workbooks": files,
            "observation_download": False,
            "limitations": [
                "FRED coverage is the imported public release-listed snapshot.",
                "EIA bulk manifest does not cover every API v2 facet combination.",
                "Yahoo candidates use exchange directories plus reference symbols.",
                "Verify Yahoo-derived symbols unless marked downloaded.",
                "Candidate matches are not validated economic proxies.",
            ],
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        stage.rename(final)
    print(f"Catalog identifiers: {counts}", flush=True)
    return final
