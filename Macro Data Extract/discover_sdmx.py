"""Search provider SDMX dataflow metadata by keyword."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET

import httpx

ENDPOINTS = {
    "oecd": "https://sdmx.oecd.org/public/rest/dataflow/all/all/latest",
    "imf": "https://api.imf.org/external/sdmx/2.1/dataflow/all/all/latest",
    "ecb": "https://data-api.ecb.europa.eu/service/dataflow/ECB/all/latest",
    "bis": "https://stats.bis.org/api/v1/dataflow/BIS/all/latest",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("keyword", help="Case-insensitive text to search for.")
    parser.add_argument(
        "--provider",
        choices=(*ENDPOINTS, "all"),
        default="all",
        help="Provider whose dataflow metadata to search.",
    )
    args = parser.parse_args()
    providers = (
        ENDPOINTS
        if args.provider == "all"
        else {args.provider: ENDPOINTS[args.provider]}
    )
    found = False

    for provider, endpoint in providers.items():
        try:
            response = httpx.get(
                endpoint,
                headers={"Accept": "application/vnd.sdmx.structure+xml;version=2.1"},
                timeout=45,
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except (httpx.HTTPError, ET.ParseError) as error:
            print(f"{provider}: metadata query failed: {error}", file=sys.stderr)
            continue

        for flow in root.iter():
            if _local_name(flow.tag) != "Dataflow":
                continue
            names = [
                (child.attrib.get("lang", ""), "".join(child.itertext()).strip())
                for child in flow
                if _local_name(child.tag) == "Name"
            ]
            searchable = " ".join(name for _, name in names)
            if args.keyword.casefold() not in searchable.casefold():
                continue
            agency = flow.attrib.get("agencyID", flow.attrib.get("agencyIDRef", ""))
            flow_id = flow.attrib.get("id", "")
            version = flow.attrib.get("version", "")
            label = next((name for _, name in names if name), flow_id)
            print(f"{provider}\t{agency}\t{flow_id}\t{version}\t{label}")
            found = True

    if not found:
        print("No matching dataflows found.", file=sys.stderr)
        raise SystemExit(1)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


if __name__ == "__main__":
    main()
