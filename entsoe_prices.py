"""ENTSO-E API client for fetching day-ahead electricity prices.

This module provides functionality to fetch electricity price data from the
ENTSO-E Transparency Platform API for specified bidding zones and time periods.
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import pandas as pd
import pytz
import requests

# Constants
ENTSOE_BASE = "https://web-api.tp.entsoe.eu/api"
DE_LU_BZN = "10Y1001A1001A82H"  # Germany-Luxembourg bidding zone
REQUEST_TIMEOUT = 30  # seconds


@dataclass
class EntsoeConfig:
    """Configuration for ENTSO-E API requests.
    
    Attributes:
        token: ENTSO-E API security token
        zone: Bidding zone EIC code (default: DE-LU)
        base_url: ENTSO-E API base URL
    """
    token: str
    zone: str = DE_LU_BZN
    base_url: str = ENTSOE_BASE

def _iso_day_bounds_utc(start_yyyy_mm_dd: str, end_yyyy_mm_dd: str) -> Tuple[str, str]:
    """Convert date strings to ENTSO-E UTC timestamp format.
    
    ENTSO-E API requires UTC timestamps in YYYYMMDDHHMM format (no separators).
    periodStart is inclusive, periodEnd is exclusive.
    
    Args:
        start_yyyy_mm_dd: Start date in YYYY-MM-DD format
        end_yyyy_mm_dd: End date in YYYY-MM-DD format
        
    Returns:
        Tuple of (start_timestamp, end_timestamp) in YYYYMMDDHHMM format
        
    Raises:
        ValueError: If date strings are not in correct format
    """
    def parse_day(d: str) -> datetime:
        """Parse date string to UTC midnight datetime."""
        try:
            return datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError as e:
            raise ValueError(f"Invalid date format '{d}'. Expected YYYY-MM-DD") from e

    start = parse_day(start_yyyy_mm_dd)
    end = parse_day(end_yyyy_mm_dd)
    return start.strftime("%Y%m%d%H%M"), end.strftime("%Y%m%d%H%M")

def _ack_error_text(xml_bytes: bytes) -> Optional[str]:
    """Extract error message from ENTSO-E acknowledgement response.
    
    Args:
        xml_bytes: Raw XML response bytes from ENTSO-E API
        
    Returns:
        Error message string if acknowledgement found, None otherwise
    """
    if b"Acknowledgement_MarketDocument" not in xml_bytes:
        return None
    
    # Extract error code and text from XML
    code_match = re.search(rb"<code>(.*?)</code>", xml_bytes, flags=re.I | re.S)
    text_match = re.search(rb"<text>(.*?)</text>", xml_bytes, flags=re.I | re.S)
    
    code_txt = code_match.group(1).decode("utf-8", "ignore").strip() if code_match else "UNKNOWN"
    text = text_match.group(1).decode("utf-8", "ignore").strip() if text_match else "Unknown acknowledgement"
    
    return f"{code_txt}: {text}"

def _parse_iso_utc(zstr: str) -> datetime:
    """Parse ISO 8601 timestamp string to UTC datetime.
    
    Handles both 'Z' suffix and explicit timezone offsets.
    
    Args:
        zstr: ISO 8601 timestamp string (e.g., '2024-06-01T00:00Z' or '2024-06-01T00:00+00:00')
        
    Returns:
        Timezone-aware datetime in UTC
        
    Raises:
        ValueError: If timestamp string cannot be parsed
    """
    try:
        if zstr.endswith("Z"):
            return datetime.strptime(zstr, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
        
        dt = datetime.fromisoformat(zstr)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Invalid ISO timestamp format: '{zstr}'") from e

def entsoe_day_ahead_prices(
    token: str,
    zone: str,
    start_yyyy_mm_dd: str,
    end_yyyy_mm_dd: str,
) -> pd.DataFrame:
    """Fetch day-ahead electricity prices from ENTSO-E API.
    
    Retrieves Day-Ahead Market prices (document type A44) for a specified
    bidding zone and time period.
    
    Args:
        token: ENTSO-E API security token
        zone: Bidding zone EIC code (e.g., '10Y1001A1001A82H' for DE-LU)
        start_yyyy_mm_dd: Start date in YYYY-MM-DD format (inclusive)
        end_yyyy_mm_dd: End date in YYYY-MM-DD format (exclusive)
        
    Returns:
        DataFrame with columns:
            - utc: UTC timestamp string
            - berlin_time: Berlin timezone timestamp string
            - EUR_per_MWh: Price in EUR/MWh
            - EUR_per_kWh: Price in EUR/kWh
            - ct_per_kWh: Price in cents/kWh
            - position: Hour position in the period
            
    Raises:
        ValueError: If token is missing or dates are invalid
        requests.HTTPError: If API request fails
        RuntimeError: If API returns acknowledgement instead of data or parsing fails
    """
    if not token:
        raise ValueError("ENTSOE_API_TOKEN is required but not provided")

    period_start, period_end = _iso_day_bounds_utc(start_yyyy_mm_dd, end_yyyy_mm_dd)

    params = {
        "documentType": "A44",        # Day-ahead prices
        "processType": "A01",         # Day-ahead process
        "in_Domain": zone,            # Bidding zone (in)
        "out_Domain": zone,           # Bidding zone (out)
        "periodStart": period_start,  # UTC start
        "periodEnd": period_end,      # UTC end (exclusive)
        "securityToken": token,
    }

    try:
        resp = requests.get(ENTSOE_BASE, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch data from ENTSO-E API: {e}") from e

    # Check for acknowledgement (error) response
    ack = _ack_error_text(resp.content)
    if ack:
        raise RuntimeError(f"ENTSO-E API error: {ack}")

    # Parse XML without worrying about namespaces via local-name() logic
    try:
        # lazy import to avoid forcing lxml dependency
        from lxml import etree as ET  # type: ignore
        root = ET.fromstring(resp.content)
        def findall(elem, path): return elem.xpath(path, namespaces={})
        def text(elem, path):
            r = elem.xpath(path, namespaces={})
            return r[0].text if r else None
        TS_PATH = ".//*[local-name()='TimeSeries']"
        PERIOD_PATH = ".//*[local-name()='Period']"
        START_PATH = ".//*[local-name()='timeInterval']/*[local-name()='start']"
        RES_PATH = ".//*[local-name()='resolution']"
        POINT_PATH = ".//*[local-name()='Point']"
        POS_PATH = ".//*[local-name()='position']"
        PRICE_PATH = ".//*[local-name()='price.amount']"
    except Exception:
        # fallback to stdlib ElementTree
        import xml.etree.ElementTree as ET  # type: ignore
        root = ET.fromstring(resp.content)
        def _ln(tag):  # strip namespace
            return tag.split("}", 1)[-1] if "}" in tag else tag
        def iter_local(elem, name):
            for e in elem.iter():
                if _ln(e.tag) == name:
                    yield e
        def text_elem(e, name):
            for x in e:
                if _ln(x.tag) == name:
                    return x.text
            return None
        def findall(elem, path):  # minimal compatible paths
            if path.endswith("TimeSeries']"):
                return list(iter_local(elem, "TimeSeries"))
            if path.endswith("Period']"):
                return list(iter_local(elem, "Period"))
            if path.endswith("Point']"):
                return list(iter_local(elem, "Point"))
            raise NotImplementedError("Stdlib fallback only supports known paths.")
        def text(elem, path):
            # very small helper for known paths
            if path.endswith("start']"):
                ti = next((x for x in elem if _ln(x.tag) == "timeInterval"), None)
                if ti is None: return None
                return text_elem(ti, "start")
            if path.endswith("resolution']"):
                return next((x.text for x in elem if _ln(x.tag) == "resolution"), None)
            if path.endswith("position']"):
                return next((x.text for x in elem if _ln(x.tag) == "position"), None)
            if path.endswith("price.amount']"):
                return next((x.text for x in elem if _ln(x.tag) == "price.amount"), None)
            return None
        TS_PATH = ".//*[local-name()='TimeSeries']"
        PERIOD_PATH = ".//*[local-name()='Period']"
        START_PATH = ".//*[local-name()='timeInterval']/*[local-name()='start']"
        RES_PATH = ".//*[local-name()='resolution']"
        POINT_PATH = ".//*[local-name()='Point']"
        POS_PATH = ".//*[local-name()='position']"
        PRICE_PATH = ".//*[local-name()='price.amount']"

    rows = []
    tz_berlin = pytz.timezone("Europe/Berlin")

    for ts in findall(root, TS_PATH):
        for period in findall(ts, PERIOD_PATH):
            start_txt = text(period, START_PATH)
            if not start_txt:
                continue
            start_utc = _parse_iso_utc(start_txt)

            res_txt = text(period, RES_PATH) or "PT60M"
            m = re.match(r"PT(\d+)M", res_txt)
            step_min = int(m.group(1)) if m else 60

            for pt in findall(period, POINT_PATH):
                pos = int(text(pt, POS_PATH) or "1")
                eur_mwh = float(text(pt, PRICE_PATH))
                utc_dt = start_utc + timedelta(minutes=step_min * (pos - 1))
                # Convert UTC->Berlin
                berlin_dt = tz_berlin.fromutc(utc_dt.replace(tzinfo=timezone.utc))

                rows.append({
                    "utc": utc_dt.strftime("%Y-%m-%d %H:%M"),
                    "berlin_time": berlin_dt.strftime("%Y-%m-%d %H:%M"),
                    "EUR_per_MWh": eur_mwh,
                    "EUR_per_kWh": round(eur_mwh / 1000.0, 5),
                    "ct_per_kWh": round(eur_mwh / 10.0, 2),
                    "position": pos,
                })

    if not rows:
        raise RuntimeError("Parsed no time-series rows; the XML may not contain A44 data.")

    df = pd.DataFrame(rows).sort_values("utc").reset_index(drop=True)
    return df

def _main() -> None:
    """CLI entry point for fetching ENTSO-E prices.
    
    Usage:
        python -m prosumer_advisor.entsoe_prices --start 2024-06-01 --end 2024-06-02 --out prices.csv
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fetch ENTSO-E Day-Ahead electricity prices (A44)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch prices for DE-LU zone and save to CSV
  python -m prosumer_advisor.entsoe_prices --start 2024-06-01 --end 2024-06-02 --out prices.csv
  
  # Fetch and display prices for a different zone
  python -m prosumer_advisor.entsoe_prices --zone 10YFR-RTE------C --start 2024-06-01 --end 2024-06-02

Environment Variables:
  ENTSOE_API_TOKEN: Your ENTSO-E API security token (required)
        """
    )
    parser.add_argument(
        "--zone",
        default=DE_LU_BZN,
        help=f"Bidding zone EIC code (default: {DE_LU_BZN} for DE-LU)"
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date in YYYY-MM-DD format (inclusive, UTC day boundary)"
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date in YYYY-MM-DD format (exclusive, UTC day boundary)"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV file path (if not specified, prints to stdout)"
    )
    
    args = parser.parse_args()

    token = os.environ.get("ENTSOE_API_TOKEN", "")
    if not token:
        raise SystemExit(
            "ERROR: ENTSOE_API_TOKEN environment variable is not set.\n"
            "Please set it with your ENTSO-E API token."
        )

    try:
        df = entsoe_day_ahead_prices(
            token=token,
            zone=args.zone,
            start_yyyy_mm_dd=args.start,
            end_yyyy_mm_dd=args.end
        )
        
        if args.out:
            df.to_csv(args.out, index=False)
            print(f"✓ Successfully wrote {len(df)} rows to {args.out}")
        else:
            print(df.to_string(index=False))
            
    except Exception as e:
        raise SystemExit(f"ERROR: {e}") from e


if __name__ == "__main__":
    _main()
