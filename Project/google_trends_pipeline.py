"""
Google Trends Scraping Pipeline
================================
Loops through a list/CSV of keywords, fetches interest-over-time data
for a custom multi-year range, and saves results to CSV with auto-resume.

Usage:
    pip install pytrends pandas

    # Option A: pass a Python list (edit KEYWORDS below)
    python google_trends_pipeline.py

    # Option B: pass a CSV file with a column of keywords
    python google_trends_pipeline.py --input keywords.csv --column keyword

    # Full options:
    python google_trends_pipeline.py \
        --input final_keywords.csv \
        --column keyword \
        --start 2016-01-01 \
        --end   2025-12-31 \
        --geo   US \
        --output trends_final.csv \
        --delay 15 \
        --retries 5

    # Single keyword example (defaults scrape last 10 years through today):
    python google_trends_pipeline.py --keywords GOOGL --geo US --output trends_googl.csv
"""

import argparse
import time
import random
import logging
import datetime as dt
import pandas as pd
from pathlib import Path
from pytrends.request import TrendReq

# ─── Inline keyword list (used if no --input file is given) ───────────────────
KEYWORDS = [
    "GOOGL",  # Alphabet (Google) ticker; change this or pass --input/--keywords
]

# ─── Anchor word for rescaling across batches ─────────────────────────────────
ANCHOR = "stock market"

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Core fetcher ─────────────────────────────────────────────────────────────

def build_timeframe(start: str, end: str) -> str:
    """Convert 'YYYY-MM-DD' strings to PyTrends timeframe format."""
    return f"{start} {end}"

def _years_ago(date: dt.date, years: int) -> dt.date:
    try:
        return date.replace(year=date.year - years)
    except ValueError:
        # Handles Feb 29th → Feb 28th on non-leap years
        return date.replace(month=2, day=28, year=date.year - years)


def fetch_batch(
    pytrends: TrendReq,
    keywords: list[str],
    timeframe: str,
    geo: str,
    retries: int,
    base_delay: float,
    anchor: str | None,
) -> pd.DataFrame | None:
    """
    Fetch interest-over-time for up to 4 keywords + anchor.
    Anchor is included in every batch for cross-batch rescaling.
    Returns a DataFrame or None if all retries fail.
    """
    if anchor:
        keywords_with_anchor = [k for k in keywords if k != anchor] + [anchor]
    else:
        keywords_with_anchor = keywords

    for attempt in range(1, retries + 1):
        try:
            pytrends.build_payload(keywords_with_anchor, timeframe=timeframe, geo=geo)
            df = pytrends.interest_over_time()

            if df.empty:
                log.warning("Empty response for %s", keywords)
                return pd.DataFrame()

            # Drop the 'isPartial' column Google adds
            df = df.drop(columns=["isPartial"], errors="ignore")
            df = df.reset_index()           # brings 'date' back as a column
            return df

        except Exception as exc:
            wait = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 3)
            log.warning(
                "Attempt %d/%d failed for %s — %s. Retrying in %.1fs …",
                attempt, retries, keywords, exc, wait,
            )
            time.sleep(wait)

    log.error("All retries exhausted for batch: %s", keywords)
    return None


def rescale_batch(df: pd.DataFrame, keywords: list[str], anchor: str | None) -> pd.DataFrame:
    """
    Rescale keyword scores relative to anchor word.
    Divides each keyword series by the anchor series so scores
    are comparable across batches.
    """
    if not anchor:
        return df

    if anchor not in df.columns:
        log.warning("Anchor '%s' not in response — skipping rescale", anchor)
        return df

    anchor_series = df[anchor].replace(0, float("nan"))  # avoid division by zero

    for kw in keywords:
        if kw in df.columns:
            df[kw] = df[kw] / anchor_series * 100  # rescale to anchor=100 baseline

    return df


def melt_batch(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    """
    Wide → long: one row per (keyword, date) pair.
    Excludes the anchor word from the output.
    """
    cols = ["date"] + [k for k in keywords if k in df.columns]
    df = df[cols].melt(id_vars="date", var_name="keyword", value_name="interest")
    return df


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run(
    keywords: list[str],
    start: str,
    end: str,
    geo: str,
    output_path: Path,
    delay: float,
    retries: int,
    anchor: str | None = ANCHOR,
    batch_size: int = 4,          # 4 keywords + 1 anchor = 5 (Google Trends max)
) -> pd.DataFrame:

    timeframe = build_timeframe(start, end)
    log.info("Timeframe : %s", timeframe)
    log.info("Geography : %s", geo or "worldwide")
    log.info("Anchor    : %s", anchor or "(none)")
    log.info("Keywords  : %d total, batches of %d", len(keywords), batch_size)
    log.info("Output    : %s", output_path)

    # ── Auto-resume: skip already-fetched keywords ──────────────────────────
    already_done: set[str] = set()
    if output_path.exists():
        try:
            existing = pd.read_csv(output_path)
            if "keyword" in existing.columns and not existing.empty:
                already_done = set(existing["keyword"].dropna().unique())
                log.info("Resuming — %d keywords already in output file", len(already_done))
        except Exception as exc:
            log.warning("Could not read existing output (%s); starting fresh.", exc)
    else:
        # Write header
        pd.DataFrame(columns=["date", "keyword", "interest"]).to_csv(
            output_path, index=False
        )

    remaining = [k for k in keywords if k not in already_done]
    log.info("%d keywords left to fetch", len(remaining))

    # ── Chunk into batches of ≤4 ────────────────────────────────────────────
    batches = [
        remaining[i : i + batch_size]
        for i in range(0, len(remaining), batch_size)
    ]

    pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 30))

    all_results = []

    for idx, batch in enumerate(batches, 1):
        log.info("[%d/%d] Fetching: %s", idx, len(batches), batch)

        raw = fetch_batch(pytrends, batch, timeframe, geo, retries, delay, anchor=anchor)

        if raw is None:
            log.error("Skipping batch %s after all retries.", batch)
            continue

        if raw.empty:
            # Google returned no data — record None so we know we tried
            placeholder = pd.DataFrame({
                "date": [start] * len(batch),
                "keyword": batch,
                "interest": [None] * len(batch),
            })
            placeholder.to_csv(output_path, mode="a", header=False, index=False)
            continue

        # Rescale relative to anchor
        raw = rescale_batch(raw, batch, anchor=anchor)

        # Melt to long format (anchor excluded)
        melted = melt_batch(raw, batch)
        all_results.append(melted)

        # Append this batch to disk immediately (crash-safe)
        melted.to_csv(output_path, mode="a", header=False, index=False)
        log.info("  ✓ saved %d rows", len(melted))

        # ── Polite delay between requests ───────────────────────────────────
        if idx < len(batches):
            jitter = random.uniform(0, delay * 0.4)
            sleep_time = delay + jitter
            log.info("  sleeping %.1fs …", sleep_time)
            time.sleep(sleep_time)

    log.info("Done. Results saved to %s", output_path)

    # Return full combined DataFrame
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.read_csv(output_path)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    today = dt.date.today()
    default_end = today.isoformat()
    default_start = _years_ago(today, 10).isoformat()

    p = argparse.ArgumentParser(description="Google Trends bulk scraper")
    p.add_argument("--input",    help="CSV file with keywords", default=None)
    p.add_argument("--column",   help="Column name in CSV",     default="keyword")
    p.add_argument("--keywords", help="Comma-separated keywords (overrides inline list)", default=None)
    p.add_argument("--start",    default=default_start,         help="Start date YYYY-MM-DD (default: 10y ago)")
    p.add_argument("--end",      default=default_end,           help="End date   YYYY-MM-DD (default: today)")
    p.add_argument("--geo",      default="US",                  help="Country code e.g. US, GB (blank = worldwide)")
    p.add_argument("--output",   default="trends_final.csv",    help="Output CSV path")
    p.add_argument("--delay",    type=float, default=15,        help="Base delay in seconds between requests")
    p.add_argument("--retries",  type=int,   default=5,         help="Retries per batch on failure")
    p.add_argument("--anchor",   default=ANCHOR,               help="Anchor term for rescaling (blank disables)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load keywords
    if args.input:
        kw_df = pd.read_csv(args.input)
        keywords = kw_df[args.column].dropna().str.strip().unique().tolist()
        log.info("Loaded %d keywords from %s[%s]", len(keywords), args.input, args.column)
    elif args.keywords:
        keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
        log.info("Loaded %d keywords from --keywords", len(keywords))
    else:
        keywords = KEYWORDS
        log.info("Using inline keyword list (%d words)", len(keywords))

    anchor = args.anchor.strip() if isinstance(args.anchor, str) else args.anchor
    if anchor == "":
        anchor = None

    result_df = run(
        keywords=keywords,
        start=args.start,
        end=args.end,
        geo=args.geo,
        output_path=Path(args.output),
        delay=args.delay,
        retries=args.retries,
        anchor=anchor,
    )

    print(result_df.head(20).to_string(index=False))
