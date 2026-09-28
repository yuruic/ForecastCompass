#!/usr/bin/env python3
"""
Prophet Arena Dataset Loader.

Load prediction market data from HuggingFace's Prophet Arena dataset.
Dataset: prophetarena/Prophet-Arena-Subset-1200

This dataset provides high-quality prediction market data with:
- Real market odds (yes_ask, yes_bid, no_ask, no_bid)
- Market liquidity
- Augmented titles and rules
- Curated sources
"""

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import ast

import pandas as pd

logger = logging.getLogger(__name__)


def load_prophet_arena_dataset(cache_dir: str = None) -> pd.DataFrame:
    """
    Load the Prophet Arena dataset from HuggingFace.
    
    Args:
        cache_dir: Directory to cache the dataset
        
    Returns:
        DataFrame with the dataset
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    logger.info("Loading Prophet Arena dataset from HuggingFace...")
    
    # Load with caching
    if cache_dir:
        ds = load_dataset(
            'prophetarena/Prophet-Arena-Subset-1200',
            cache_dir=cache_dir
        )
    else:
        ds = load_dataset('prophetarena/Prophet-Arena-Subset-1200')
    
    df = ds['train'].to_pandas()
    
    # Parse JSON string fields
    df['market_data_parsed'] = df['market_data'].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    df['market_outcome_parsed'] = df['market_outcome'].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    df['markets_parsed'] = df['markets'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df['sources_parsed'] = df['sources'].apply(
        lambda x: json.loads(x) if isinstance(x, str) and x else []
    )
    
    # Parse datetime fields
    df['close_time_dt'] = pd.to_datetime(df['close_time'], format='ISO8601')
    df['snapshot_time_dt'] = pd.to_datetime(df['snapshot_time'], format='ISO8601')
    
    logger.info(f"Loaded {len(df)} events from Prophet Arena")
    logger.info(f"Date range: {df['close_time_dt'].min()} to {df['close_time_dt'].max()}")
    
    return df


def get_prophet_arena_events_by_period(
    df: pd.DataFrame,
    period_back: int,
    time_unit: str = "weeks",
    max_events: int = 50,
    min_markets: int = 1,
    reference_date: datetime = None
) -> List[Dict]:
    """
    Get events from a specific time period, where period 1 is the most recent.

    The Prophet Arena dataset has events from June 2025 to November 2025.
    We divide this into time periods going backwards from the latest date.

    Args:
        df: Prophet Arena DataFrame
        period_back: Period number (1 = most recent, 2 = second most recent, etc.)
        time_unit: "weeks" or "days" for time granularity
        max_events: Maximum events to return
        min_markets: Minimum markets per event
        reference_date: Reference date for period calculation (default: latest in dataset)

    Returns:
        List of event dictionaries
    """
    if reference_date is None:
        reference_date = df['close_time_dt'].max()

    # Calculate date range based on time_unit
    if time_unit == "days":
        end_date = reference_date - timedelta(days=period_back - 1)
        start_date = end_date - timedelta(days=1)
        unit_name = "day"
    else:  # weeks (default)
        end_date = reference_date - timedelta(weeks=period_back - 1)
        start_date = end_date - timedelta(weeks=1)
        unit_name = "week"

    logger.info(f"Getting events for {unit_name} {period_back}: {start_date.date()} to {end_date.date()}")
    
    # Filter by date range
    mask = (df['close_time_dt'] >= start_date) & (df['close_time_dt'] < end_date)
    period_df = df[mask].copy()

    if len(period_df) == 0:
        logger.warning(f"No events found in {unit_name} {period_back}")
        return []

    # Filter by minimum markets
    if min_markets > 1:
        period_df = period_df[period_df['markets_parsed'].apply(len) >= min_markets]

    # Sort by close time and limit
    period_df = period_df.sort_values('close_time_dt').head(max_events)

    events = []
    for _, row in period_df.iterrows():
        event = format_prophet_arena_event(row)
        if event:
            events.append(event)

    logger.info(f"Found {len(events)} events for {unit_name} {period_back}")
    return events


def get_prophet_arena_events_by_week(
    df: pd.DataFrame,
    week_number: int,
    max_events: int = 50,
    min_markets: int = 1,
    reference_date: datetime = None
) -> List[Dict]:
    """
    Backward compatibility wrapper for get_prophet_arena_events_by_period.

    Args:
        df: Prophet Arena DataFrame
        week_number: Week number (1 = most recent, 2 = second most recent, etc.)
        max_events: Maximum events to return
        min_markets: Minimum markets per event
        reference_date: Reference date for week calculation (default: latest in dataset)

    Returns:
        List of event dictionaries
    """
    return get_prophet_arena_events_by_period(
        df=df,
        period_back=week_number,
        time_unit="weeks",
        max_events=max_events,
        min_markets=min_markets,
        reference_date=reference_date
    )


def format_prophet_arena_event(row: pd.Series) -> Optional[Dict]:
    """
    Format a Prophet Arena row into the standard event format.
    
    Args:
        row: DataFrame row
        
    Returns:
        Event dictionary compatible with the pipeline
    """
    try:
        markets = row['markets_parsed']
        market_data = row['market_data_parsed']
        market_outcome = row['market_outcome_parsed']
        
        if not markets or not market_outcome:
            return None
        
        # Build market_info with real odds from Prophet Arena
        market_info = {}
        for market_name in markets:
            if market_name in market_data:
                md = market_data[market_name]
                market_info[market_name] = {
                    "ticker": f"{row['event_ticker']}-{market_name}",
                    "event_ticker": row['event_ticker'],
                    "market_type": "binary",
                    "title": row['title'],
                    # Real market odds (in cents, 0-100)
                    "yes_ask": md.get('yes_ask', 50),
                    "yes_bid": md.get('yes_bid', 50),
                    "no_ask": md.get('no_ask', 50),
                    "no_bid": md.get('no_bid', 50),
                    "liquidity": md.get('liquidity', 0),
                    "result": "yes" if market_outcome.get(market_name, 0) == 1 else "no",
                }
            else:
                market_info[market_name] = {
                    "ticker": f"{row['event_ticker']}-{market_name}",
                    "event_ticker": row['event_ticker'],
                    "market_type": "binary",
                    "title": row['title'],
                    "yes_ask": 50,
                    "no_ask": 50,
                    "result": "yes" if market_outcome.get(market_name, 0) == 1 else "no",
                }
        
        # Format sources
        sources = row.get('sources_parsed', [])
        if sources and isinstance(sources, str):
            try:
                sources = json.loads(sources)
            except:
                sources = []
        
        # Determine the best title to use (prefer augmented_title, then title, then original_title)
        augmented_title = row.get('augmented_title', '')
        main_title = row.get('title', '')
        original_title = row.get('original_title', row.get('title', ''))
        
        # Handle NaN values and empty strings
        import pandas as pd
        if pd.isna(augmented_title) or not augmented_title:
            augmented_title = ''
        if pd.isna(main_title) or not main_title:
            main_title = ''
        if pd.isna(original_title) or not original_title:
            original_title = ''
        
        # Use the first non-empty title
        best_title = augmented_title or main_title or original_title or f"Event: {row['event_ticker']}"
        
        event = {
            "event_ticker": row['event_ticker'],
            "title": best_title,
            "original_title": original_title or main_title or best_title,
            "category": row.get('category', ''),
            "markets": json.dumps(markets),
            "close_time": row['close_time'],
            "market_outcome": json.dumps(market_outcome),
            "sources": json.dumps(sources) if sources else json.dumps([]),
            "market_info": json.dumps(market_info),
            "market_data": row['market_data'],  # Keep original for reference
            "snapshot_time": row.get('snapshot_time', datetime.now(timezone.utc).isoformat()),
            "submission_id": row.get('submission_id', str(uuid.uuid4())),
            # Prophet Arena specific fields
            "rules": row.get('rules', ''),
            "augmented_title": row.get('augmented_title', ''),
        }
        
        return event
        
    except Exception as e:
        logger.error(f"Error formatting event {row.get('event_ticker', 'unknown')}: {e}")
        return None


def collect_prophet_arena_weekly(
    weeks_back: int = 2,
    week_duration: int = 1,
    time_unit: str = "weeks",
    output_dir: str = None,
    min_markets: int = 1,
    max_events: int = 50,
    cache_dir: str = None,
) -> str:
    """
    Collect events from Prophet Arena for a specific time range.

    This mirrors the interface of collect_weekly_markets() for Kalshi.

    Args:
        weeks_back: Number of time units back from latest data (name kept for compatibility)
        week_duration: Duration of collection period in time units (default 1)
        time_unit: "weeks" or "days" for time granularity (default "weeks")
        output_dir: Directory to save output CSV
        min_markets: Minimum markets per event
        max_events: Maximum events to collect
        cache_dir: Directory for dataset cache

    Returns:
        Path to the output CSV file
    """
    # Load dataset
    df = load_prophet_arena_dataset(cache_dir)

    # Get reference date (latest in dataset)
    reference_date = df['close_time_dt'].max()

    # Collect events for the time range
    all_events = []
    for offset in range(week_duration):
        period_num = weeks_back - offset
        events = get_prophet_arena_events_by_period(
            df=df,
            period_back=period_num,
            time_unit=time_unit,
            max_events=max_events,
            min_markets=min_markets,
            reference_date=reference_date
        )
        all_events.extend(events)
    
    if not all_events:
        logger.warning("No events found!")
        return None
    
    # Limit to max_events
    if len(all_events) > max_events:
        all_events = all_events[:max_events]
    
    # Create DataFrame
    rows_df = pd.DataFrame(all_events)
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate date range for filename
    close_times = pd.to_datetime(rows_df['close_time'], format='ISO8601')
    start_date = close_times.min()
    end_date = close_times.max()

    date_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    unit_label = "day" if time_unit == "days" else "week"
    output_path = output_dir / f"prophet_arena_{unit_label}{weeks_back}_{date_str}.csv"
    
    rows_df.to_csv(output_path, index=False)
    
    logger.info(f"Saved {len(rows_df)} events to {output_path}")
    
    # Print summary
    print(f"\n=== Prophet Arena Collection Summary ===")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Total events: {len(rows_df)}")
    total_markets = sum(len(json.loads(row['markets'])) for _, row in rows_df.iterrows())
    print(f"Total markets: {total_markets}")
    print(f"Categories: {rows_df['category'].value_counts().to_dict()}")
    print(f"Output: {output_path}")
    
    return str(output_path)


def get_prophet_arena_week_count(cache_dir: str = None) -> Tuple[int, datetime, datetime]:
    """
    Get the number of available weeks in the Prophet Arena dataset.
    
    Returns:
        Tuple of (num_weeks, earliest_date, latest_date)
    """
    df = load_prophet_arena_dataset(cache_dir)
    
    earliest = df['close_time_dt'].min()
    latest = df['close_time_dt'].max()
    
    # Calculate weeks
    delta = latest - earliest
    num_weeks = delta.days // 7 + 1
    
    return num_weeks, earliest, latest


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect Prophet Arena events")
    parser.add_argument("--weeks-back", "-w", type=int, default=2,
                        help="Number of weeks back from latest data (default: 2)")
    parser.add_argument("--duration", "-d", type=int, default=1,
                        help="Duration in weeks to collect (default: 1)")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory for CSV")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directory for dataset cache")
    parser.add_argument("--min-markets", type=int, default=1,
                        help="Minimum markets per event (default: 1)")
    parser.add_argument("--max-events", type=int, default=50,
                        help="Maximum events to collect (default: 50)")
    parser.add_argument("--info", action="store_true",
                        help="Show dataset info and exit")
    
    args = parser.parse_args()
    
    if args.info:
        num_weeks, earliest, latest = get_prophet_arena_week_count(args.cache_dir)
        print(f"\n=== Prophet Arena Dataset Info ===")
        print(f"Total weeks available: {num_weeks}")
        print(f"Date range: {earliest.date()} to {latest.date()}")
        print(f"Total events: 1200")
    else:
        output_path = collect_prophet_arena_weekly(
            weeks_back=args.weeks_back,
            week_duration=args.duration,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            min_markets=args.min_markets,
            max_events=args.max_events
        )
        
        if output_path:
            print(f"\n✅ Data saved to: {output_path}")
