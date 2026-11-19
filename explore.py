"""Download nflverse play-by-play data and print summary statistics.

Usage:
    python src/explore.py                      # default seasons 2016-2023
    python src/explore.py --start 2020 --end 2023
"""
import argparse
from pathlib import Path

import pandas as pd

URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
       "pbp/play_by_play_{season}.csv.gz")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Columns known before the snap. Formation/personnel are deliberately
# excluded for now (see README, "Leakage").
COLUMNS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam",
    "play_type", "down", "ydstogo", "yardline_100", "qtr",
    "game_seconds_remaining", "score_differential",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
]


def load_season(season: int) -> pd.DataFrame:
    """Load one season, caching the downloaded file under data/."""
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / f"pbp_{season}.parquet"
    if cache.exists():
        try:
            return pd.read_parquet(cache)
        except Exception:
            pass  # fall through and re-download
    print(f"Downloading {season}...")
    df = pd.read_csv(URL.format(season=season), compression="gzip",
                     usecols=lambda c: c in COLUMNS, low_memory=False)
    try:
        df.to_parquet(cache)
    except Exception:
        df.to_csv(DATA_DIR / f"pbp_{season}.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2016)
    ap.add_argument("--end", type=int, default=2023)
    args = ap.parse_args()

    pbp = pd.concat([load_season(s) for s in range(args.start, args.end + 1)],
                    ignore_index=True)
    print(f"\nAll plays: {len(pbp):,} rows, seasons {args.start}-{args.end}")

    # Keep regular-season offensive plays with a known play type.
    plays = pbp[(pbp["season_type"] == "REG") & pbp["play_type"].notna()
                & pbp["posteam"].notna()]
    print(f"Regular-season offensive plays: {len(plays):,}\n")

    print("Play type counts:")
    print(plays["play_type"].value_counts().to_string(), "\n")

    # Baseline: always predict the most common play type.
    share = plays["play_type"].value_counts(normalize=True)
    print(f"Majority-class baseline accuracy: {share.iloc[0]:.3f} "
          f"('{share.index[0]}')\n")

    rp = plays[plays["play_type"].isin(["run", "pass"])]
    print("Pass rate by season:")
    print(rp.groupby("season")["play_type"]
            .apply(lambda s: (s == "pass").mean()).round(3).to_string(), "\n")

    print("Pass rate by down:")
    print(rp.groupby("down")["play_type"]
            .apply(lambda s: (s == "pass").mean()).round(3).to_string(), "\n")

    print("Plays per team (top 5 / bottom 5):")
    counts = rp["posteam"].value_counts()
    print(pd.concat([counts.head(), counts.tail()]).to_string(), "\n")

    print("Missing values in key columns:")
    print(rp[COLUMNS[7:]].isna().sum().to_string())


if __name__ == "__main__":
    main()
