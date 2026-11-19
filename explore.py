"""Download nflverse play-by-play data and print summary statistics.

Usage:
    python src/explore.py                      # default seasons 2016-2023
    python src/explore.py --start 2020 --end 2023
"""

try:
    import truststore
    truststore.inject_into_ssl()   # use Windows' trusted certificates
except ImportError:
    pass

import argparse
from pathlib import Path

import shutil, time, urllib.request

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


def download(url: str, dest: Path, tries: int = 3) -> None:
    """Download url to dest with a timeout and retries."""
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as r, \
                    open(part, "wb") as f:
                shutil.copyfileobj(r, f)
            part.replace(dest)          # only appears once fully downloaded
            return
        except Exception as e:
            print(f"  attempt {attempt} failed: {e}")
            part.unlink(missing_ok=True)
            time.sleep(2)
    raise RuntimeError(f"Could not download {url}")

def load_season(season: int) -> pd.DataFrame:
    """Load one season, caching a trimmed copy under data/."""
    DATA_DIR.mkdir(exist_ok=True)
    trimmed = DATA_DIR / f"pbp_{season}.csv"
    if trimmed.exists():
        return pd.read_csv(trimmed, low_memory=False)
    raw = DATA_DIR / f"play_by_play_{season}.csv.gz"
    if not raw.exists():
        print(f"Downloading {season}...")
        download(URL.format(season=season), raw)
    df = pd.read_csv(raw, compression="gzip",
                     usecols=lambda c: c in COLUMNS, low_memory=False)
    df.to_csv(trimmed, index=False)
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
