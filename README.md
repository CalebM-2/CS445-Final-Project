# CS445-Final-Project

NFL Play Prediction

Authors: Caleb Meadows, Thomas Fowler (CS 445, Fall 2026)

A model that predicts what play an NFL offense will call, given the game situation (team, down, distance, field position, quarter, time left, score, timeouts), plus a small demo where you pick a situation and see the predicted play probabilities.

Research question

Does knowing which team has the ball improve play prediction over situation-only features, measured by log loss and accuracy on a held-out season?

Data

nflverse play-by-play data (derived from nflfastR). src/explore.py downloads it automatically; files are cached in data/ (not committed).

Setup
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
Run
python src/explore.py                         # seasons 2016-2023
python src/explore.py --start 2020 --end 2023

This downloads the data and prints play-type counts, the majority-class baseline accuracy, and pass rates by season and down.

Plan
Step	Status
Download data, summary statistics	done
Train/validation/test split by season	planned
Baselines and models (logistic regression, XGBoost, small neural net)	planned
Evaluation: log loss, accuracy, calibration, failure analysis	planned
Interactive demo	planned
Leakage

Only information available before the snap is used. Formation and personnel are excluded unless a decision says otherwise. Splits are by season, never by random play, because plays within a game are correlated.

Use of AI tools:
