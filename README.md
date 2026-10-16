# SDS Sports Model — public-facing website

A user-friendly Streamlit website built around the existing Python odds engine.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

## Publish a link people can visit

1. Create a GitHub repository and upload `app.py`, `odds_engine.py`, `sample_markets.csv`, and `requirements.txt` (do not upload sensitive data or API keys).
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, your repository, branch, and `app.py` as the entrypoint.
4. Deploy. Streamlit Community Cloud will provide a shareable public URL, subject to its current hosting terms.

Other options include hosting the Streamlit app on Render or a VPS.

## Real data format

Upload a CSV containing:

| Column | Example | Meaning |
|---|---|---|
| game | Team A @ Team B | Game identifier |
| market | Moneyline | Market name |
| selection | Team A | Side or prop selection |
| line | -3.5 | Spread/total/prop line; blank for moneyline or TD |
| american_odds | -110 | Nonzero American odds |
| model_probability | 0.55 | Your calibrated win probability between 0 and 1 |

Sample rows are entirely fictional and are never live lines or trained forecasts.

## Architecture / limitations

- `odds_engine.py` is the existing calculation module: implied probability, two-way no-vig normalization, EV and quarter-Kelly.
- `app.py` provides the public UI, filters, CSV ingestion, calculator, charts, and CSV export.
- `sample_markets.csv` is intentionally fictional demonstration data.
- No trained predictive model or odds API is integrated yet. To turn this into a live projection service, add time-aware NFL data ingestion, calibrated game/player forecasting, authorized odds feeds, backtests, and scheduled data refresh.
- No sportsbook accounts, bet placement, or user tracking.

This is analytics software, not betting advice. Validate forecasts out of sample before using them.

## Real NFL statistics integration (October 2026)

The website now has a public NFL statistics panel powered by `nflreadpy` and nflverse. It shows upcoming regular-season schedules, season-to-date team game averages, and player statistical leaderboards. The `nfl_stats.py` module holds data access and transformation functions. Streamlit caches fetched results for one hour, and a manual refresh button clears that cache. Upstream nflverse releases may lag games, and no data is fabricated if a feed is unavailable.

Run locally:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

**Data rights:** Review nflverse and any underlying provider terms before commercial deployment or redistribution. **No trained projections or live sportsbook prices** are supplied by this integration. The market board remains fictional demo data unless the visitor uploads independently produced projections.

For deployment, push the contents of this folder to a GitHub repository and configure Streamlit Community Cloud to run `app.py`. A public URL is not created by downloading this package.
