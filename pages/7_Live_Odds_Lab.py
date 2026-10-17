"""SDS Sports Model 4.5: FanDuel moneyline research lab.

Requires an API key in Streamlit secrets: ODDS_API_KEY.
Does not automatically join sportsbook events to model predictions.
"""
import math
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="SDS | Odds Lab", page_icon="🏈", layout="wide")
st.title("🏈 SDS Sportsbook Odds Lab 4.5")
st.caption("FanDuel NFL moneylines from The Odds API • Research and pricing calculator")
st.warning("Experimental comparison only. Odds may change or be unavailable. No verified betting edge, guaranteed return, or automatic SDS model linkage.")

API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"


def implied(american):
    american = int(american)
    if american == 0:
        raise ValueError("American odds cannot be zero")
    return 100 / (american + 100) if american > 0 else -american / (-american + 100)


def profit_per_dollar(american):
    american = int(american)
    if american == 0:
        raise ValueError("American odds cannot be zero")
    return american / 100 if american > 0 else 100 / -american


@st.cache_data(ttl=900, show_spinner=False)
def fetch_fanduel(api_key):
    response = requests.get(
        API_URL,
        params={"apiKey": api_key, "bookmakers": "fanduel", "markets": "h2h", "oddsFormat": "american", "dateFormat": "iso"},
        timeout=25,
    )
    if response.status_code != 200:
        # Do not include the response URL (which contains the secret key).
        raise RuntimeError(f"Odds provider returned HTTP {response.status_code}. Check your API key, plan quota, or provider status.")
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected odds response from provider")
    return payload, response.headers.get("x-requests-remaining", "Not supplied")


def parse_games(events):
    games = []
    now = datetime.now(timezone.utc)
    for event in events:
        try:
            start = pd.to_datetime(event.get("commence_time"), utc=True)
            if pd.isna(start) or start.to_pydatetime() <= now:
                continue  # pregame comparisons only
            home, away = event["home_team"], event["away_team"]
            for bookmaker in event.get("bookmakers", []):
                if bookmaker.get("key") != "fanduel":
                    continue
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    odds = {o.get("name"): o.get("price") for o in market.get("outcomes", [])}
                    if home not in odds or away not in odds:
                        continue
                    home_odds, away_odds = int(odds[home]), int(odds[away])
                    p_home, p_away = implied(home_odds), implied(away_odds)
                    book_sum = p_home + p_away
                    games.append({
                        "Game ID": str(event.get("id", "")),
                        "Kickoff (UTC)": start.strftime("%Y-%m-%d %H:%M UTC"),
                        "Home": home, "Away": away,
                        "Home odds": home_odds, "Away odds": away_odds,
                        "Home implied %": 100 * p_home,
                        "Away implied %": 100 * p_away,
                        "Overround %": 100 * (book_sum - 1),
                        "Home no-vig %": 100 * p_home / book_sum,
                        "Away no-vig %": 100 * p_away / book_sum,
                        "Odds updated": bookmaker.get("last_update", "Unknown"),
                    })
        except (ValueError, TypeError, KeyError, ZeroDivisionError):
            continue
    return pd.DataFrame(games)


try:
    key = st.secrets.get("ODDS_API_KEY", "")
except Exception:
    key = ""

if not key:
    st.info("To enable live odds, add your personal The Odds API key to Streamlit Community Cloud → App settings → Secrets as `ODDS_API_KEY = \"your_key_here\"`. Never paste your key into GitHub or public chat.")
    st.markdown("[Get an API key from The Odds API](https://the-odds-api.com/)")
    st.stop()

st.caption("Data refreshes at most every 15 minutes per app cache. Provider usage credits may apply.")
try:
    with st.spinner("Retrieving FanDuel NFL moneylines..."):
        events, remaining = fetch_fanduel(key)
    st.caption(f"API credits remaining (provider-reported): {remaining}")
    frame = parse_games(events)
    if frame.empty:
        st.info("No upcoming FanDuel NFL moneyline markets were returned. The bookmaker may not have available markets at this time.")
        st.stop()
    st.subheader("Available pregame moneylines")
    columns = ["Kickoff (UTC)", "Away", "Home", "Away odds", "Home odds", "Away implied %", "Home implied %", "Overround %", "Away no-vig %", "Home no-vig %", "Odds updated"]
    st.dataframe(frame[columns].round(2), hide_index=True, use_container_width=True)
    st.download_button("Download current odds snapshot CSV", frame.to_csv(index=False), file_name="sds_fanduel_odds_snapshot.csv", mime="text/csv")

    st.subheader("Manual model-probability comparison")
    st.caption("Enter a probability from your SDS model for the chosen team. It is NOT pulled automatically from your Command Center; check that the matchup and projection are for the same game.")
    choice = st.selectbox("Choose matchup", range(len(frame)), format_func=lambda i: f"{frame.iloc[i]['Away']} @ {frame.iloc[i]['Home']} — {frame.iloc[i]['Kickoff (UTC)']}")
    game = frame.iloc[choice]
    team = st.radio("Evaluate team", [game["Home"], game["Away"]], horizontal=True)
    selected_odds = int(game["Home odds"] if team == game["Home"] else game["Away odds"])
    selected_market = float(game["Home no-vig %"] if team == game["Home"] else game["Away no-vig %"])
    model_probability = st.number_input("Your model's win probability (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.5)
    p = model_probability / 100
    ev = p * profit_per_dollar(selected_odds) - (1 - p)
    a, b, c = st.columns(3)
    a.metric("FanDuel moneyline", f"{selected_odds:+d}")
    b.metric("No-vig market estimate", f"{selected_market:.1f}%")
    c.metric("Hypothetical EV / $100", f"${ev*100:+.2f}")
    st.caption(f"Model minus no-vig market: {model_probability - selected_market:+.2f} percentage points. EV assumes your entered probability is correct, with no additional costs or uncertainty. Positive hypothetical EV is not proof of an actionable edge.")
    st.info("Snapshot export downloads to your device only. Streamlit caching does not provide permanent historical storage; a durable prediction-and-odds archive is a future upgrade.")
except requests.RequestException:
    st.error("Could not reach the odds provider. Please try again later.")
except Exception as exc:
    st.error("The odds page could not load. Your other SDS pages are unaffected.")
    st.caption(f"{type(exc).__name__}: {exc}")
