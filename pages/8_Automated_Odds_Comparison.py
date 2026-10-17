"""SDS 4.6: automatic Advanced EPA + FanDuel pregame moneyline comparison.

Uses the Command Center 4.2 feature definitions and training window.
ODDS_API_KEY must be configured in Streamlit secrets. Research only.
"""
import math
from datetime import datetime, timezone

import nflreadpy as nfl
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="SDS | Automated Odds", page_icon="🏈", layout="wide")
st.title("🏈 SDS 4.6 | Automated Odds Comparison")
st.warning("Experimental research only. SDS probabilities are uncalibrated. Positive estimated EV is not a validated betting edge.")
st.caption("FanDuel pregame NFL moneylines via The Odds API, automatically joined to SDS Advanced EPA projections. Odds cached up to 15 minutes; EPA cached up to 6 hours.")

START_SEASON, LAST_TRAIN, CURRENT_SEASON = 2022, 2025, 2026
API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
TEAM_NAMES = {
    "Arizona Cardinals":"ARI", "Atlanta Falcons":"ATL", "Baltimore Ravens":"BAL",
    "Buffalo Bills":"BUF", "Carolina Panthers":"CAR", "Chicago Bears":"CHI",
    "Cincinnati Bengals":"CIN", "Cleveland Browns":"CLE", "Dallas Cowboys":"DAL",
    "Denver Broncos":"DEN", "Detroit Lions":"DET", "Green Bay Packers":"GB",
    "Houston Texans":"HOU", "Indianapolis Colts":"IND", "Jacksonville Jaguars":"JAX",
    "Kansas City Chiefs":"KC", "Las Vegas Raiders":"LV", "Los Angeles Chargers":"LAC",
    "Los Angeles Rams":"LA", "Miami Dolphins":"MIA", "Minnesota Vikings":"MIN",
    "New England Patriots":"NE", "New Orleans Saints":"NO", "New York Giants":"NYG",
    "New York Jets":"NYJ", "Philadelphia Eagles":"PHI", "Pittsburgh Steelers":"PIT",
    "San Francisco 49ers":"SF", "Seattle Seahawks":"SEA", "Tampa Bay Buccaneers":"TB",
    "Tennessee Titans":"TEN", "Washington Commanders":"WAS",
}


def implied(odds):
    odds = int(odds)
    if odds == 0:
        raise ValueError("Odds cannot be zero")
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def profit(odds):
    return int(odds) / 100 if int(odds) > 0 else 100 / -int(odds)


def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


@st.cache_data(ttl=21600, show_spinner=False)
def load_model_data():
    schedules = nfl.load_schedules(list(range(START_SEASON, CURRENT_SEASON + 1))).to_pandas()
    chunks, current_games = [], 0
    current_status = "2026 play-by-play available"
    for year in range(START_SEASON, CURRENT_SEASON + 1):
        try:
            pbp = nfl.load_pbp([year]).to_pandas()
            wanted = ["game_id", "posteam", "defteam", "epa", "success", "play_type"]
            missing = set(wanted) - set(pbp.columns)
            if missing:
                raise ValueError(f"Missing PBP columns: {sorted(missing)}")
            pbp = pbp[wanted]
            pbp = pbp[pbp.play_type.isin(["run", "pass"]) & pbp.posteam.notna() &
                      pbp.defteam.notna() & pbp.epa.notna()]
            agg = pbp.groupby(["game_id", "posteam", "defteam"], as_index=False).agg(
                off_epa=("epa", "mean"), off_success=("success", "mean"), plays=("epa", "size"))
            if year == CURRENT_SEASON:
                current_games = int(agg.game_id.nunique())
                if current_games == 0:
                    current_status = "2026 play-by-play has no completed games"
            chunks.append(agg)
        except Exception as exc:
            if year != CURRENT_SEASON:
                raise
            current_status = f"2026 play-by-play unavailable ({type(exc).__name__}); historical fallback"
    return schedules, pd.concat(chunks, ignore_index=True), current_status, current_games


def assemble(schedules, play_stats):
    games = schedules.copy()
    if "game_type" in games.columns:
        games = games[games.game_type.eq("REG")].copy()
    needed = ["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"]
    missing = set(needed) - set(games.columns)
    if missing:
        raise ValueError(f"Missing schedule columns: {sorted(missing)}")
    games = games.sort_values(["season", "week", "game_id"])
    stats = {}
    for r in play_stats.itertuples(index=False):
        stats[(str(r.game_id), str(r.posteam))] = (
            float(r.off_epa), float(r.off_success) if pd.notna(r.off_success) else 0.0, int(r.plays))
    history, rows = {}, []
    for (season, week), group in games.groupby(["season", "week"], sort=True):
        updates = []
        for g in group.itertuples(index=False):
            home, away, gid = str(g.home_team), str(g.away_team), str(g.game_id)
            def prior(team):
                h = history.get(team, [])[-5:]
                if not h:
                    return [0.0] * 5
                arr = np.asarray(h, dtype=float)
                return [float(v) for v in arr.mean(axis=0)[:4]] + [len(h) / 5.0]
            h, a = prior(home), prior(away)
            features = [h[0]-a[0], h[1]-a[1], h[2]-a[2], h[3]-a[3], h[4], a[4]]
            completed = pd.notna(g.home_score) and pd.notna(g.away_score)
            # NFL schedules expose gameday/gametime; preserve them for kickoff verification.
            gameday = getattr(g, "gameday", None)
            gametime = getattr(g, "gametime", None)
            kickoff = pd.NaT
            if pd.notna(gameday) and pd.notna(gametime):
                # nflverse gametime is US Eastern local time. Convert to UTC using DST rules.
                try:
                    kickoff = pd.Timestamp(f"{gameday} {gametime}").tz_localize(
                        "America/New_York", ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
                except (ValueError, TypeError):
                    pass
            rows.append(dict(season=int(season), week=int(week), home=home, away=away,
                             game_id=gid, completed=completed, features=features, kickoff=kickoff,
                             margin=float(g.home_score-g.away_score) if completed else np.nan,
                             total=float(g.home_score+g.away_score) if completed else np.nan))
            if completed:
                hs, az = stats.get((gid, home)), stats.get((gid, away))
                if hs and az and hs[2] >= 10 and az[2] >= 10:
                    updates.append((home, [hs[0], az[0], hs[1], az[1]]))
                    updates.append((away, [az[0], hs[0], az[1], hs[1]]))
        for team, values in updates:
            history.setdefault(team, []).append(values)
    return pd.DataFrame(rows)


def ridge_fit(df, target, penalty=15.0):
    x = np.asarray(df.features.tolist(), dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    y = df[target].to_numpy(dtype=float)
    reg = np.diag([0.0] + [penalty] * (x.shape[1] - 1))
    beta = np.linalg.solve(x.T @ x + reg, x.T @ y)
    sd = max(float(np.std(y - x @ beta, ddof=min(x.shape[1], len(y)-1))), 1.0)
    return beta, sd


def forecast(df, beta):
    x = np.asarray(df.features.tolist(), dtype=float)
    return np.column_stack([np.ones(len(x)), x]) @ beta


@st.cache_data(ttl=900, show_spinner=False)
def fetch_odds(api_key):
    response = requests.get(API_URL, params={"apiKey": api_key, "bookmakers": "fanduel",
        "markets": "h2h", "oddsFormat": "american", "dateFormat": "iso"}, timeout=25)
    if response.status_code != 200:
        # Never expose the request URL; it contains the secret.
        raise RuntimeError(f"Odds API HTTP {response.status_code}; check quota, credentials, or provider status")
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected odds provider response")
    return payload, response.headers.get("x-requests-remaining", "Unknown")


def parse_odds(events):
    rows, now = [], pd.Timestamp.now(tz="UTC")
    for event in events:
        home_name, away_name = event.get("home_team"), event.get("away_team")
        home, away = TEAM_NAMES.get(home_name), TEAM_NAMES.get(away_name)
        kickoff = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
        if pd.isna(kickoff) or kickoff <= now:
            continue
        for book in event.get("bookmakers", []):
            if book.get("key") != "fanduel":
                continue
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                prices = {item.get("name"): item.get("price") for item in market.get("outcomes", [])}
                try:
                    home_odds, away_odds = int(prices[home_name]), int(prices[away_name])
                    ph, pa = implied(home_odds), implied(away_odds)
                    updated = pd.to_datetime(market.get("last_update") or book.get("last_update"), utc=True, errors="coerce")
                except (KeyError, ValueError, TypeError, ZeroDivisionError):
                    continue
                rows.append(dict(home=home, away=away, home_name=home_name, away_name=away_name,
                    kickoff_odds=kickoff, home_odds=home_odds, away_odds=away_odds,
                    home_implied=ph, away_implied=pa, overround=ph+pa-1,
                    home_novig=ph/(ph+pa), away_novig=pa/(ph+pa), updated=updated,
                    provider_event_id=str(event.get("id", ""))))
    return pd.DataFrame(rows)


try:
    api_key = st.secrets.get("ODDS_API_KEY", "")
except Exception:
    api_key = ""
if not api_key:
    st.error("Add ODDS_API_KEY to your Streamlit app Secrets (same key used by SDS 4.5).")
    st.stop()

try:
    with st.spinner("Loading Advanced EPA model and FanDuel markets..."):
        schedules, pbp, status, current_games = load_model_data()
        all_games = assemble(schedules, pbp)
        train = all_games[all_games.completed & all_games.season.between(START_SEASON, LAST_TRAIN)]
        if len(train) < 300:
            raise ValueError("Insufficient historical games to train Advanced EPA")
        margin_beta, margin_sd = ridge_fit(train, "margin")
        total_beta, _ = ridge_fit(train, "total")
        future = all_games[all_games.season.eq(CURRENT_SEASON) & ~all_games.completed].copy()
        events, remaining = fetch_odds(api_key)
        odds = parse_odds(events)
    st.caption(f"Model status: {status} ({current_games} current-season games with PBP). Odds API credits remaining: {remaining}.")
    if "unavailable" in status:
        st.warning("Current-season EPA unavailable. Model form may be stale.")
    if future.empty or odds.empty:
        st.info("No future regular-season model games or upcoming FanDuel moneylines available to compare.")
        st.stop()

    future["home_model"] = forecast(future, margin_beta)
    future["total_model"] = forecast(future, total_beta)
    future["p_home"] = future.home_model.apply(lambda m: normal_cdf(m / margin_sd))
    future["p_away"] = 1 - future.p_home
    # One match per exact home/away pair. Do not silently match unknown team names.
    joined = odds.merge(future, on=["home", "away"], how="left", indicator=True, validate="many_to_one")
    joined["kickoff_delta_hours"] = (joined.kickoff_odds - joined.kickoff).abs().dt.total_seconds() / 3600
    # Schedule kickoff must be present and within 6 hours to guard against mismatched events.
    matched = joined[(joined._merge == "both") & joined.kickoff_delta_hours.notna() &
                     (joined.kickoff_delta_hours <= 6)].copy()
    unmatched = joined.loc[~joined.index.isin(matched.index)].copy()
    now = pd.Timestamp.now(tz="UTC")
    matched["odds_age_min"] = (now - matched.updated).dt.total_seconds() / 60
    matched["stale"] = matched.odds_age_min.isna() | (matched.odds_age_min > 60) | (matched.odds_age_min < -5)
    matched["home_ev_pct"] = 100 * (matched.p_home * matched.home_odds.map(profit) - (1 - matched.p_home))
    matched["away_ev_pct"] = 100 * (matched.p_away * matched.away_odds.map(profit) - (1 - matched.p_away))
    matched["home_gap_pp"] = 100 * (matched.p_home - matched.home_novig)
    matched["away_gap_pp"] = 100 * (matched.p_away - matched.away_novig)
    matched["matchup"] = matched.away_name + " @ " + matched.home_name
    matched["kickoff_utc"] = matched.kickoff_odds.dt.strftime("%Y-%m-%d %H:%M UTC")

    st.subheader("Automatic game matching")
    a, b, c = st.columns(3)
    a.metric("Matched games", len(matched))
    b.metric("Unmatched odds events", len(unmatched))
    c.metric("Stale / undated matched odds", int(matched.stale.sum()))
    if matched.empty:
        st.warning("No events passed the team and kickoff verification checks. Check the unmatched events below.")
    else:
        st.caption("Both sides of each matchup are shown. EV assumes the uncalibrated SDS probability is accurate. Sorted by largest hypothetical EV, not a recommendation.")
        records = []
        for r in matched.itertuples(index=False):
            for side in ("home", "away"):
                is_home = side == "home"
                records.append({
                    "Week": int(r.week), "Kickoff (UTC)": r.kickoff_utc, "Matchup": r.matchup,
                    "Team": r.home_name if is_home else r.away_name,
                    "FanDuel ML": r.home_odds if is_home else r.away_odds,
                    "SDS win %": 100 * (r.p_home if is_home else r.p_away),
                    "No-vig market %": 100 * (r.home_novig if is_home else r.away_novig),
                    "SDS minus market (pp)": r.home_gap_pp if is_home else r.away_gap_pp,
                    "Hypothetical EV %": r.home_ev_pct if is_home else r.away_ev_pct,
                    "Projected home margin": r.home_model,
                    "Projected total": r.total_model,
                    "Odds age (min)": r.odds_age_min,
                    "Odds status": "STALE/UNKNOWN" if r.stale else "Recent",
                })
        table = pd.DataFrame(records).sort_values("Hypothetical EV %", ascending=False)
        selected_week = st.selectbox("NFL week", sorted(table.Week.unique().tolist()))
        week_table = table[table.Week == selected_week]
        st.dataframe(week_table.round(2), hide_index=True, use_container_width=True)
        st.download_button("Download SDS + FanDuel comparison CSV", week_table.to_csv(index=False),
                           file_name=f"sds_4_6_week_{selected_week}_comparison.csv", mime="text/csv")
        st.info("The model uses 2022–2025 completed games for coefficients and previous-week EPA for 2026 form. It does not account for confirmed starting QBs, injuries, lineup changes, or live in-game developments.")

    with st.expander("Unmatched games and diagnostics"):
        if unmatched.empty:
            st.success("All returned pregame FanDuel events matched a 2026 regular-season game and kickoff.")
        else:
            diag = unmatched[["away_name", "home_name", "kickoff_odds", "away", "home", "_merge", "kickoff_delta_hours"]].copy()
            diag["Reason"] = np.where(diag["away"].isna() | diag["home"].isna(), "Unknown team name",
                np.where(diag["_merge"].eq("left_only"), "No 2026 regular-season schedule match", "Kickoff unavailable or >6h apart"))
            st.dataframe(diag.drop(columns=["_merge"]), hide_index=True, use_container_width=True)
        st.caption("Team aliases and kickoff timestamps must both match. Playoff games are intentionally excluded because Command Center 4.2 models regular-season games only.")
    st.caption("Odds data is not archived automatically. Export snapshots if you want to preserve a record. Do not share API keys or private app secrets.")
except requests.RequestException:
    st.error("Network error fetching FanDuel odds. Your other SDS pages are unaffected.")
except Exception as exc:
    st.error("SDS 4.6 could not complete the comparison. Existing pages remain unaffected.")
    st.caption(f"{type(exc).__name__}: {exc}")
