"""SDS Command Center 4.2: current-season EPA with historical fallback."""
import math
import numpy as np
import pandas as pd
import streamlit as st
import nflreadpy as nfl

st.set_page_config(page_title="SDS Command Center", page_icon="🏈", layout="wide")
st.title("🏈 SDS Sports Model | Command Center")
st.caption("Experimental EPA-based NFL projections, recorded model evaluation, and manual odds calculations.")
st.warning("Research tool only. Win probabilities are uncalibrated; no live sportsbook odds.")

START_SEASON = 2022
HOLDOUT = 2025
LAST_TRAIN = 2025
CURRENT_SEASON = 2026

@st.cache_data(ttl=21600, show_spinner=False)
def load_data():
    schedule = nfl.load_schedules(list(range(START_SEASON, CURRENT_SEASON + 1))).to_pandas()
    chunks = []
    current_status = "2026 play-by-play available"
    current_games = 0
    for year in range(START_SEASON, CURRENT_SEASON + 1):
        try:
            pbp = nfl.load_pbp([year]).to_pandas()
            wanted = ["game_id", "posteam", "defteam", "epa", "success", "play_type"]
            absent = set(wanted) - set(pbp.columns)
            if absent:
                raise ValueError(f"Missing play-by-play fields: {sorted(absent)}")
            pbp = pbp[wanted]
            pbp = pbp[pbp["play_type"].isin(["run", "pass"]) &
                      pbp["posteam"].notna() & pbp["defteam"].notna() &
                      pbp["epa"].notna()]
            aggregate = pbp.groupby(["game_id", "posteam", "defteam"], as_index=False).agg(
                off_epa=("epa", "mean"), off_success=("success", "mean"),
                plays=("epa", "size"))
            if year == CURRENT_SEASON:
                current_games = int(aggregate["game_id"].nunique())
                if current_games == 0:
                    current_status = "2026 play-by-play contains no completed games yet"
            chunks.append(aggregate)
        except Exception as exc:
            if year != CURRENT_SEASON:
                raise
            current_status = f"2026 play-by-play unavailable ({type(exc).__name__}); using older EPA history"
    if not chunks:
        raise ValueError("No NFL play-by-play statistics available")
    return schedule, pd.concat(chunks, ignore_index=True), current_status, current_games

def assemble(schedule, play_stats):
    games = schedule.copy()
    if "game_type" in games.columns:
        games = games[games["game_type"].eq("REG")]
    needed = ["game_id", "season", "week", "home_team", "away_team",
              "home_score", "away_score"]
    absent = set(needed) - set(games.columns)
    if absent:
        raise ValueError(f"Missing schedule columns: {sorted(absent)}")
    games = games.sort_values(["season", "week", "game_id"])
    # One row per offense per game; the defense is its opponent.
    stats = {}
    for r in play_stats.itertuples(index=False):
        stats[(str(r.game_id), str(r.posteam))] = (float(r.off_epa),
            float(r.off_success) if pd.notna(r.off_success) else 0.0, int(r.plays))
    history = {}
    results = []
    for (season, week), group in games.groupby(["season", "week"], sort=True):
        updates = []
        for g in group.itertuples(index=False):
            home, away, gid = str(g.home_team), str(g.away_team), str(g.game_id)
            def prior(team):
                # Recent games, excluding all games from the current week.
                h = history.get(team, [])[-5:]
                if not h:
                    return [0.0, 0.0, 0.0, 0.0, 0.0]
                arr = np.asarray(h, dtype=float)
                return [float(v) for v in arr.mean(axis=0)[:4]] + [len(h)/5.0]
            h, a = prior(home), prior(away)
            features = [h[0]-a[0], h[1]-a[1], h[2]-a[2],
                        h[3]-a[3], h[4], a[4]]
            completed = pd.notna(g.home_score) and pd.notna(g.away_score)
            results.append(dict(season=int(season), week=int(week), home=home,
                away=away, game_id=gid, completed=completed, features=features,
                margin=float(g.home_score-g.away_score) if completed else np.nan,
                total=float(g.home_score+g.away_score) if completed else np.nan))
            if completed:
                hs, az = stats.get((gid, home)), stats.get((gid, away))
                # Only update efficiency features for games with BOTH teams' stats.
                if hs and az and hs[2] >= 10 and az[2] >= 10:
                    updates.append((home, [hs[0], az[0], hs[1], az[1]]))
                    updates.append((away, [az[0], hs[0], az[1], hs[1]]))
        for team, values in updates:
            history.setdefault(team, []).append(values)
    return pd.DataFrame(results)

def ridge_fit(df, target, penalty=15.0):
    X = np.asarray(df.features.tolist(), dtype=float)
    X = np.column_stack([np.ones(len(X)), X])
    y = df[target].to_numpy(dtype=float)
    regularizer = np.diag([0.0] + [penalty] * (X.shape[1]-1))
    beta = np.linalg.solve(X.T @ X + regularizer, X.T @ y)
    sd = max(float(np.std(y-X@beta, ddof=min(X.shape[1],len(y)-1))), 1.0)
    return beta, sd

def forecast(df, beta):
    X = np.asarray(df.features.tolist(), dtype=float)
    return np.column_stack([np.ones(len(X)), X]) @ beta

def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))

def american_to_probability(odds):
    if odds == 0:
        raise ValueError("American odds cannot be zero.")
    return 100/(odds+100) if odds > 0 else -odds/(-odds+100)

def decimal_profit(odds):
    return odds/100 if odds > 0 else 100/(-odds)

tab1, tab2, tab3, tab4 = st.tabs(["Predictions", "Performance", "Odds Calculator", "About"])
with tab1:
    st.subheader("Advanced EPA game projections")
    st.caption("2026 completed-game EPA and success rates are included when available. "
               "Predictions use previous weeks only; no quarterback adjustment yet.")
    try:
        with st.spinner("Loading NFL play-by-play data. The first load may take several minutes..."):
            schedules, pbp_stats, current_status, current_games = load_data()
            data = assemble(schedules, pbp_stats)
        st.caption(f"Data status: {current_status}. 2026 games with play-by-play: {current_games}.")
        if "unavailable" in current_status:
            st.warning("Current-season efficiency data could not be retrieved. "
                       "Projections fall back to older team form and may be stale.")
        completed = data[data.completed & data.season.between(START_SEASON, LAST_TRAIN)]
        if len(completed) < 300:
            st.error("Insufficient historical games to train the model.")
        else:
            margin_beta, margin_sd = ridge_fit(completed, "margin")
            total_beta, _ = ridge_fit(completed, "total")
            upcoming = data[(data.season.eq(2026)) & (~data.completed)].copy()
            if upcoming.empty:
                st.info("No unplayed 2026 regular-season matchups were returned.")
            else:
                upcoming["Projected home margin"] = forecast(upcoming, margin_beta)
                upcoming["Projected total"] = forecast(upcoming, total_beta)
                upcoming["Home win % (uncalibrated)"] = [
                    round(100 * normal_cdf(v / margin_sd), 1)
                    for v in upcoming["Projected home margin"]]
                upcoming["Away win % (uncalibrated)"] = (
                    100 - upcoming["Home win % (uncalibrated)"])
                upcoming = upcoming.rename(columns={
                    "week": "Week", "home": "Home", "away": "Away"})
                week = st.selectbox("NFL week", sorted(upcoming["Week"].unique().tolist()))
                display = upcoming[upcoming["Week"].eq(week)]
                columns = ["Week", "Away", "Home", "Projected home margin",
                           "Projected total", "Home win % (uncalibrated)",
                           "Away win % (uncalibrated)"]
                st.dataframe(display[columns].round(1), hide_index=True,
                             use_container_width=True)
                st.download_button("Download model projections CSV",
                    display[columns].to_csv(index=False),
                    file_name="sds_epa_week_projections.csv", mime="text/csv")
                st.info("This is an EPA model projection, not a validated betting edge. "
                        "Starting-quarterback changes are not incorporated. "
                        "Current-season data is included only when available.")
    except Exception as error:
        st.error("Advanced EPA data could not be loaded. Other SDS pages remain available.")
        st.caption(f"{type(error).__name__}: {error}")
with tab2:
    st.subheader("Recorded 2024–2025 validation results")
    st.caption("Manually entered from your previous Validation Lab results. Not recalculated here.")
    comparison = pd.DataFrame([
        {"Model":"Baseline", "Spread MAE":10.31, "Total MAE":10.58,
         "Winner accuracy (%)":60.0, "Brier":0.2260},
        {"Model":"Advanced EPA", "Spread MAE":10.22, "Total MAE":10.60,
         "Winner accuracy (%)":63.5, "Brier":0.2224},
    ])
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.info("These metrics came from the earlier model comparison. The Predictions tab now uses the Advanced EPA pipeline, but these recorded scores come from a separate walk-forward validation and should not be presented as a live performance audit.")
with tab3:
    st.subheader("Manual moneyline value calculator")
    st.caption("Enter YOUR independently estimated win probability and the sportsbook's American odds. No odds are fetched automatically.")
    probability = st.number_input("Your estimated win probability (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.5)
    odds = st.number_input("American odds", value=-110, step=5)
    stake = st.number_input("Illustrative stake ($)", min_value=1.0, value=10.0, step=1.0)
    if odds == 0:
        st.error("Odds cannot equal zero.")
    else:
        p = probability / 100
        implied = american_to_probability(odds)
        profit = decimal_profit(odds)
        ev = p*profit - (1-p)
        a, b, c = st.columns(3)
        a.metric("Raw implied probability", f"{implied*100:.1f}%")
        b.metric("Expected return / $1", f"{ev*100:+.1f}%")
        c.metric("Illustrative expected P/L", f"${ev*stake:+.2f}")
        st.caption("Single-outcome implied probability includes bookmaker margin. EV is meaningful only if the entered probability is reliable. No betting recommendation is made.")
with tab4:
    st.markdown("""### SDS development status
- **Current:** Historical NFL statistics and experimental predictions.
- **This page:** Advanced EPA projections, recorded model comparison, and manual moneyline EV calculator.
- **Not yet implemented:** Verified starting QB adjustments, rest and schedule strength, calibrated win probabilities, automatic FanDuel lines, and audited bet tracking.
- **This update:** Attempts to load completed 2026 EPA data; safely falls back if unavailable.
- **Next:** Time-ordered quarterback-aware backtests, probability calibration, and weekly prediction snapshots. Promote models only after independent holdout improvements.

Existing SDS pages remain available; this add-on does not change them.
""")