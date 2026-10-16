"""SDS Sports Model — EPA-based experimental game projections (independent page)."""
import math
import numpy as np
import pandas as pd
import streamlit as st
import nflreadpy as nfl

st.set_page_config(page_title="SDS Advanced Game Model", page_icon="🏈", layout="wide")
st.title("🏈 SDS Advanced Game Model")
st.caption("Experimental EPA + success-rate baseline. Does not use live odds or injury reports.")
st.warning("Research model only. Backtest errors do not establish a profitable betting edge.")

START_SEASON = 2022
HOLDOUT = 2025
LAST_TRAIN = 2025
FEATURE_NAMES = ["off_epa_diff", "def_epa_diff", "off_success_diff",
                 "def_success_diff", "off_epa_coverage", "def_epa_coverage"]

@st.cache_data(ttl=21600, show_spinner=False)
def load_data():
    schedule = nfl.load_schedules(list(range(START_SEASON, 2027))).to_pandas()
    chunks = []
    for year in range(START_SEASON, LAST_TRAIN + 1):
        pbp = nfl.load_pbp([year]).to_pandas()
        wanted = ["game_id", "posteam", "defteam", "epa", "success", "play_type"]
        absent = set(wanted) - set(pbp.columns)
        if absent:
            raise ValueError(f"Missing play-by-play columns for {year}: {sorted(absent)}")
        pbp = pbp[wanted]
        pbp = pbp[pbp["play_type"].isin(["run", "pass"]) &
                  pbp["posteam"].notna() & pbp["defteam"].notna() &
                  pbp["epa"].notna()]
        chunks.append(pbp.groupby(["game_id", "posteam", "defteam"], as_index=False).agg(
            off_epa=("epa", "mean"),
            off_success=("success", "mean"),
            plays=("epa", "size")))
    return schedule, pd.concat(chunks, ignore_index=True)

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

try:
    with st.spinner("Loading NFL play-by-play data (first load may take several minutes)..."):
        schedules, pbp_stats = load_data()
        data = assemble(schedules, pbp_stats)
    completed = data[data.completed].copy()
    training = completed[completed.season.between(START_SEASON, HOLDOUT-1)]
    testing = completed[completed.season.eq(HOLDOUT)]
    if len(training) < 300:
        st.error("Insufficient historical games for training.")
        st.stop()
    st.subheader("2025 holdout evaluation")
    if testing.empty:
        st.info("No completed 2025 games available to evaluate.")
    else:
        mb, msd = ridge_fit(training, "margin")
        tb, tsd = ridge_fit(training, "total")
        margins = forecast(testing, mb)
        totals = forecast(testing, tb)
        actual_margin = testing.margin.to_numpy()
        actual_total = testing.total.to_numpy()
        # Avoid giving a tie a forced winner.
        non_ties = actual_margin != 0
        accuracy = np.mean((margins[non_ties] > 0) == (actual_margin[non_ties] > 0)) if non_ties.any() else float("nan")
        a, b, c = st.columns(3)
        a.metric("Spread MAE", f"{np.mean(np.abs(margins-actual_margin)):.2f} pts")
        b.metric("Total MAE", f"{np.mean(np.abs(totals-actual_total)):.2f} pts")
        c.metric("Winner accuracy", f"{100*accuracy:.1f}%")
        st.caption("Training: 2022–2024 regular season. Holdout: 2025 regular season. "
                   "These metrics are not sportsbook profitability or calibrated probability scores.")
    final_train = completed[completed.season.between(START_SEASON, LAST_TRAIN)]
    mb, margin_sd = ridge_fit(final_train, "margin")
    tb, _ = ridge_fit(final_train, "total")
    upcoming = data[(data.season.eq(2026)) & (~data.completed)].copy()
    st.subheader("2026 game projections")
    if upcoming.empty:
        st.info("No unplayed 2026 regular-season matchups were found in the schedule feed.")
    else:
        upcoming["Projected home margin"] = forecast(upcoming, mb)
        upcoming["Projected total"] = forecast(upcoming, tb)
        upcoming["Home win % (uncalibrated)"] = [
            round(100*normal_cdf(x/margin_sd), 1)
            for x in upcoming["Projected home margin"]]
        upcoming["Away win % (uncalibrated)"] = 100-upcoming["Home win % (uncalibrated)"]
        display = upcoming.rename(columns={"week":"Week","home":"Home","away":"Away"})
        columns = ["Week", "Away", "Home", "Projected home margin",
                   "Projected total", "Home win % (uncalibrated)",
                   "Away win % (uncalibrated)"]
        st.dataframe(display[columns].round(1), hide_index=True, use_container_width=True)
        st.download_button("Download advanced projections", display[columns].to_csv(index=False),
                           file_name="sds_advanced_game_projections.csv", mime="text/csv")
    with st.expander("Method and limitations"):
        st.markdown("""
        - **Inputs:** Offensive/defensive EPA per run or pass play, and play success rate.
        - **Timing:** Features use only earlier weeks; current-week results are never used.
        - **Window:** Last five games with available play-by-play, carried into the next season.
        - **Training:** Ridge regression for point margin and game total.
        - **Probability:** Normal-residual approximation; **not calibrated**.
        - **Missing:** Quarterback, injuries, weather, opponent adjustment, market lines,
          home/away-specific features, closing-line comparisons, and uncertainty intervals.
        - **Data quality:** If play-by-play is missing, the feature history can be sparse.
          A strong production model needs walk-forward evaluation across multiple seasons.
        """)
except Exception as error:
    st.error("The advanced model could not load NFL data. Your main dashboard is unaffected.")
    st.code(f"{type(error).__name__}: {error}")
