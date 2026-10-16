"""SDS Sports Model 4.0: consolidated, read-only research dashboard."""
import math
import pandas as pd
import streamlit as st
import nflreadpy as nfl

st.set_page_config(page_title="SDS Command Center", page_icon="🏈", layout="wide")
st.title("🏈 SDS Sports Model | Command Center")
st.caption("Experimental NFL projections, historical validation, and manual odds comparison.")
st.warning("Research tool only. Projections are not validated betting edges; no live sportsbook feed is connected.")

@st.cache_data(ttl=3600, show_spinner=False)
def load_schedules():
    return nfl.load_schedules(list(range(2021, 2027))).to_pandas()

@st.cache_data(ttl=3600, show_spinner=False)
def prepare():
    schedule = load_schedules()
    if "game_type" in schedule.columns:
        schedule = schedule[schedule["game_type"].eq("REG")]
    required = {"season", "week", "home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(schedule.columns):
        raise ValueError(f"Schedule missing columns: {sorted(required - set(schedule.columns))}")
    schedule = schedule.sort_values(["season", "week"]).copy()
    history = {}
    records = []
    for (season, week), group in schedule.groupby(["season", "week"], sort=True):
        updates = []
        for g in group.itertuples(index=False):
            home, away = str(g.home_team), str(g.away_team)
            def prior(team):
                games = history.get(team, [])[-5:]
                if not games:
                    return (0.0, 0.0, 0.0)
                return (sum(x[0] for x in games)/len(games),
                        sum(x[1] for x in games)/len(games), len(games)/5)
            h, a = prior(home), prior(away)
            # Conservative baseline; coefficients deliberately not claimed to be optimized.
            margin = 1.5 + 0.40*(h[0]-a[1]) - 0.40*(a[0]-h[1])
            total = 44.0 + 0.35*((h[0]+a[0])-44.0)
            completed = pd.notna(g.home_score) and pd.notna(g.away_score)
            records.append({"Season": int(season), "Week": int(week), "Away": away,
                            "Home": home, "Projected home margin": round(margin,1),
                            "Projected total": round(total,1), "Completed": completed,
                            "Actual home margin": float(g.home_score-g.away_score) if completed else None,
                            "Actual total": float(g.home_score+g.away_score) if completed else None,
                            "History available": h[2] > 0 and a[2] > 0})
            if completed:
                updates.extend([(home, (float(g.home_score),float(g.away_score))),
                                (away, (float(g.away_score),float(g.home_score)))])
        for team, score in updates:
            history.setdefault(team, []).append(score)
    return pd.DataFrame(records)

def american_to_probability(odds):
    if odds == 0:
        raise ValueError("American odds cannot be zero.")
    return 100/(odds+100) if odds > 0 else -odds/(-odds+100)

def decimal_profit(odds):
    return odds/100 if odds > 0 else 100/(-odds)

tab1, tab2, tab3, tab4 = st.tabs(["Predictions", "Performance", "Odds Calculator", "About"])
with tab1:
    st.subheader("Upcoming NFL matchups")
    st.caption("Score-based illustrative baseline, not the Advanced EPA engine. No calibrated win probabilities.")
    try:
        games = prepare()
        upcoming = games[(games.Season == 2026) & (~games.Completed)].copy()
        if upcoming.empty:
            st.info("No unplayed 2026 regular-season matchups were returned.")
        else:
            weeks = sorted(upcoming.Week.unique().tolist())
            week = st.selectbox("NFL week", weeks)
            display = upcoming[upcoming.Week == week]
            columns = ["Week", "Away", "Home", "Projected home margin", "Projected total"]
            st.dataframe(display[columns], hide_index=True, use_container_width=True)
            st.download_button("Download projections CSV", display[columns].to_csv(index=False),
                               file_name="sds_week_projections.csv", mime="text/csv")
    except Exception as error:
        st.error("NFL schedule data could not be loaded.")
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
    st.info("These metrics came from the earlier model comparison. The Predictions tab above uses a separate illustrative formula. Do not attribute these scores to it.")
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
- **This page:** Unified navigation, recorded model comparison, and manual moneyline EV calculator.
- **Not yet implemented:** Verified starting QB adjustments, rest and schedule strength, calibrated win probabilities, automatic FanDuel lines, and audited bet tracking.
- **Next:** A shared feature pipeline and time-ordered quarterback-aware backtest. Promote models only after independent holdout improvements.

Existing SDS pages remain available; this add-on does not change them.
""")
