"""SDS 5.1 — fan-friendly NFL Game Center. Read-only; no database writes."""
import math
import pandas as pd
import streamlit as st
from sds50_engine import (
    TEAM_NAMES, load_model_data, assemble, ridge_fit, forecast,
    normal_cdf, fit_2024_calibrator, sigmoid, logit,
)

st.set_page_config(page_title="SDS 5.1 | Game Center", page_icon="🏈", layout="wide")
st.title("🏈 SDS Sports Model | Game Center")
st.caption("Simple NFL predictions, backed by SDS team-efficiency research.")
st.warning("Experimental forecasts, not betting advice. Injuries, confirmed starting QBs, and late roster news are not modeled.")

FULL_NAMES = {v: k for k, v in TEAM_NAMES.items()}

def friendly(team):
    return FULL_NAMES.get(str(team), str(team))


def kickoff_label(value):
    if pd.isna(value):
        return "Kickoff time unavailable"
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("America/New_York").strftime("%a, %b %-d · %-I:%M %p ET")


def explain(row):
    """Only describe actual model inputs; never invent QB/injury data."""
    f = row["features"]
    if not isinstance(f, (list, tuple)) or len(f) < 6:
        return "SDS uses recent offensive and defensive team efficiency to estimate the result."
    off = float(f[0])
    defense = float(f[1])
    winner = row["home"] if row["home_prob"] >= 0.5 else row["away"]
    reasons = []
    # Differences are home minus away; higher offense EPA is positive for home.
    if abs(off) >= 0.025:
        fav_off = row["home"] if off > 0 else row["away"]
        if fav_off == winner:
            reasons.append(f"{friendly(fav_off)} has the stronger recent offensive EPA profile")
    # Defensive EPA allowed: lower is better. The difference is home minus away.
    if abs(defense) >= 0.025:
        fav_def = row["away"] if defense > 0 else row["home"]
        if fav_def == winner:
            reasons.append(f"{friendly(fav_def)} has allowed less EPA per play recently")
    if reasons:
        return "; ".join(reasons).capitalize() + ". These are team-level trends, not injury-adjusted scouting reports."
    return "Recent team-efficiency indicators are relatively close or mixed. The forecast also reflects historical scoring patterns."


@st.cache_data(ttl=21600, show_spinner=False)
def build_predictions():
    schedules, pbp, status, current_games = load_model_data()
    games = assemble(schedules, pbp)
    training = games[games.completed & games.season.between(2022, 2025)]
    if len(training) < 300:
        raise ValueError("Not enough completed historical games to train SDS.")
    margin_beta, margin_sd = ridge_fit(training, "margin")
    total_beta, _ = ridge_fit(training, "total")
    calibrator = fit_2024_calibrator(games)
    upcoming = games[games.season.eq(2026) & ~games.completed].copy()
    if upcoming.empty:
        return upcoming, status, current_games
    upcoming["home_margin"] = forecast(upcoming, margin_beta)
    upcoming["total_points"] = forecast(upcoming, total_beta)
    upcoming["raw_home"] = [normal_cdf(v / margin_sd) for v in upcoming.home_margin]
    upcoming["home_prob"] = sigmoid(calibrator[0] + calibrator[1] * logit(upcoming.raw_home))
    upcoming["away_prob"] = 1 - upcoming.home_prob
    upcoming["home_score_pred"] = (upcoming.total_points + upcoming.home_margin) / 2
    upcoming["away_score_pred"] = (upcoming.total_points - upcoming.home_margin) / 2
    return upcoming, status, current_games


try:
    with st.spinner("Preparing this week's NFL matchups..."):
        games, status, current_games = build_predictions()
except Exception as exc:
    st.error(f"Unable to load the Game Center: {type(exc).__name__}: {exc}")
    st.info("Your existing SDS pages and saved predictions are unaffected.")
    st.stop()

if "unavailable" in status.lower():
    st.warning("Current-season EPA is unavailable. SDS is using historical team data; predictions may be stale.")

if games.empty:
    st.info("No upcoming 2026 regular-season games are available in the NFL schedule feed.")
    st.stop()

# Only show games not already past kickoff where kickoff information exists.
now = pd.Timestamp.now(tz="UTC")
known_kickoff = pd.to_datetime(games.kickoff, utc=True, errors="coerce")
active = games[known_kickoff.isna() | (known_kickoff > now)].copy()
if active.empty:
    st.info("No future regular-season games remain on the current schedule.")
    st.stop()

weeks = sorted(int(w) for w in active.week.dropna().unique())
week = st.selectbox("Choose an NFL week", weeks, index=0, format_func=lambda w: f"Week {w}")
view = active[active.week.eq(week)].sort_values(["kickoff", "game_id"], na_position="last")
st.caption(f"{len(view)} upcoming games · 2026 season · 2026 games with EPA data: {current_games}")

st.subheader(f"Week {week} matchups")
for _, r in view.iterrows():
    away, home = str(r.away), str(r.home)
    hp = float(r.home_prob)
    ap = float(r.away_prob)
    winner = home if hp >= ap else away
    chance = max(hp, ap) * 100
    # Rounded score estimates are descriptive, not a discrete scoreline probability.
    away_pts = max(0, round(float(r.away_score_pred)))
    home_pts = max(0, round(float(r.home_score_pred)))
    with st.container(border=True):
        st.caption(kickoff_label(r.kickoff))
        st.markdown(f"#### {friendly(away)} @ {friendly(home)}")
        st.markdown(f"**SDS leans: {friendly(winner)}** · {chance:.1f}% estimated chance to win")
        st.progress(min(1.0, max(0.0, chance / 100)), text=f"{friendly(winner)} win probability: {chance:.1f}%")
        c1, c2 = st.columns(2)
        c1.metric(f"{friendly(away)}", f"{ap*100:.1f}%", help="Estimated chance of winning")
        c2.metric(f"{friendly(home)}", f"{hp*100:.1f}%", help="Estimated chance of winning")
        st.write(f"**Estimated score:** {friendly(away)} {away_pts} — {friendly(home)} {home_pts}")
        st.caption("Scores are rounded estimates from separate margin and total models; exact scores are not predicted with certainty.")
        st.write("**Why SDS leans this way:** " + explain(r))
        with st.expander("Advanced analytics"):
            st.write(f"Projected home margin: **{float(r.home_margin):+.2f} points**")
            st.write(f"Projected game total: **{float(r.total_points):.2f} points**")
            st.write(f"Original home win probability: **{float(r.raw_home)*100:.2f}%**")
            st.write(f"Calibrated home win probability: **{hp*100:.2f}%**")
            st.caption("Calibrator was trained on 2024 results and evaluated on 2025. Historical improvement does not guarantee future accuracy.")

with st.expander("What do these predictions mean?"):
    st.markdown("""
**Win probability** is the model's estimate of how often a team would win in similar situations. A 65% estimate is not a guarantee.

**Estimated score** is a rounded projection based on separate expected-margin and total-points models. It is not an exact-score prediction.

**EPA** (expected points added) measures how much a play changes a team's expected scoring. SDS summarizes recent offensive and defensive EPA to describe team form.

**Calibrated** means historical outcomes were used to adjust probability estimates. The adjustment was tested on a separate 2025 season, but future performance remains uncertain.

SDS currently does **not** incorporate confirmed starters, injuries, weather, or late-breaking roster news. These forecasts should not be treated as verified betting edges.
""")

export = view.copy()
export["away_name"] = export.away.map(friendly)
export["home_name"] = export.home.map(friendly)
export["home_win_pct"] = (100 * export.home_prob).round(2)
export["away_win_pct"] = (100 * export.away_prob).round(2)
export["projected_home_score"] = export.home_score_pred.round(1)
export["projected_away_score"] = export.away_score_pred.round(1)
fields = ["game_id", "week", "kickoff", "away_name", "home_name", "away_win_pct", "home_win_pct", "projected_away_score", "projected_home_score"]
st.download_button("Download Week predictions (CSV)", export[fields].to_csv(index=False), file_name=f"sds51_week_{week}.csv", mime="text/csv")
st.caption("SDS 5.1 is read-only. It does not save, change, or grade Supabase records, and it does not request FanDuel odds.")
