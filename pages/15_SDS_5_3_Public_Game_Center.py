"""SDS 5.3 — public-friendly NFL Game Center. Read-only; no database writes."""
import math
import html
import pandas as pd
import streamlit as st
from sds50_engine import (
    TEAM_NAMES, load_model_data, assemble, ridge_fit, forecast,
    normal_cdf, fit_2024_calibrator, sigmoid, logit,
)

st.set_page_config(page_title="SDS 5.3 | Game Center", page_icon="🏈", layout="wide")
st.title("🏈 SDS | Game Center 5.3")
st.caption("NFL predictions made simple · See the numbers, then explore why.")
st.caption("Experimental model estimates, not betting picks. Injuries, confirmed QBs and late roster news are not included.")

FULL_NAMES = {v: k for k, v in TEAM_NAMES.items()}
# ESPN's publicly served team marks. A text abbreviation remains visible if an image fails.
LOGO_CODES = {"ARI":"ari", "ATL":"atl", "BAL":"bal", "BUF":"buf", "CAR":"car", "CHI":"chi", "CIN":"cin", "CLE":"cle", "DAL":"dal", "DEN":"den", "DET":"det", "GB":"gb", "HOU":"hou", "IND":"ind", "JAX":"jax", "KC":"kc", "LV":"lv", "LAC":"lac", "LA":"lar", "MIA":"mia", "MIN":"min", "NE":"ne", "NO":"no", "NYG":"nyg", "NYJ":"nyj", "PHI":"phi", "PIT":"pit", "SF":"sf", "SEA":"sea", "TB":"tb", "TEN":"ten", "WAS":"wsh"}

def logo(team):
    code = LOGO_CODES.get(str(team))
    if not code:
        return f'<span class="sds-fallback">{html.escape(str(team))}</span>'
    return (f'<span class="sds-logo-wrap"><span class="sds-fallback">{html.escape(str(team))}</span>'
            f'<img src="https://a.espncdn.com/i/teamlogos/nfl/500/{code}.png" '
            f'alt="{html.escape(friendly(team), quote=True)} logo" loading="lazy"></span>')

def lean_label(prob):
    # Describes probability distance from 50%, NOT empirically validated confidence.
    edge = abs(float(prob) - 0.5)
    if edge < 0.08:
        return "Close matchup"
    if edge < 0.20:
        return "Moderate model lean"
    return "Strong model lean"


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


st.markdown("""
<style>
.block-container {padding-top: 1.1rem; max-width: 980px;}
.sds-card {border: 1px solid rgba(145,155,175,.30); border-radius: 16px; padding: 15px 16px; margin: 0 0 12px; background: rgba(125,135,155,.055);}
.sds-time {font-size: .78rem; opacity:.72; margin-bottom: 8px;}
.sds-teams {display:grid;grid-template-columns: minmax(0,1fr) auto minmax(0,1fr); gap:10px;align-items:center;}
.sds-side {min-width:0;}
.sds-side.right {text-align:right;}
.sds-team-head {display:flex;align-items:center;gap:8px;min-height:37px;}
.sds-side.right .sds-team-head {flex-direction:row-reverse;}
.sds-logo-wrap {position:relative;display:inline-flex;align-items:center;justify-content:center;width:32px;height:32px;flex:0 0 32px;}
.sds-logo-wrap img {position:absolute;inset:0;width:32px;height:32px;object-fit:contain;}
.sds-fallback {font-size:.65rem;font-weight:800;opacity:.6;}
.sds-chip {display:inline-block;font-size:.72rem;border-radius:999px;padding:3px 8px;border:1px solid rgba(145,155,175,.4);margin-top:9px;}
.sds-name {font-size:clamp(.95rem,3.3vw,1.13rem);font-weight:700;line-height:1.3;overflow-wrap:anywhere;}
.sds-pct {font-size:1.65rem;font-weight:750;line-height:1.4;}
.sds-score {font-size:.82rem;opacity:.8;}
.sds-at {font-size:.83rem;opacity:.5;}
.sds-track {display:flex; height:9px;border-radius:20px;overflow:hidden;margin:12px 0 10px;background:rgba(125,135,155,.25);}
.sds-away {background:#60a5fa;}
.sds-home {background:#f59e0b;}
.sds-bottom {font-size:.86rem;display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;}
.sds-pick {font-weight:700;}
.sds-subtle {opacity:.68;}
@media(max-width:450px){.sds-card{padding:12px}.sds-pct{font-size:1.42rem}.sds-bottom{font-size:.78rem}.sds-teams{gap:6px}}
</style>
""", unsafe_allow_html=True)

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
st.caption("Blue = away · Gold = home · Percentages are estimates, not guarantees.")
for _, r in view.iterrows():
    away, home = str(r.away), str(r.home)
    hp, ap = float(r.home_prob), float(r.away_prob)
    winner = home if hp >= ap else away
    away_pts = max(0, round(float(r.away_score_pred)))
    home_pts = max(0, round(float(r.home_score_pred)))
    esc = lambda v: html.escape(str(v), quote=True)
    markup = f"""
<div class="sds-card">
  <div class="sds-time">{esc(kickoff_label(r.kickoff))} · Week {week}</div>
  <div class="sds-teams">
    <div class="sds-side"><div class="sds-team-head">{logo(away)}<div class="sds-name">{esc(friendly(away))}</div></div><div class="sds-pct">{ap*100:.1f}%</div><div class="sds-score">Projected: {away_pts} pts</div></div>
    <div class="sds-at">AT</div>
    <div class="sds-side right"><div class="sds-team-head">{logo(home)}<div class="sds-name">{esc(friendly(home))}</div></div><div class="sds-pct">{hp*100:.1f}%</div><div class="sds-score">Projected: {home_pts} pts</div></div>
  </div>
  <div class="sds-track" role="img" aria-label="Away win probability {ap*100:.1f} percent; home win probability {hp*100:.1f} percent"><div class="sds-away" style="width:{ap*100:.4f}%"></div><div class="sds-home" style="width:{hp*100:.4f}%"></div></div>
  <div class="sds-bottom"><span class="sds-pick">SDS favors {esc(friendly(winner))}</span><span class="sds-subtle">Projected: {away_pts}–{home_pts}</span></div>
  <div class="sds-chip">{lean_label(max(hp, ap))}</div>
</div>
"""
    st.markdown(markup, unsafe_allow_html=True)
    with st.expander(f"Why SDS favors {friendly(winner)}"):
        st.write(explain(r))
        st.caption(f"{lean_label(max(hp, ap))} describes how far the estimated probability is from 50%. It does not measure proven forecast reliability.")
        st.caption("The model uses recent team efficiency and historical scoring; it does not know confirmed QBs, injuries or weather.")
        with st.expander("Advanced model numbers"):
            st.write(f"Projected home margin: **{float(r.home_margin):+.2f} points**")
            st.write(f"Projected total: **{float(r.total_points):.2f} points**")
            st.write(f"Original home win chance: **{float(r.raw_home)*100:.2f}%**")
            st.write(f"Calibrated home win chance: **{hp*100:.2f}%**")
            st.caption("Probability calibration was fitted on 2024 and evaluated on 2025. Future accuracy is not assured.")

with st.expander("New here? Understand the predictions"):
    st.markdown("""
**Win chance:** An estimated likelihood, not a guarantee. A team with 70% can still lose.

**Projected points:** A rough score estimate derived from separate spread and total models, not an exact score prediction.

**SDS favors:** The team with the higher modeled win chance. It is **not** a betting recommendation.

**Model lean strength:** Close / moderate / strong only describes how far a forecast is from 50%. It is **not** a measured confidence rating, and does not establish a betting edge.

**EPA:** Expected Points Added; a way to measure how efficiently teams perform on each play.

**Calibrated:** Probabilities adjusted using prior-season results, then checked on a different season.
""")

export = view.copy()
export["away_name"] = export.away.map(friendly)
export["home_name"] = export.home.map(friendly)
export["home_win_pct"] = (100 * export.home_prob).round(2)
export["away_win_pct"] = (100 * export.away_prob).round(2)
export["projected_home_score"] = export.home_score_pred.round(1)
export["projected_away_score"] = export.away_score_pred.round(1)
fields = ["game_id", "week", "kickoff", "away_name", "home_name", "away_win_pct", "home_win_pct", "projected_away_score", "projected_home_score"]
st.download_button("Download Week predictions (CSV)", export[fields].to_csv(index=False), file_name=f"sds53_week_{week}.csv", mime="text/csv")
st.caption("SDS 5.3 is read-only. No Supabase records are created or modified, and no sportsbook API is called.")