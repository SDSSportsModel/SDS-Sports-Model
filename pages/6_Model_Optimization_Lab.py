"""SDS 4.4: rest-day feature experiment and fixed-weight ensemble backtest."""
import math
import numpy as np
import pandas as pd
import streamlit as st
import nflreadpy as nfl

st.set_page_config(page_title="SDS 4.4 Model Optimization", page_icon="🏈", layout="wide")
st.title("🏈 SDS Model Optimization Lab — 4.4")
st.warning("Experimental backtest only; not validated for wagering. No sportsbook lines.")
st.caption("Evaluates identical completed 2024–2025 games, training on prior seasons only.")
SEASONS = [2022, 2023, 2024, 2025]
EVAL_YEARS = [2024, 2025]
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_data():
    schedule = nfl.load_schedules(SEASONS).to_pandas()
    needed = {"game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"}
    if not needed.issubset(schedule.columns):
        raise ValueError(f"Missing schedule fields: {sorted(needed-set(schedule.columns))}")
    if "game_type" in schedule:
        schedule = schedule[schedule.game_type.eq("REG")]
    schedule = schedule.dropna(subset=["home_score", "away_score"]).sort_values(["season", "week", "game_id"])
    team_game = {}
    for season in SEASONS:
        pbp = nfl.load_pbp([season]).to_pandas()
        required = {"game_id", "posteam", "defteam", "play_type", "epa", "success", }
        if not required.issubset(pbp.columns):
            raise ValueError(f"Missing play-by-play fields in {season}: {sorted(required-set(pbp.columns))}")
        plays = pbp.loc[
            pbp.play_type.isin(["run", "pass"]) & pbp.posteam.notna() &
            pbp.defteam.notna() & pbp.epa.notna(),
            ["game_id", "posteam", "epa", "success", "play_type"]
        ].copy()
        team = plays.groupby(["game_id", "posteam"], as_index=False).agg(
            epa=("epa", "mean"), success=("success", "mean"), n=("epa", "size"))
        for row in team.itertuples(index=False):
            team_game[(str(row.game_id), str(row.posteam))] = (
                float(row.epa), float(row.success) if pd.notna(row.success) else 0., int(row.n))
    return schedule, team_game, None

def rolling(history, team, ncols):
    records = history.get(team, [])[-5:]
    if not records:
        return [0.] * ncols + [0.]
    means = np.asarray(records, dtype=float).mean(axis=0)
    return [float(v) for v in means] + [len(records)/5.]

def ridge(train, test, model, target, alpha=20.):
    x = np.asarray(train[model].tolist(), dtype=float)
    z = np.asarray(test[model].tolist(), dtype=float)
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd < 1e-8] = 1.
    x = np.column_stack([np.ones(len(x)), (x-mu)/sd])
    z = np.column_stack([np.ones(len(z)), (z-mu)/sd])
    y = train[target].to_numpy(dtype=float)
    regularizer = np.diag([0.] + [alpha]*(x.shape[1]-1))
    beta = np.linalg.solve(x.T@x + regularizer, x.T@y)
    residual_sd = max(float(np.std(y-x@beta)), 1.)
    return z@beta, residual_sd

def normal_cdf(value):
    return 0.5*(1. + math.erf(float(value)/math.sqrt(2.)))


def make_features(schedule, team_game):
    scores, efficiency, last_date = {}, {}, {}
    rows = []
    if "gameday" not in schedule.columns:
        raise ValueError("Schedule is missing gameday; rest-day features unavailable.")
    schedule = schedule.copy()
    schedule["gameday"] = pd.to_datetime(schedule.gameday, errors="coerce")
    for (season, week), games in schedule.groupby(["season", "week"], sort=True):
        updates = []
        for g in games.itertuples(index=False):
            home, away, gid = str(g.home_team), str(g.away_team), str(g.game_id)
            def rest(team):
                prev = last_date.get(team)
                if prev is None or pd.isna(g.gameday) or pd.isna(prev):
                    return 7.0, 0.0
                days = (g.gameday-prev).days
                # Reset rest after offseason: long gaps do not imply extra game readiness.
                return (float(np.clip(days, 3, 14)), 1.0) if 0 < days <= 21 else (7.0, 0.0)
            hs, az = rolling(scores, home, 2), rolling(scores, away, 2)
            he, ae = rolling(efficiency, home, 4), rolling(efficiency, away, 4)
            hr, hr_known = rest(home)
            ar, ar_known = rest(away)
            baseline = [hs[0]-az[0], hs[1]-az[1], hs[2], az[2]]
            advanced = baseline + [he[0]-ae[0], he[1]-ae[1],
                                   he[2]-ae[2], he[3]-ae[3], he[4], ae[4]]
            # The intercept in each regression captures average home-field advantage.
            # Rest is known before kickoff; no future game outcomes are used.
            with_rest = advanced + [hr-ar, hr_known, ar_known]
            rows.append(dict(season=int(season), week=int(week), game_id=gid,
                             home=home, away=away,
                             margin=float(g.home_score-g.away_score),
                             total=float(g.home_score+g.away_score),
                             Baseline=baseline, Advanced=advanced,
                             **{"Advanced + Rest":with_rest}))
            updates += [("score", home, [float(g.home_score),float(g.away_score)]),
                        ("score", away, [float(g.away_score),float(g.home_score)])]
            ht, at = team_game.get((gid, home)), team_game.get((gid, away))
            if ht and at and ht[2] >= 10 and at[2] >= 10:
                updates += [("eff", home, [ht[0],at[0],ht[1],at[1]]),
                            ("eff", away, [at[0],ht[0],at[1],ht[1]])]
            if pd.notna(g.gameday):
                updates += [("date", home, g.gameday), ("date", away, g.gameday)]
        for kind, team, value in updates:
            if kind == "score":
                scores.setdefault(team, []).append(value)
            elif kind == "eff":
                efficiency.setdefault(team, []).append(value)
            else:
                last_date[team] = value
    return pd.DataFrame(rows)


def evaluate(frame):
    outputs = []
    for year in EVAL_YEARS:
        train, test = frame[frame.season < year], frame[frame.season == year]
        if len(train) < 150 or test.empty:
            continue
        by_model = {}
        for model in ["Baseline", "Advanced", "Advanced + Rest"]:
            margin, sigma = ridge(train, test, model, "margin")
            total, _ = ridge(train, test, model, "total")
            by_model[model] = (margin, total, sigma)
        # 50/50 weight is PRE-SPECIFIED, not optimized using the holdout seasons.
        b, a = by_model["Baseline"], by_model["Advanced"]
        blended_margin = 0.5*b[0]+0.5*a[0]
        blended_total = 0.5*b[1]+0.5*a[1]
        # Conservative approximation for blend residual scale; not calibrated.
        blend_sigma = 0.5*b[2]+0.5*a[2]
        by_model["50/50 Blend"] = (blended_margin, blended_total, blend_sigma)
        for model, (margin, total, sigma) in by_model.items():
            result = test[["season", "week", "game_id", "home", "away", "margin", "total"]].copy()
            result["model"] = model
            result["pred_margin"] = margin
            result["pred_total"] = total
            result["home_win_probability"] = [normal_cdf(v/sigma) for v in margin]
            outputs.append(result)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def metrics(sub):
    nt = sub[sub.margin.ne(0)]
    return {"Games":len(sub),
            "Spread MAE":round(float((sub.pred_margin-sub.margin).abs().mean()),2),
            "Total MAE":round(float((sub.pred_total-sub.total).abs().mean()),2),
            "Winner accuracy %":round(100*float(((nt.pred_margin>0)==(nt.margin>0)).mean()),1) if len(nt) else np.nan,
            "Brier score":round(float(((nt.home_win_probability-(nt.margin>0).astype(float))**2).mean()),4) if len(nt) else np.nan}

try:
    with st.spinner("Loading NFL play-by-play data; the first run can take several minutes..."):
        schedule, team_games, _ = fetch_data()
        frame = make_features(schedule, team_games)
        predictions = evaluate(frame)
    if predictions.empty:
        st.error("Not enough historical games to run the backtest.")
        st.stop()
    period = st.selectbox("Evaluation period", ["Both seasons", "2024", "2025"])
    view = predictions if period == "Both seasons" else predictions[predictions.season.eq(int(period))]
    st.subheader("Model leaderboard")
    names = ["Baseline", "Advanced", "Advanced + Rest", "50/50 Blend"]
    leaderboard = pd.DataFrame([{"Model":name, **metrics(view[view.model.eq(name)])} for name in names])
    st.dataframe(leaderboard, hide_index=True, use_container_width=True)
    st.caption("Home-field advantage is already captured by the regression intercept. Rest is tested as an additional feature. Lower MAE/Brier is better.")
    st.subheader("Performance by season")
    yearly = pd.DataFrame([{"Season":int(year),"Model":name, **metrics(sub[sub.model.eq(name)])}
        for year, sub in predictions.groupby("season") for name in names])
    st.dataframe(yearly, hide_index=True, use_container_width=True)
    st.subheader("Weekly performance")
    model = st.selectbox("Model to inspect", names, index=1)
    selected = view[view.model.eq(model)]
    weekly = pd.DataFrame([{"Season":int(year), "Week":int(week), **metrics(group)}
        for (year,week), group in selected.groupby(["season","week"])])
    st.dataframe(weekly, hide_index=True, use_container_width=True)
    st.download_button("Download game-level backtest CSV", predictions.to_csv(index=False),
        file_name="sds_model_44_backtest.csv", mime="text/csv")
    with st.expander("Methodology and limitations"):
        st.markdown("""
        - All models use the same completed 2024–2025 regular-season games.
        - 2024 evaluation trains on 2022–2023; 2025 evaluation trains on 2022–2024.
        - All team performance features use prior weeks only; entire weeks are frozen before updates.
        - Baseline: recent scoring; Advanced: scoring + offensive/defensive EPA and success rate.
        - Advanced + Rest: adds pregame days since previous game, capped at 14 days.
        - Home field is represented by the **regression intercept**, not a new learned team-specific home adjustment.
        - 50/50 Blend: equal-weight Baseline and Advanced predictions; weight chosen in advance.
        - Brier probabilities use uncalibrated normal approximations, including the blended residual scale.
        - No new weight tuning, opponent-strength correction, verified QB starter changes, or live odds.
        - These are season-level walk-forward tests, not weekly retraining or proof of betting profitability.
        """)
except Exception as exc:
    st.error("Model Optimization Lab could not load. Your existing pages are unaffected.")
    st.code(f"{type(exc).__name__}: {exc}")
