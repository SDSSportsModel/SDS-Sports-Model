"""SDS Sports Model: experimental game projections from historical NFL results."""
import math
import numpy as np
import pandas as pd
import streamlit as st
import nflreadpy as nfl

st.set_page_config(page_title="SDS | Game Projections", page_icon="🏈", layout="wide")
st.title("🏈 SDS Game Projections")
st.warning("EXPERIMENTAL: Historical-score baseline, not a validated betting model. No live odds, player props, or guaranteed edge.")
st.caption("Uses completed regular-season scores only. Five-game rolling form is computed from games played BEFORE each kickoff. Model is retrained from prior seasons; not live in-game data.")

@st.cache_data(ttl=3600, show_spinner=False)
def get_schedule():
    return nfl.load_schedules(list(range(2019, 2027))).to_pandas()

def build_features(schedule):
    df = schedule.copy()
    required = ["season", "week", "gameday", "home_team", "away_team", "home_score", "away_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing NFL schedule fields: {', '.join(missing)}")
    if "game_type" in df.columns:
        df = df[df["game_type"].eq("REG")]
    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    df = df.dropna(subset=["gameday", "home_team", "away_team"]).sort_values(["gameday","week"]).copy()
    history = {}
    rows = []
    for _, g in df.iterrows():
        home, away = str(g.home_team), str(g.away_team)
        def prior(team):
            games = history.get(team, [])[-5:]
            if not games:
                return [0., 0., 0., 0.]
            return [float(np.mean([v[0] for v in games])),
                    float(np.mean([v[1] for v in games])),
                    float(np.mean([v[0]-v[1] for v in games])),
                    float(len(games))/5.]
        h, a = prior(home), prior(away)
        x = [h[0]-a[0], h[1]-a[1], h[2]-a[2], h[3], a[3]]
        complete = pd.notna(g.home_score) and pd.notna(g.away_score)
        rows.append({"season":int(g.season),"week":int(g.week),"date":g.gameday,
                     "home":home,"away":away,"completed":complete,
                     "features":x,
                     "margin":float(g.home_score-g.away_score) if complete else np.nan,
                     "total":float(g.home_score+g.away_score) if complete else np.nan})
        if complete:
            history.setdefault(home, []).append((float(g.home_score),float(g.away_score)))
            history.setdefault(away, []).append((float(g.away_score),float(g.home_score)))
    return pd.DataFrame(rows)

def fit_ridge(frame, target, alpha=30.):
    X = np.asarray(frame["features"].tolist(),dtype=float)
    X = np.column_stack([np.ones(len(X)), X])
    y = frame[target].to_numpy(dtype=float)
    penalty = np.diag([0.] + [alpha]*(X.shape[1]-1))
    beta = np.linalg.solve(X.T @ X + penalty, X.T @ y)
    residual = y-X@beta
    sigma = max(float(np.std(residual,ddof=X.shape[1])), 1.0)
    return beta, sigma

def predict(frame, beta):
    X = np.asarray(frame["features"].tolist(),dtype=float)
    return np.column_stack([np.ones(len(X)), X]) @ beta

def normal_cdf(z):
    return 0.5*(1.0+math.erf(float(z)/math.sqrt(2)))

try:
    with st.spinner("Retrieving historical NFL schedules..."):
        all_games = build_features(get_schedule())
    completed = all_games[all_games.completed]
    # 2025 is a held-out season; report only results from that evaluation.
    train = completed[completed.season.between(2019,2024)]
    test = completed[completed.season.eq(2025)]
    if len(train) < 500:
        st.error("Not enough historical games to fit the baseline.")
        st.stop()
    if len(test):
        m0,_ = fit_ridge(train,"margin")
        t0,_ = fit_ridge(train,"total")
        st.subheader("Historical evaluation — 2025 holdout")
        c1,c2 = st.columns(2)
        c1.metric("Margin MAE",f"{np.mean(np.abs(predict(test,m0)-test.margin.to_numpy())):.1f} pts")
        c2.metric("Total MAE",f"{np.mean(np.abs(predict(test,t0)-test.total.to_numpy())):.1f} pts")
        st.caption("Trained on 2019–2024; evaluated on 2025. These are prediction errors, not betting returns.")
    fit = completed[completed.season.le(2025)]
    margin_beta, margin_sd = fit_ridge(fit,"margin")
    total_beta, total_sd = fit_ridge(fit,"total")
    future = all_games[(all_games.season.eq(2026)) & (~all_games.completed)].copy()
    if future.empty:
        st.info("No unplayed 2026 regular-season games were returned.")
    else:
        future["Projected home margin"] = predict(future,margin_beta)
        future["Projected total"] = predict(future,total_beta)
        future["Home win % (approx.)"] = [
            round(100*normal_cdf(v/margin_sd),1) for v in future["Projected home margin"]
        ]
        future["Away win % (approx.)"] = 100-future["Home win % (approx.)"]
        show = future.rename(columns={"date":"Date","away":"Away","home":"Home","week":"Week"})
        cols = ["Week","Date","Away","Home","Projected home margin","Projected total","Home win % (approx.)","Away win % (approx.)"]
        st.subheader("Upcoming game projections")
        st.dataframe(show[cols].round(1),hide_index=True,use_container_width=True)
        st.download_button("Download projections CSV",show[cols].to_csv(index=False),
                           file_name="sds_game_projections.csv",mime="text/csv")
    st.info("Model limitations: Scores-only baseline; no injury, quarterback, EPA, weather, or market adjustments. Normal approximation treats ties as negligible. Do not use for betting decisions without calibration, temporal backtests and line-history evaluation.")
except Exception as exc:
    st.error("Projection data could not be loaded. The main dashboard remains available.")
    st.caption(f"{type(exc).__name__}: {exc}")