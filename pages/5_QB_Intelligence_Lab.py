"""SDS QB Intelligence 4.3: chronological research comparison, not betting advice."""
import math
import numpy as np
import pandas as pd
import streamlit as st
import nflreadpy as nfl

st.set_page_config(page_title="SDS QB Intelligence Lab", page_icon="🏈", layout="wide")
st.title("🏈 SDS QB Intelligence Lab — 4.3")
st.warning("Experimental historical backtest. No confirmed upcoming starters, live odds, or betting recommendations.")
st.caption("Same completed games and training windows for all models. Current-week information is excluded from features.")
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
    qb_game = {}
    for season in SEASONS:
        pbp = nfl.load_pbp([season]).to_pandas()
        required = {"game_id", "posteam", "defteam", "play_type", "epa", "success", "passer_player_id"}
        if not required.issubset(pbp.columns):
            raise ValueError(f"Missing play-by-play fields in {season}: {sorted(required-set(pbp.columns))}")
        plays = pbp.loc[
            pbp.play_type.isin(["run", "pass"]) & pbp.posteam.notna() &
            pbp.defteam.notna() & pbp.epa.notna(),
            ["game_id", "posteam", "epa", "success", "play_type", "passer_player_id"]
        ].copy()
        team = plays.groupby(["game_id", "posteam"], as_index=False).agg(
            epa=("epa", "mean"), success=("success", "mean"), n=("epa", "size"))
        for row in team.itertuples(index=False):
            team_game[(str(row.game_id), str(row.posteam))] = (
                float(row.epa), float(row.success) if pd.notna(row.success) else 0., int(row.n))
        passes = plays[plays.play_type.eq("pass") & plays.passer_player_id.notna()]
        qbs = passes.groupby(["game_id", "posteam", "passer_player_id"], as_index=False).agg(
            qb_epa=("epa", "mean"), qb_success=("success", "mean"), attempts=("epa", "size"))
        if not qbs.empty:
            # Primary passer by number of recorded pass plays, NOT a verified starter.
            qbs = qbs.sort_values(["game_id", "posteam", "attempts"], ascending=[True, True, False])
            qbs = qbs.drop_duplicates(["game_id", "posteam"])
            for row in qbs.itertuples(index=False):
                qb_game[(str(row.game_id), str(row.posteam))] = (
                    str(row.passer_player_id), float(row.qb_epa),
                    float(row.qb_success) if pd.notna(row.qb_success) else 0., int(row.attempts))
    return schedule, team_game, qb_game


def rolling(history, team, ncols):
    records = history.get(team, [])[-5:]
    if not records:
        return [0.] * ncols + [0.]
    means = np.asarray(records, dtype=float).mean(axis=0)
    return [float(v) for v in means] + [len(records)/5.]


def make_features(schedule, team_game, qb_game):
    scores, efficiency, quarterback = {}, {}, {}
    rows = []
    # Freeze histories for an entire week; update only AFTER the week is processed.
    for (season, week), games in schedule.groupby(["season", "week"], sort=True):
        pending = []
        for g in games.itertuples(index=False):
            home, away, gid = str(g.home_team), str(g.away_team), str(g.game_id)
            hs, az = rolling(scores, home, 2), rolling(scores, away, 2)
            he, ae = rolling(efficiency, home, 4), rolling(efficiency, away, 4)
            hq, aq = rolling(quarterback, home, 2), rolling(quarterback, away, 2)
            baseline = [hs[0]-az[0], hs[1]-az[1], hs[2], az[2]]
            advanced = baseline + [he[0]-ae[0], he[1]-ae[1], he[2]-ae[2], he[3]-ae[3], he[4], ae[4]]
            with_qb = advanced + [hq[0]-aq[0], hq[1]-aq[1], hq[2], aq[2]]
            rows.append(dict(season=int(season), week=int(week), game_id=gid,
                             home=home, away=away, margin=float(g.home_score-g.away_score),
                             total=float(g.home_score+g.away_score),
                             Baseline=baseline, Advanced=advanced, **{"Advanced + QB":with_qb}))
            pending += [("score", home, [float(g.home_score), float(g.away_score)]),
                        ("score", away, [float(g.away_score), float(g.home_score)])]
            ht, at = team_game.get((gid, home)), team_game.get((gid, away))
            if ht and at and ht[2] >= 10 and at[2] >= 10:
                pending += [("eff", home, [ht[0], at[0], ht[1], at[1]]),
                            ("eff", away, [at[0], ht[0], at[1], ht[1]])]
            hqb, aqb = qb_game.get((gid, home)), qb_game.get((gid, away))
            if hqb and hqb[3] >= 10:
                pending.append(("qb", home, [hqb[1], hqb[2]]))
            if aqb and aqb[3] >= 10:
                pending.append(("qb", away, [aqb[1], aqb[2]]))
        for kind, team, values in pending:
            {"score":scores, "eff":efficiency, "qb":quarterback}[kind].setdefault(team, []).append(values)
    return pd.DataFrame(rows)


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

try:
    with st.spinner("Loading 2022–2025 NFL play-by-play. First load can take several minutes..."):
        schedule, team_games, qb_games = fetch_data()
        frame = make_features(schedule, team_games, qb_games)
    predictions = []
    for year in EVAL_YEARS:
        train, test = frame[frame.season < year], frame[frame.season == year]
        if len(train) < 150 or test.empty:
            continue
        for model in ["Baseline", "Advanced", "Advanced + QB"]:
            margin, sigma = ridge(train, test, model, "margin")
            total, _ = ridge(train, test, model, "total")
            result = test[["season", "week", "game_id", "home", "away", "margin", "total"]].copy()
            result["model"] = model
            result["projected_margin"] = margin
            result["projected_total"] = total
            result["home_win_probability"] = [normal_cdf(v/sigma) for v in margin]
            predictions.append(result)
    if not predictions:
        st.error("Insufficient data to run the evaluation.")
        st.stop()
    pred = pd.concat(predictions, ignore_index=True)
    choice = st.selectbox("Evaluation period", ["Both seasons", "2024", "2025"])
    view = pred if choice == "Both seasons" else pred[pred.season.eq(int(choice))]
    scores = []
    for model in ["Baseline", "Advanced", "Advanced + QB"]:
        sub = view[view.model.eq(model)]
        no_ties = sub[sub.margin.ne(0)]
        scores.append({"Model":model, "Games":len(sub),
                       "Spread MAE":round(float((sub.projected_margin-sub.margin).abs().mean()), 2),
                       "Total MAE":round(float((sub.projected_total-sub.total).abs().mean()), 2),
                       "Winner accuracy %":round(float(100*((no_ties.projected_margin > 0) == (no_ties.margin > 0)).mean()), 1),
                       "Brier score":round(float(((no_ties.home_win_probability-(no_ties.margin > 0).astype(float))**2).mean()), 4)})
    st.subheader("Quarterback feature test")
    st.dataframe(pd.DataFrame(scores), hide_index=True, use_container_width=True)
    st.caption("Lower MAE and Brier are better; higher winner accuracy is better. Results can differ from older SDS pages because the feature definitions are standardized here.")
    st.subheader("Game-level predictions")
    st.dataframe(view[["season", "week", "home", "away", "model", "projected_margin", "projected_total", "home_win_probability"]].round(3), hide_index=True, use_container_width=True)
    st.download_button("Download validation CSV", pred.to_csv(index=False), file_name="sds_qb_43_validation.csv", mime="text/csv")
    with st.expander("How QB features work — read before interpreting"):
        st.markdown("""
        - QB statistics are from the **primary passer by pass-play count** in each completed game, not a confirmed starter.
        - Quarterback features summarize each team's last five games with an eligible primary passer; they do **not** identify or adjust for an upcoming starter change.
        - All feature histories are frozen before each week's games, preventing current-week game leakage.
        - **2024 evaluation** trains on 2022–2023; **2025 evaluation** trains on 2022–2024.
        - All three models use identical game samples, train-only scaling and the same ridge penalty.
        - Win probabilities use an uncalibrated normal approximation. The Brier score is for historical evaluation only.
        - This is a preliminary research comparison. To model confirmed starting QBs, we need an independently sourced, timestamped pregame starter feed.
        - More seasons, opponent adjustments, and walk-forward probability calibration are needed before any betting-value claims.
        """)
except Exception as error:
    st.error("QB Intelligence Lab couldn't load. Your existing SDS pages are unaffected.")
    st.code(f"{type(error).__name__}: {error}")
