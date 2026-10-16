"""SDS Sports Model 3.0: fair, chronological baseline comparison."""
import math
import numpy as np
import pandas as pd
import streamlit as st
import nflreadpy as nfl

st.set_page_config(page_title="SDS Model Validation", page_icon="🏈", layout="wide")
st.title("🏈 SDS Model Validation Lab")
st.caption("Head-to-head comparison on identical NFL games, with walk-forward testing.")
st.warning("Research only. These scores do not measure betting profitability. No sportsbook odds are used.")

YEARS = list(range(2022, 2026))
TARGET_YEARS = [2024, 2025]

@st.cache_data(ttl=86400, show_spinner=False)
def load_games():
    schedules = nfl.load_schedules(YEARS).to_pandas()
    required = {"game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(schedules.columns):
        raise ValueError(f"Missing schedule columns: {sorted(required-set(schedules.columns))}")
    if "game_type" in schedules.columns:
        schedules = schedules[schedules.game_type.eq("REG")]
    schedules = schedules.dropna(subset=["home_score", "away_score"]).copy()
    schedules = schedules.sort_values(["season", "week", "game_id"])
    pbp_by_game = {}
    for season in YEARS:
        pbp = nfl.load_pbp([season]).to_pandas()
        needed = {"game_id", "posteam", "defteam", "epa", "success", "play_type"}
        if not needed.issubset(pbp.columns):
            raise ValueError(f"Missing PBP columns: {sorted(needed-set(pbp.columns))}")
        plays = pbp.loc[pbp.play_type.isin(["run", "pass"]) & pbp.posteam.notna()
                        & pbp.defteam.notna() & pbp.epa.notna(),
                        ["game_id", "posteam", "epa", "success"]]
        aggs = plays.groupby(["game_id", "posteam"], as_index=False).agg(
            epa=("epa", "mean"), success=("success", "mean"), plays=("epa", "size"))
        for r in aggs.itertuples(index=False):
            pbp_by_game[(str(r.game_id), str(r.posteam))] = (
                float(r.epa), float(r.success) if pd.notna(r.success) else 0.0, int(r.plays))
    return schedules, pbp_by_game

def mean_recent(values, size):
    return float(np.mean(values[-5:])) if values else 0.0

def feature_table(schedules, pbp):
    # Same-week games are evaluated against a frozen history snapshot.
    scores, efficiencies = {}, {}
    output = []
    for (season, week), games in schedules.groupby(["season", "week"], sort=True):
        updates = []
        for g in games.itertuples(index=False):
            home, away, gid = str(g.home_team), str(g.away_team), str(g.game_id)
            def s(team):
                vals = scores.get(team, [])[-5:]
                return [mean_recent([v[0] for v in vals], 5),
                        mean_recent([v[1] for v in vals], 5),
                        len(vals)/5.0]
            def e(team):
                vals = efficiencies.get(team, [])[-5:]
                if not vals:
                    return [0.0, 0.0, 0.0, 0.0, 0.0]
                arr = np.asarray(vals, dtype=float)
                return list(arr.mean(axis=0)) + [len(vals)/5.0]
            hs, az = s(home), s(away)
            he, ae = e(home), e(away)
            baseline = [hs[0]-az[0], hs[1]-az[1], hs[2], az[2]]
            advanced = baseline + [he[0]-ae[0], he[1]-ae[1],
                                   he[2]-ae[2], he[3]-ae[3], he[4], ae[4]]
            margin = float(g.home_score - g.away_score)
            output.append({"season":int(season), "week":int(week), "game_id":gid,
                           "home":home, "away":away, "margin":margin,
                           "total":float(g.home_score+g.away_score),
                           "baseline":baseline, "advanced":advanced})
            updates.append(("score", home, (float(g.home_score),float(g.away_score))))
            updates.append(("score", away, (float(g.away_score),float(g.home_score))))
            hpbp, apbp = pbp.get((gid, home)), pbp.get((gid, away))
            if hpbp and apbp and hpbp[2] >= 10 and apbp[2] >= 10:
                updates.append(("epa", home, (hpbp[0], apbp[0], hpbp[1], apbp[1])))
                updates.append(("epa", away, (apbp[0], hpbp[0], apbp[1], hpbp[1])))
        for kind, team, values in updates:
            (scores if kind == "score" else efficiencies).setdefault(team, []).append(values)
    return pd.DataFrame(output)

def train_predict(train, test, feature, target, alpha=20.0):
    x = np.asarray(train[feature].tolist(), dtype=float)
    xt = np.asarray(test[feature].tolist(), dtype=float)
    # Fit scaling on training games only (no holdout leakage).
    avg = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (x-avg)/scale
    xt = (xt-avg)/scale
    x = np.column_stack([np.ones(len(x)), x])
    xt = np.column_stack([np.ones(len(xt)), xt])
    penalty = np.diag([0.0] + [alpha]*(x.shape[1]-1))
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ train[target].to_numpy(dtype=float))
    fitted = x @ beta
    sigma = max(float(np.std(train[target].to_numpy(dtype=float)-fitted)), 1.0)
    return xt @ beta, sigma

def win_prob(margin, sigma):
    return 0.5 * (1.0 + math.erf(float(margin)/(sigma*math.sqrt(2.0))))

try:
    with st.spinner("Downloading historical NFL schedules and play-by-play (first run may take several minutes)..."):
        schedules, pbp = load_games()
        frame = feature_table(schedules, pbp)
    all_predictions = []
    for season in TARGET_YEARS:
        train = frame[frame.season < season]
        test = frame[frame.season == season].copy()
        if len(train) < 150 or test.empty:
            continue
        for model in ["baseline", "advanced"]:
            margins, sigma = train_predict(train, test, model, "margin")
            totals, _ = train_predict(train, test, model, "total")
            tmp = test[["season", "week", "game_id", "home", "away", "margin", "total"]].copy()
            tmp["model"] = model.title()
            tmp["pred_margin"] = margins
            tmp["pred_total"] = totals
            tmp["home_win_probability"] = [win_prob(m, sigma) for m in margins]
            all_predictions.append(tmp)
    if not all_predictions:
        st.error("Insufficient historical data for walk-forward evaluation.")
        st.stop()
    pred = pd.concat(all_predictions, ignore_index=True)
    pred["margin_abs_error"] = (pred.pred_margin-pred.margin).abs()
    pred["total_abs_error"] = (pred.pred_total-pred.total).abs()
    # Ties excluded from binary classification and Brier score.
    pred["non_tie"] = pred.margin.ne(0)
    pred["actual_home_win"] = (pred.margin > 0).astype(float)
    pred["correct_winner"] = (pred.pred_margin > 0).eq(pred.margin > 0)
    pred["brier"] = (pred.home_win_probability-pred.actual_home_win)**2

    st.subheader("Fair model comparison")
    season_choice = st.selectbox("Evaluation season", ["Both seasons", "2024", "2025"], index=0)
    view = pred if season_choice == "Both seasons" else pred[pred.season.eq(int(season_choice))]
    summaries = []
    for model, grp in view.groupby("model"):
        nt = grp[grp.non_tie]
        summaries.append({"Model":model, "Games":len(grp),
                          "Spread MAE":round(grp.margin_abs_error.mean(),2),
                          "Total MAE":round(grp.total_abs_error.mean(),2),
                          "Winner accuracy %":round(100*nt.correct_winner.mean(),1) if len(nt) else np.nan,
                          "Brier score":round(nt.brier.mean(),4) if len(nt) else np.nan})
    st.dataframe(pd.DataFrame(summaries).sort_values("Model"),
                 hide_index=True, use_container_width=True)
    st.caption("Lower MAE and Brier score are better. Models train only on earlier seasons. "
               "Both use identical games and the same training seasons. "
               "Winner accuracy and Brier exclude tied games.")
    st.subheader("Probability calibration")
    selected_model = st.selectbox("Model", ["Baseline", "Advanced"])
    cal = view[(view.model.eq(selected_model)) & view.non_tie].copy()
    cal["probability_bin"] = pd.cut(cal.home_win_probability,
                                  bins=np.linspace(0,1,11), include_lowest=True)
    grouped = cal.groupby("probability_bin", observed=True).agg(
        predicted=("home_win_probability", "mean"),
        actual=("actual_home_win", "mean"),
        games=("actual_home_win", "size")).reset_index()
    if not grouped.empty:
        st.line_chart(grouped.set_index("predicted")["actual"])
        st.dataframe(grouped.rename(columns={"predicted":"Mean predicted win %",
                                             "actual":"Actual home win %",
                                             "games":"Games per bin"}).assign(
            **{"Mean predicted win %":lambda x:100*x["Mean predicted win %"],
               "Actual home win %":lambda x:100*x["Actual home win %"]}).round(1),
            hide_index=True, use_container_width=True)
    st.caption("Calibration bins with few games are noisy. Normal-residual probabilities are not separately calibrated.")
    st.download_button("Download game-level evaluation CSV", pred.to_csv(index=False),
                       file_name="sds_walk_forward_validation.csv", mime="text/csv")
    with st.expander("Important methodology notes"):
        st.markdown("""
        - **2024 test:** train on 2022–2023. **2025 test:** train on 2022–2024.
        - Each game's features use earlier weeks only. The current week's games update history after the entire week is scored.
        - Baseline uses last-five-game scoring. Advanced adds last-five-game offensive and defensive EPA and success rate.
        - Same game samples, same training periods, same ridge penalty and same probability approximation.
        - This is a **season-by-season** walk-forward test, not a full per-week retraining backtest.
        - Advanced EPA history may be missing for some games; the model uses neutral defaults.
        - No hyperparameters are optimized on 2024/2025. No sportsbook prices, vig, or bet returns.
        - A real wagering evaluation requires historical odds snapshots available **before kickoff**, plus closing-line and transaction-cost analysis.
        """)
except Exception as error:
    st.error("Validation Lab could not load. Your existing SDS pages are unaffected.")
    st.code(f"{type(error).__name__}: {error}")
