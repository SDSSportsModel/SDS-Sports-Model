"""SDS 4.8: calibrated Advanced EPA + FanDuel pregame moneyline comparison.

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

st.set_page_config(page_title="SDS | Calibrated Edge Lab", page_icon="🏈", layout="wide")
st.title("🏈 SDS 4.8 | Calibrated Edge Finder")
st.warning("Experimental research only. Probabilities remain experimental. Positive hypothetical EV is NOT a verified betting edge.")
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


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-35,35)))

def logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return np.log(p/(1-p))

def fit_platt(p,y):
    # Logistic calibration with a small L2 penalty, fitted on 2024 only.
    z=logit(p); x=np.column_stack([np.ones(len(z)),z]); y=np.asarray(y,dtype=float)
    b=np.array([0.0,1.0]); penalty=np.diag([0.01,0.1])
    for _ in range(100):
        q=sigmoid(x@b)
        gradient=x.T@(q-y)+penalty@(b-np.array([0.0,1.0]))
        w=np.maximum(q*(1-q),1e-7)
        hessian=x.T@(x*w[:,None])+penalty
        step=np.linalg.solve(hessian,gradient)
        b-=step
        if np.max(np.abs(step))<1e-8: break
    return b



@st.cache_data(ttl=21600, show_spinner=False)
def fit_2024_calibrator(data):
    # The exact 4.7 procedure: train EPA through 2023, predict 2024,
    # then fit a regularized Platt calibrator on 2024 home outcomes only.
    training = data[data.completed & data.season.between(2022, 2023)]
    calibration = data[data.completed & data.season.eq(2024)]
    if len(training) < 200 or len(calibration) < 200:
        raise ValueError('Insufficient 2022–2024 games for 2024 calibration')
    beta, sd = ridge_fit(training, 'margin')
    p = np.clip(np.asarray([normal_cdf(m / sd) for m in forecast(calibration, beta)]), 1e-6, 1-1e-6)
    y = (calibration.margin.to_numpy(dtype=float) > 0).astype(float)
    return fit_platt(p, y)


def score_ev(prob, odds):
    return 100 * (prob * profit(odds) - (1 - prob))

try:
    api_key = st.secrets.get('ODDS_API_KEY', '')
except Exception:
    api_key = ''
if not api_key:
    st.error('Configure ODDS_API_KEY in Streamlit Secrets as in SDS 4.5 and 4.6.')
    st.stop()

try:
    with st.spinner('Loading NFL EPA, 2024 calibration, and FanDuel odds...'):
        schedules, pbp, status, current_games = load_model_data()
        data = assemble(schedules, pbp)
        cal = fit_2024_calibrator(data)
        train = data[data.completed & data.season.between(2022, 2025)]
        if len(train) < 300:
            raise ValueError('Not enough completed games for the live model')
        margin_beta, margin_sd = ridge_fit(train, 'margin')
        total_beta, _ = ridge_fit(train, 'total')
        future = data[data.season.eq(2026) & ~data.completed].copy()
        events, remaining = fetch_odds(api_key)
        odds = parse_odds(events)

    st.caption(f'2026 EPA: {status}; completed 2026 games with PBP: {current_games}. Odds API credits remaining: {remaining}.')
    st.caption('Calibration: regularized Platt scaling fitted on 2024 predictions from an EPA model trained through 2023. Current EPA coefficients trained through 2025. This carries the 2024 adjustment forward; it is not newly validated on 2026 games.')
    if 'unavailable' in status:
        st.warning('Current-season EPA unavailable; predictions may be stale.')
    if future.empty or odds.empty:
        st.info('No upcoming regular-season games or FanDuel moneylines returned.')
        st.stop()
    future['margin_model'] = forecast(future, margin_beta)
    future['total_model'] = forecast(future, total_beta)
    future['raw_home'] = np.asarray([normal_cdf(v / margin_sd) for v in future.margin_model])
    future['cal_home'] = sigmoid(cal[0] + cal[1] * logit(future.raw_home))
    joined = odds.merge(future, on=['home','away'], how='left', indicator=True, validate='many_to_one')
    joined['kickoff_delta_hours'] = (joined.kickoff_odds - joined.kickoff).abs().dt.total_seconds() / 3600
    matched = joined[(joined._merge == 'both') & joined.kickoff_delta_hours.notna() & (joined.kickoff_delta_hours <= 6)].copy()
    unmatched = joined.loc[~joined.index.isin(matched.index)].copy()
    now = pd.Timestamp.now(tz='UTC')
    st.subheader('Matching & data quality')
    a,b,c = st.columns(3)
    a.metric('Matched games', len(matched))
    b.metric('Unmatched events', len(unmatched))
    if matched.empty:
        c.metric('Stale / undated', 0)
        st.warning('No valid kickoff-verified matches. Check team aliases, schedule, and odds provider.')
        st.stop()
    matched['odds_age_min'] = (now - matched.updated).dt.total_seconds() / 60
    matched['stale'] = matched.odds_age_min.isna() | (matched.odds_age_min > 60) | (matched.odds_age_min < -5)
    c.metric('Stale / undated', int(matched.stale.sum()))
    matched['matchup'] = matched.away_name + ' @ ' + matched.home_name
    matched['kickoff_utc'] = matched.kickoff_odds.dt.strftime('%Y-%m-%d %H:%M UTC')
    rows = []
    for r in matched.itertuples(index=False):
        for side in ('home','away'):
            home_side = side == 'home'
            raw = float(r.raw_home if home_side else 1-r.raw_home)
            calibrated = float(r.cal_home if home_side else 1-r.cal_home)
            price = int(r.home_odds if home_side else r.away_odds)
            novig = float(r.home_novig if home_side else r.away_novig)
            rows.append({'Week':int(r.week),'Kickoff (UTC)':r.kickoff_utc,
                'Matchup':r.matchup,'Team':r.home_name if home_side else r.away_name,
                'FanDuel ML':price,'Original SDS %':100*raw,'Calibrated SDS %':100*calibrated,
                'Calibration change (pp)':100*(calibrated-raw),'No-vig market %':100*novig,
                'Calibrated minus market (pp)':100*(calibrated-novig),
                'Original hypothetical EV %':score_ev(raw,price),
                'Calibrated hypothetical EV %':score_ev(calibrated,price),
                'EV change (pp)':score_ev(calibrated,price)-score_ev(raw,price),
                'Projected home margin':float(r.margin_model),'Projected total':float(r.total_model),
                'Odds age (min)':float(r.odds_age_min) if pd.notna(r.odds_age_min) else np.nan,
                'Odds status':'STALE/UNKNOWN' if r.stale else 'Recent',
                'Market disagreement':'REVIEW >10pp' if abs(100*(calibrated-novig)) > 10 else 'Within 10pp',
                'Provider event ID':r.provider_event_id})
    table = pd.DataFrame(rows)
    week = st.selectbox('NFL week', sorted(table['Week'].unique().tolist()))
    selected = table[table['Week'].eq(week)].copy()
    selected = selected.sort_values('Calibrated hypothetical EV %',ascending=False)
    st.subheader('Calibrated Edge Finder — experimental')
    st.warning('Large hypothetical EV can reflect model misspecification. Do not interpret these rows as picks. Starting QBs, injuries and roster news are not modeled.')
    if selected['Odds status'].ne('Recent').any():
        st.error('Some odds are stale or have unknown timestamps. Recheck the sportsbook before any interpretation.')
    st.dataframe(selected.drop(columns=['Provider event ID']).round(2),hide_index=True,use_container_width=True)
    st.caption('Both teams appear for each game. The no-vig percentage is proportional normalization of both moneyline implied probabilities. Hypothetical EV assumes the SDS probability is correct; neither model is proven profitable.')
    st.subheader('Calibration impact')
    st.metric('Largest original hypothetical EV (selected week)', f"{selected['Original hypothetical EV %'].max():+.1f}%")
    st.metric('Largest calibrated hypothetical EV (selected week)', f"{selected['Calibrated hypothetical EV %'].max():+.1f}%")
    st.caption('These maxima may refer to different teams. Compare the per-team rows to see how calibration changed individual values.')
    with st.expander('Unmatched event diagnostics'):
        if unmatched.empty:
            st.success('Every returned event passed team and kickoff matching.')
        else:
            st.dataframe(unmatched[['home_name','away_name','kickoff_odds','kickoff_delta_hours']],hide_index=True)
    st.subheader('Pregame snapshot export')
    exported = selected.copy()
    exported.insert(0,'Snapshot generated UTC',datetime.now(timezone.utc).isoformat())
    exported['Record type']='manual_pregame_export_not_automatically_archived'
    st.download_button('Download timestamped week snapshot CSV', exported.to_csv(index=False),
        file_name=f'sds_4_8_week_{week}_snapshot.csv',mime='text/csv')
    st.info('Download the CSV BEFORE kickoff and preserve the original file. This is a manual export, not a durable automatic database or a verified historical pregame record. The snapshot timestamp is export time, while odds timestamps come from the provider.')
    st.caption('Streamlit caching: odds up to 15 minutes; NFL data up to 6 hours. Refreshing the browser does not necessarily fetch new data.')
except requests.RequestException:
    st.error('Network error fetching FanDuel odds. Other SDS pages are unaffected.')
except Exception as exc:
    st.error('SDS 4.8 could not load. Other SDS pages remain available.')
    st.caption(f'{type(exc).__name__}: {exc}')
