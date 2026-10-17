"""SDS 5.0: admin-triggered direct pregame capture, result grading and honest evaluation."""
from datetime import datetime, timezone
import hmac
import numpy as np
import pandas as pd
import requests
import streamlit as st
import nflreadpy as nfl
from sds50_engine import (load_model_data, assemble, fit_2024_calibrator,
    ridge_fit, forecast, normal_cdf, sigmoid, logit, fetch_odds, parse_odds)

st.set_page_config(page_title='SDS 5.0 | Performance Hub', page_icon='📊', layout='wide')
st.title('📊 SDS 5.0 | Performance Hub')
st.caption('Direct, owner-triggered snapshots • Persistent Supabase history • Result grading')
st.warning('Research tool. Model probabilities are experimental. A hypothetical advantage is not a validated betting edge.')

def secret(k):
    try: return str(st.secrets.get(k, ''))
    except Exception: return ''

URL = secret('SUPABASE_URL').rstrip('/')
KEY = secret('SUPABASE_SERVICE_ROLE_KEY')
ADMIN = secret('SDS_TRACKER_ADMIN_PASSWORD')
ODDS_KEY = secret('ODDS_API_KEY')
TABLE = URL + '/rest/v1/sds_snapshots'
HEAD = {'apikey':KEY, 'Authorization':f'Bearer {KEY}', 'Content-Type':'application/json'}
if not all((URL,KEY,ADMIN,ODDS_KEY)):
    st.error('Configure SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, SDS_TRACKER_ADMIN_PASSWORD and ODDS_API_KEY in Streamlit Secrets.')
    st.stop()

with st.sidebar:
    st.subheader('Owner controls')
    password = st.text_input('Tracker admin password', type='password')
    is_admin = bool(password) and hmac.compare_digest(password, ADMIN)
    st.caption('Only the owner can fetch/save snapshots or grade results. Public visitors can view summary performance.')

def api(method, params=None, payload=None, extra=None):
    headers = dict(HEAD)
    if extra: headers.update(extra)
    response = requests.request(method, TABLE, params=params, json=payload, headers=headers, timeout=30)
    if not response.ok:
        raise RuntimeError(f'Supabase HTTP {response.status_code}: {response.text[:240]}')
    return response.json() if response.content else []

try:
    history = pd.DataFrame(api('GET', {'select':'*','order':'captured_at.desc','limit':'10000'}))
except Exception as exc:
    st.error(f'Unable to read Supabase: {exc}')
    st.stop()

st.subheader('Saved prediction history')
a,b,c = st.columns(3)
a.metric('Recorded team selections', len(history))
b.metric('Completed selections', int(history.status.eq('graded').sum()) if 'status' in history else 0)
c.metric('Pending selections', int(history.status.eq('pending').sum()) if 'status' in history else 0)
st.caption('Two teams per matchup may be saved. A selection is not necessarily a recommended wager.')
if not history.empty:
    shown = [c for c in ('captured_at','game_id','team','opponent','fanduel_ml','raw_win_pct','calibrated_win_pct','status','won') if c in history]
    with st.expander('View recorded predictions'):
        st.dataframe(history[shown], hide_index=True, use_container_width=True)
        st.download_button('Download tracker backup',history.to_csv(index=False),'sds50_tracker_backup.csv','text/csv')

st.divider()
st.subheader('Capture directly from FanDuel + SDS')
st.caption('Owner-triggered capture; not a background scheduler. Fetches live provider data when you press the button. The Odds API may charge request credits.')

# Avoid persisting proposed rows across browser sessions or silently refreshing stale odds.
if 'sds50_preview' not in st.session_state:
    st.session_state.sds50_preview = None
if st.button('Fetch fresh predictions and odds', disabled=not is_admin):
    st.session_state.sds50_preview = None
    try:
        with st.spinner('Loading model data and current FanDuel odds...'):
            schedules, pbp, status, current_games = load_model_data()
            games = assemble(schedules, pbp)
            calibrator = fit_2024_calibrator(games)
            training = games[games.completed & games.season.between(2022, 2025)]
            if len(training) < 300: raise ValueError('Insufficient historical training games')
            margin_beta, margin_sd = ridge_fit(training,'margin')
            future = games[(games.season.eq(2026)) & (~games.completed)].copy()
            if future.empty: raise ValueError('No unplayed games returned')
            future['raw_home'] = [normal_cdf(m/margin_sd) for m in forecast(future,margin_beta)]
            future['cal_home'] = sigmoid(calibrator[0] + calibrator[1]*logit(future.raw_home))
            # Fetch fresh FanDuel odds directly from the provider.
            events, remaining = fetch_odds(ODDS_KEY)
            odds = parse_odds(events)
            if odds.empty: raise ValueError('No future FanDuel moneylines available')
            joined = odds.merge(future,on=['home','away'],how='inner',validate='many_to_one')
            now = pd.Timestamp.now(tz='UTC')
            proposed = []
            existing = set(zip(history.game_id.astype(str),history.team.astype(str))) if not history.empty else set()
            for r in joined.itertuples(index=False):
                if pd.isna(r.kickoff) or pd.isna(r.kickoff_odds) or pd.isna(r.updated): continue
                # Strict kickoff verification (not the 6-hour exploratory 4.8 threshold).
                if abs((r.kickoff_odds-r.kickoff).total_seconds()) > 600: continue
                if r.kickoff_odds <= now: continue
                age = (now-r.updated).total_seconds()/60
                if not (0 <= age <= 60): continue
                for side in ('home','away'):
                    home_side = side == 'home'
                    team = r.home if home_side else r.away
                    opponent = r.away if home_side else r.home
                    if (str(r.game_id),str(team)) in existing: continue
                    ml = int(r.home_odds if home_side else r.away_odds)
                    raw = float(r.raw_home if home_side else 1-r.raw_home)
                    cal = float(r.cal_home if home_side else 1-r.cal_home)
                    if not (-100 >= ml or ml >= 100): continue
                    if not (0 < raw < 1 and 0 < cal < 1): continue
                    proposed.append({'game_id':str(r.game_id),'kickoff_utc':r.kickoff_odds.isoformat(),
                        'team':str(team),'opponent':str(opponent),'fanduel_ml':ml,
                        'raw_win_pct':round(100*raw,6),'calibrated_win_pct':round(100*cal,6)})
            st.session_state.sds50_preview = {'rows':proposed,'created':now.isoformat()}
            st.success(f'Found {len(proposed)} new eligible team selections. Skipped stale, unmatched and previously recorded rows. Odds API credits remaining: {remaining}.')
            if 'unavailable' in status: st.warning('Current-season EPA unavailable; historical fallback in effect.')
    except Exception as exc:
        st.error(f'Unable to prepare direct snapshot: {type(exc).__name__}: {exc}')

preview = st.session_state.sds50_preview
if preview is not None:
    st.caption(f"Preview fetched at {preview['created']}. Refresh odds if more than 15 minutes have passed.")
    if preview['rows']:
        st.dataframe(pd.DataFrame(preview['rows']),hide_index=True,use_container_width=True)
        if st.button('Save eligible pregame records to Supabase',type='primary',disabled=not is_admin):
            try:
                now = datetime.now(timezone.utc)
                created = pd.to_datetime(preview['created'],utc=True).to_pydatetime()
                if (now-created).total_seconds() > 900:
                    raise ValueError('Preview is older than 15 minutes. Fetch fresh odds first.')
                # Verify current DB state and cutoff again at save time.
                current = pd.DataFrame(api('GET',{'select':'game_id,team','limit':'10000'}))
                existing = set(zip(current.game_id.astype(str),current.team.astype(str))) if not current.empty else set()
                to_save = []
                for row in preview['rows']:
                    if (row['game_id'],row['team']) in existing: continue
                    kickoff = pd.to_datetime(row['kickoff_utc'],utc=True).to_pydatetime()
                    if kickoff <= now: continue
                    to_save.append({**row,'captured_at':now.isoformat(),'status':'pending'})
                if not to_save: raise ValueError('No new eligible future records remain.')
                # Existing SQL UNIQUE(game_id,team) protects against accidental duplicates.
                api('POST',payload=to_save,extra={'Prefer':'return=representation'})
                st.session_state.sds50_preview = None
                st.success(f'Saved {len(to_save)} records at {now.isoformat()}. Refresh the page to verify persistence.')
            except Exception as exc:
                st.error(f'Save failed: {exc}')
    else:
        st.info('No new eligible games. Existing snapshots remain unchanged.')

st.divider()
st.subheader('Grade completed games')
st.caption('Owner-triggered grading using official NFL schedule results. No automatic background grading yet.')
if st.button('Grade pending results now',disabled=not is_admin):
    try:
        schedules = nfl.load_schedules([2026]).to_pandas()
        if 'game_type' in schedules: schedules = schedules[schedules.game_type.eq('REG')]
        final = {str(g.game_id):g for g in schedules[schedules.home_score.notna() & schedules.away_score.notna()].itertuples(index=False)}
        count = 0
        for r in history.to_dict('records'):
            if r.get('status') != 'pending': continue
            g = final.get(str(r['game_id']))
            if g is None: continue
            if pd.to_datetime(r['captured_at'],utc=True) >= pd.to_datetime(r['kickoff_utc'],utc=True): continue
            home, away = str(g.home_team), str(g.away_team)
            if r['team'] not in (home,away) or int(g.home_score)==int(g.away_score): continue
            won = (g.home_score>g.away_score) if r['team']==home else (g.away_score>g.home_score)
            endpoint = TABLE + '?id=eq.' + str(int(r['id'])) + '&status=eq.pending'
            response = requests.patch(endpoint,headers=HEAD,json={'status':'graded','won':bool(won),
                'graded_at':datetime.now(timezone.utc).isoformat()},timeout=30)
            if not response.ok: raise RuntimeError(f'Grade HTTP {response.status_code}: {response.text[:200]}')
            count += 1
        st.success(f'Graded {count} selections. Refresh to update the dashboard.')
    except Exception as exc:
        st.error(f'Grading failed: {exc}')

st.divider()
st.subheader('Performance explained')
if history.empty or 'status' not in history:
    st.info('Metrics will appear once snapshots have been saved and games have finished.')
else:
    graded = history[history.status.eq('graded') & history.won.notna()].copy()
    if graded.empty:
        st.info('No graded games yet. After games finish, use the owner grading control above.')
    else:
        y = graded.won.astype(int).to_numpy()
        raw = np.clip(pd.to_numeric(graded.raw_win_pct).to_numpy()/100,1e-6,1-1e-6)
        cal = np.clip(pd.to_numeric(graded.calibrated_win_pct).to_numpy()/100,1e-6,1-1e-6)
        brier_raw = float(np.mean((raw-y)**2)); brier_cal = float(np.mean((cal-y)**2))
        acc = float(np.mean((cal>=0.5)==y))
        a,b,c = st.columns(3)
        a.metric('Calibrated Brier',f'{brier_cal:.4f}')
        b.metric('Original Brier',f'{brier_raw:.4f}')
        c.metric('Winner accuracy',f'{acc*100:.1f}%')
        st.caption('Brier score measures probability quality; lower is better. Winner accuracy measures whether the more-likely team won. Both sides of each game can appear, so selections are not independent.')
        st.caption(f'Graded team selections: {len(graded)} across {graded.game_id.nunique()} unique games.')
        st.info('Hypothetical betting ROI is intentionally omitted from this dashboard until SDS defines and freezes a pregame selection rule. Saving both sides of every matchup is not a betting strategy.')
        st.dataframe(graded[['game_id','team','opponent','calibrated_win_pct','won']],hide_index=True,use_container_width=True)

st.caption('SDS 5.0 leaves your 4.8 Edge Finder and 4.9.2 manual tracker unchanged. For truly unattended scheduled captures, a separate secured scheduler will be required.')
