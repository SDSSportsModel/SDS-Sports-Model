"""SDS 4.9: persistent, append-only pregame snapshot and grading lab."""
import io
import json
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st
import nflreadpy as nfl

st.set_page_config(page_title='SDS Performance Tracker 4.9', page_icon='📈', layout='wide')
st.title('📈 SDS 4.9 | Performance Tracker')
st.warning('Research only. Hypothetical results are not verified betting returns. No wagers are placed.')
st.caption('Records are stored in Supabase when configured. CSV export is a backup, not automatic persistence.')

def secret(name):
    try: return st.secrets.get(name, '')
    except Exception: return ''
URL = str(secret('SUPABASE_URL')).rstrip('/')
KEY = str(secret('SUPABASE_SERVICE_ROLE_KEY'))
HEAD = {'apikey': KEY, 'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json'}
TABLE = URL + '/rest/v1/sds_snapshots'

def api(method, *, params=None, payload=None):
    headers = dict(HEAD)
    if method == 'POST': headers['Prefer'] = 'return=representation'
    resp = requests.request(method, TABLE, params=params, json=payload, headers=headers, timeout=25)
    if not resp.ok: raise RuntimeError(f'Database returned HTTP {resp.status_code}: {resp.text[:350]}')
    return resp.json() if resp.content else []

def read_records():
    rows = api('GET', params={'select':'*','order':'captured_at.desc','limit':'10000'})
    return pd.DataFrame(rows)

def clean_probability(v):
    v = float(v)
    if not 0 < v < 100: raise ValueError('Probabilities must be strictly between 0 and 100.')
    return v

def odds_profit(odds):
    odds = int(odds)
    if odds == 0 or -100 < odds < 100: raise ValueError('American odds must be <= -100 or >= +100.')
    return odds/100 if odds > 0 else 100/abs(odds)

def normalize_team(s):
    return str(s).strip().upper()

if not URL or not KEY:
    st.error('Persistent storage is not configured. Add SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to Streamlit Secrets, then create the table using the included SQL.')
    st.stop()

st.info('Only the app owner should have access to the service-role key. Never expose it in GitHub or the browser. For a public site, protect write operations with an admin password.')
ADMIN = str(secret('SDS_TRACKER_ADMIN_PASSWORD'))
if not ADMIN:
    st.error('Set SDS_TRACKER_ADMIN_PASSWORD in Streamlit Secrets before using this page.')
    st.stop()
with st.sidebar:
    st.subheader('Tracker admin')
    password = st.text_input('Admin password', type='password')
    admin = bool(password) and password == ADMIN
    st.caption('Read-only performance metrics remain public. Saving and grading require the admin password.')

try:
    history = read_records()
except Exception as exc:
    st.error(f'Unable to read tracker database: {exc}')
    st.stop()

st.subheader('Recorded pregame snapshots')
if history.empty:
    st.info('No records yet. Import a CSV of 4.8 projections and FanDuel odds below to capture the first snapshot.')
else:
    st.metric('Recorded selections', len(history))
    cols = [c for c in ['captured_at','game_id','kickoff_utc','team','opponent','fanduel_ml','raw_win_pct','calibrated_win_pct','status'] if c in history]
    st.dataframe(history[cols], hide_index=True, use_container_width=True)
    st.download_button('Download complete tracker backup', history.to_csv(index=False), 'sds49_snapshot_backup.csv', 'text/csv')

st.divider()
st.subheader('Capture pregame snapshot')
st.caption('Upload a CSV exported from your Edge Finder or prepare one using the included template. Column names must match the template. This page does not automatically import 4.8 calculations.')
upload = st.file_uploader('Snapshot CSV', type=['csv'], key='snap')
required = ['game_id','kickoff_utc','team','opponent','fanduel_ml','raw_win_pct','calibrated_win_pct']
if upload is not None:
    try:
        frame = pd.read_csv(upload, dtype={'game_id':str})
        missing = sorted(set(required)-set(frame.columns))
        if missing: raise ValueError('Missing columns: '+', '.join(missing))
        now = datetime.now(timezone.utc)
        records = []
        for r in frame.to_dict('records'):
            kickoff = pd.to_datetime(r['kickoff_utc'], utc=True, errors='raise').to_pydatetime()
            if kickoff <= now: raise ValueError(f"{r['game_id']}: kickoff has passed; cannot create a pregame record")
            team, opponent = normalize_team(r['team']), normalize_team(r['opponent'])
            if not team or not opponent or team == opponent: raise ValueError('Invalid team/opponent')
            ml = int(r['fanduel_ml'])
            odds_profit(ml)
            raw, cal = clean_probability(r['raw_win_pct']), clean_probability(r['calibrated_win_pct'])
            records.append({'game_id':str(r['game_id']).strip(), 'kickoff_utc':kickoff.isoformat(),
                'team':team,'opponent':opponent,'fanduel_ml':ml,
                'raw_win_pct':raw,'calibrated_win_pct':cal,
                'captured_at':now.isoformat(),'status':'pending'})
        if not records: raise ValueError('CSV is empty')
        st.dataframe(pd.DataFrame(records), hide_index=True)
        if st.button('Save immutable pregame snapshot', disabled=not admin, type='primary'):
            existing = set()
            if not history.empty:
                existing = set(zip(history.game_id.astype(str), history.team.astype(str)))
            duplicates = [(r['game_id'],r['team']) for r in records if (r['game_id'],r['team']) in existing]
            if duplicates: raise ValueError(f'Snapshot already exists for {duplicates[:3]}; this tracker records one snapshot per game/team.')
            if len(set((r['game_id'],r['team']) for r in records)) != len(records):
                raise ValueError('Duplicate game/team rows in upload')
            api('POST', payload=records)
            st.success(f'Saved {len(records)} timestamped records. Reload page to view.')
    except Exception as exc:
        st.error(f'CSV validation failed: {exc}')

st.divider()
st.subheader('Grade completed games')
st.caption('Grade by NFL game_id from nflverse schedules. Only final games are graded; snapshots captured after kickoff are excluded from evaluation.')
if st.button('Refresh results and grade pending', disabled=not admin):
    try:
        schedules = nfl.load_schedules([2026]).to_pandas()
        if 'game_type' in schedules.columns: schedules = schedules[schedules.game_type.eq('REG')]
        final = schedules[schedules.home_score.notna() & schedules.away_score.notna()].copy()
        final = {str(g.game_id):g for g in final.itertuples(index=False)}
        graded = 0
        for r in history.to_dict('records'):
            if r.get('status') != 'pending': continue
            g = final.get(str(r['game_id']))
            if g is None: continue
            captured = pd.to_datetime(r['captured_at'], utc=True)
            kickoff = pd.to_datetime(r['kickoff_utc'], utc=True)
            if captured >= kickoff: continue
            team = normalize_team(r['team'])
            home, away = normalize_team(g.home_team), normalize_team(g.away_team)
            if team not in (home, away): continue
            hs, aws = int(g.home_score), int(g.away_score)
            if hs == aws: continue
            won = (hs > aws) if team == home else (aws > hs)
            update = {'status':'graded','won':bool(won),'graded_at':datetime.now(timezone.utc).isoformat()}
            endpoint = TABLE + '?id=eq.' + str(int(r['id']))
            response = requests.patch(endpoint, headers=HEAD, json=update, timeout=25)
            if not response.ok: raise RuntimeError(response.text[:300])
            graded += 1
        st.success(f'Graded {graded} records. Reload the page to see updated metrics.')
    except Exception as exc:
        st.error(f'Grading failed: {exc}')

if not history.empty and 'status' in history:
    graded = history[history.status.eq('graded') & history.won.notna()].copy()
    st.divider()
    st.subheader('Recorded out-of-sample performance')
    if graded.empty:
        st.info('Performance metrics will appear after recorded games are completed and graded.')
    else:
        y = graded.won.astype(int).to_numpy()
        p = np.clip(pd.to_numeric(graded.calibrated_win_pct).to_numpy()/100, 1e-6, 1-1e-6)
        raw = np.clip(pd.to_numeric(graded.raw_win_pct).to_numpy()/100, 1e-6, 1-1e-6)
        ml = pd.to_numeric(graded.fanduel_ml).to_numpy()
        profit = np.array([odds_profit(x) for x in ml])
        returns = np.where(y == 1, profit, -1.0)
        a,b,c,d = st.columns(4)
        a.metric('Graded selections',len(y))
        b.metric('Calibrated Brier',f'{np.mean((p-y)**2):.4f}')
        c.metric('Raw Brier',f'{np.mean((raw-y)**2):.4f}')
        d.metric('Hypothetical flat-stake ROI',f'{100*np.mean(returns):+.1f}%')
        st.caption('ROI assumes a $1 hypothetical stake on EVERY recorded selection, including both sides if both were saved. This is not an executable betting strategy or a record of real bets.')
        st.caption(f'Calibrated log loss: {-np.mean(y*np.log(p)+(1-y)*np.log(1-p)):.4f}. Winner accuracy: {100*np.mean((p>=0.5)==y):.1f}%.')
        bins = np.linspace(0,1,6)
        group = pd.cut(p, bins=bins, include_lowest=True)
        reliability = pd.DataFrame({'bin':group.astype(str),'predicted':p,'observed':y}).groupby('bin',as_index=False).agg(predicted=('predicted','mean'),observed=('observed','mean'),n=('observed','size'))
        st.dataframe(reliability, hide_index=True)
        st.line_chart(reliability.set_index('predicted')['observed'])
        st.caption('Small samples and overlapping selections make performance estimates noisy. Closing-line value requires a separate timestamped closing-odds feed; not implemented in 4.9.')
