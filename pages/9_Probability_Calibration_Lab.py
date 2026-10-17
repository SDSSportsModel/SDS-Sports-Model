"""SDS 4.7: out-of-sample probability calibration and CSV snapshot audit."""
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import nflreadpy as nfl

st.set_page_config(page_title='SDS Calibration Lab 4.7', page_icon='🎯', layout='wide')
st.title('🎯 SDS Probability Calibration Lab — 4.7')
st.warning('Experimental research only. This page does not establish profitable wagers or verified betting edges.')
st.caption('Historical predictions use prior NFL weeks and prior seasons. Calibration is fitted on 2024 only and evaluated on unseen 2025 games.')
START=2022

@st.cache_data(ttl=21600, show_spinner=False)
def load_history():
    schedules=nfl.load_schedules([2022,2023,2024,2025]).to_pandas()
    chunks=[]
    for year in (2022,2023,2024,2025):
        pbp=nfl.load_pbp([year]).to_pandas()
        pbp=pbp[['game_id','posteam','defteam','epa','success','play_type']]
        pbp=pbp[pbp.play_type.isin(['run','pass']) & pbp.posteam.notna() & pbp.defteam.notna() & pbp.epa.notna()]
        agg=pbp.groupby(['game_id','posteam','defteam'],as_index=False).agg(off_epa=('epa','mean'),off_success=('success','mean'),plays=('epa','size'))
        chunks.append(agg)
    return schedules,pd.concat(chunks,ignore_index=True)

def assemble(schedule, play_stats):
    games=schedule.copy()
    if 'game_type' in games: games=games[games.game_type.eq('REG')]
    games=games.sort_values(['season','week','game_id'])
    stats={}
    for r in play_stats.itertuples(index=False):
        stats[(str(r.game_id),str(r.posteam))]=(float(r.off_epa),float(r.off_success) if pd.notna(r.off_success) else 0.0,int(r.plays))
    history={}; rows=[]
    for (season,week),group in games.groupby(['season','week'],sort=True):
        updates=[]
        for g in group.itertuples(index=False):
            home,away,gid=str(g.home_team),str(g.away_team),str(g.game_id)
            def prior(team):
                h=history.get(team,[])[-5:]
                if not h: return [0.0]*5
                a=np.asarray(h,dtype=float)
                return [float(x) for x in a.mean(axis=0)[:4]]+[len(h)/5.0]
            h,a=prior(home),prior(away)
            features=[h[0]-a[0],h[1]-a[1],h[2]-a[2],h[3]-a[3],h[4],a[4]]
            complete=pd.notna(g.home_score) and pd.notna(g.away_score)
            if complete:
                margin=float(g.home_score-g.away_score)
                rows.append(dict(season=int(season),week=int(week),game_id=gid,home=home,away=away,margin=margin,home_win=int(margin>0),features=features))
                hs,az=stats.get((gid,home)),stats.get((gid,away))
                if hs and az and hs[2]>=10 and az[2]>=10:
                    updates.extend([(home,[hs[0],az[0],hs[1],az[1]]),(away,[az[0],hs[0],az[1],hs[1]])])
        for team,values in updates: history.setdefault(team,[]).append(values)
    return pd.DataFrame(rows)

def ridge_fit(df,penalty=15.0):
    x=np.column_stack([np.ones(len(df)),np.asarray(df.features.tolist(),dtype=float)])
    y=df.margin.to_numpy(dtype=float)
    reg=np.diag([0.0]+[penalty]*(x.shape[1]-1))
    beta=np.linalg.solve(x.T@x+reg,x.T@y)
    sd=max(float(np.std(y-x@beta,ddof=min(x.shape[1],len(y)-1))),1.0)
    return beta,sd

def raw_prob(df,beta,sd):
    x=np.column_stack([np.ones(len(df)),np.asarray(df.features.tolist(),dtype=float)])
    margins=x@beta
    p=np.asarray([0.5*(1+math.erf(float(v/sd)/math.sqrt(2))) for v in margins])
    return np.clip(p,1e-6,1-1e-6)

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

def metrics(y,p):
    y=np.asarray(y,dtype=float); p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return dict(Games=len(y),**{'Winner accuracy %':round(100*np.mean((p>=.5)==y),2),'Brier score':round(float(np.mean((p-y)**2)),5),'Log loss':round(float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p))),5)})

def reliability(y,p):
    f=pd.DataFrame({'actual':y,'probability':p})
    f['bucket']=pd.cut(f.probability,bins=np.linspace(0,1,11),include_lowest=True)
    r=f.groupby('bucket',observed=False).agg(games=('actual','size'),predicted=('probability','mean'),observed=('actual','mean')).reset_index()
    r=r[r.games>0].copy()
    r['Probability bin']=r.bucket.astype(str)
    return r[['Probability bin','games','predicted','observed']]

try:
    with st.spinner('Loading historical NFL data and reconstructing pregame features...'):
        schedules,pbp=load_history()
        df=assemble(schedules,pbp)
    preds=[]
    for season in (2024,2025):
        train=df[df.season<season]
        test=df[df.season==season].copy()
        if len(train)<200 or test.empty: raise ValueError(f'Not enough data for {season} holdout')
        beta,sd=ridge_fit(train)
        test['raw_probability']=raw_prob(test,beta,sd)
        preds.append(test)
    result=pd.concat(preds,ignore_index=True)
    calibration=result[result.season==2024]
    holdout=result[result.season==2025].copy()
    params=fit_platt(calibration.raw_probability,calibration.home_win)
    holdout['calibrated_probability']=sigmoid(params[0]+params[1]*logit(holdout.raw_probability))
    st.subheader('Independent 2025 calibration evaluation')
    st.caption('EPA model trained through 2024 for 2025 predictions. Calibrator trained on 2024 predictions generated by a model trained through 2023. Neither stage fits on 2025 outcomes.')
    scores=pd.DataFrame([{'Model':'Uncalibrated EPA (2025)',**metrics(holdout.home_win,holdout.raw_probability)},{'Model':'2024-trained calibrated EPA (2025)',**metrics(holdout.home_win,holdout.calibrated_probability)}])
    st.dataframe(scores,hide_index=True,use_container_width=True)
    st.info('A lower Brier score and log loss indicate better probability estimates. Small differences may be noise; calibration must be monitored over future seasons.')
    st.subheader('Reliability chart — 2025 holdout')
    mode=st.radio('Probability version',('Uncalibrated','Calibrated'),horizontal=True)
    col='raw_probability' if mode=='Uncalibrated' else 'calibrated_probability'
    chart=reliability(holdout.home_win,holdout[col])
    st.line_chart(chart.set_index('Probability bin')[['predicted','observed']])
    st.dataframe(chart.rename(columns={'games':'Games','predicted':'Average predicted','observed':'Actual win rate'}),hide_index=True,use_container_width=True)
    st.caption('Each row groups home-team probabilities into ten-point bins. Small sample sizes make individual bins noisy. A perfect calibration chart would have predicted and observed rates equal.')
    st.subheader('Historical prediction audit')
    st.caption('These historical rows were reconstructed using past data. They are NOT timestamped pregame records or evidence of previously published picks.')
    view=result[['season','week','game_id','away','home','home_win','raw_probability']].copy()
    view['record_type']='reconstructed_historical_backtest'
    view['generated_at_utc']=datetime.now(timezone.utc).isoformat()
    st.download_button('Download historical backtest CSV',view.to_csv(index=False),file_name='sds_47_reconstructed_backtest.csv',mime='text/csv')
    st.subheader('Future snapshot tracking')
    st.markdown('To create genuine pregame records, export the **current** comparison CSV from SDS 4.6 **before kickoff**, and save it with its export timestamp. Keep the original file unchanged. Later, compare the saved prices and predictions against game results.')
    st.warning('Streamlit Community Cloud does not provide durable storage for files saved only in the app runtime. CSV downloads are manual snapshots, not an automatic historical database. Automated archival requires persistent storage and scheduled collection.')
    st.caption('SDS 4.7 does not automatically replace the uncalibrated probabilities in SDS 4.6. First compare 2025 results and audit the model pipeline.')
except Exception as exc:
    st.error('Calibration Lab could not complete. Existing SDS pages are unaffected.')
    st.exception(exc)
