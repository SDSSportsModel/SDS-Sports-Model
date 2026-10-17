"""SDS 5.3.2 — time-ordered probability calibration diagnostics (read-only)."""
import numpy as np
import pandas as pd
import streamlit as st
from sds50_engine import (load_model_data, assemble, ridge_fit, forecast,
                          normal_cdf, fit_platt, sigmoid, logit)

st.set_page_config(page_title='SDS 5.3.2 | Calibration Lab', page_icon='🧪', layout='wide')
st.title('🧪 SDS 5.3.2 | Calibration Diagnostic Lab')
st.caption('Independent 2025 evaluation • No changes to public forecasts or saved predictions')
st.warning('Research only. This evaluates historical probability quality, not betting profitability.')

@st.cache_data(ttl=21600, show_spinner=False)
def get_predictions():
    schedules, pbp, status, n_current = load_model_data()
    data = assemble(schedules, pbp)
    a = data[data.completed & data.season.between(2022, 2023)].copy()
    b = data[data.completed & data.season.eq(2024)].copy()
    c = data[data.completed & data.season.between(2022, 2024)].copy()
    d = data[data.completed & data.season.eq(2025)].copy()
    if min(len(a),len(b),len(c),len(d)) < 200:
        raise ValueError('Insufficient historical completed games for 2022–2025 validation.')
    beta23, sd23 = ridge_fit(a, 'margin')
    p24 = np.clip(np.array([normal_cdf(m/sd23) for m in forecast(b,beta23)]),1e-6,1-1e-6)
    y24 = (b.margin.to_numpy(dtype=float)>0).astype(int)
    platt = fit_platt(p24,y24)
    beta24, sd24 = ridge_fit(c, 'margin')
    p25 = np.clip(np.array([normal_cdf(m/sd24) for m in forecast(d,beta24)]),1e-6,1-1e-6)
    y25 = (d.margin.to_numpy(dtype=float)>0).astype(int)
    return p24,y24,p25,y25,platt,sd24,len(a),len(b),len(d)

def calibrated(p, coeff, strength):
    # Blend in log-odds space: 0 = unchanged; 1 = fully Platt-calibrated.
    original = logit(p)
    full = coeff[0]+coeff[1]*original
    return np.clip(sigmoid((1-strength)*original + strength*full),1e-6,1-1e-6)

def scores(p,y):
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return {'Brier score':float(np.mean((p-y)**2)),
            'Log loss':float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
            'Winner accuracy %':float(100*np.mean((p>=.5)==y))}

try:
    with st.spinner('Loading NFL history and calculating held-out predictions...'):
        p24,y24,p25,y25,coeff,sd,ntrain,ncal,ntest=get_predictions()
except Exception as exc:
    st.error(f'Unable to complete calibration test: {type(exc).__name__}: {exc}')
    st.stop()

st.subheader('How the test works')
st.markdown('**Train the score model:** 2022–2023 → **Fit calibration:** 2024 → **Evaluate:** 2025. '
            'For the 2025 test, the score model is retrained using games through 2024; the calibrator never sees 2025 outcomes.')
st.caption(f'2024 calibration games: {ncal} · 2025 independent test games: {ntest} · Score-model residual SD for 2025: {sd:.2f} points')

# Candidate weights are fixed in code, not tuned on the 2025 test results.
strengths=[0.0,0.25,0.5,0.75,1.0]
labels={0.0:'Original (no calibration)',0.25:'25% calibration',0.5:'50% calibration',
        0.75:'75% calibration',1.0:'Full 2024 Platt calibration'}
rows=[]
for w in strengths:
    r24=scores(calibrated(p24,coeff,w),y24)
    r25=scores(calibrated(p25,coeff,w),y25)
    rows.append({'Method':labels[w], 'Calibration strength':w,
                 '2024 Brier (fit data)':r24['Brier score'],
                 '2025 Brier':r25['Brier score'],
                 '2025 Log loss':r25['Log loss'],
                 '2025 Winner accuracy %':r25['Winner accuracy %']})
results=pd.DataFrame(rows)
st.subheader('Independent 2025 results')
st.dataframe(results.drop(columns='Calibration strength').style.format({
    '2024 Brier (fit data)':'{:.4f}','2025 Brier':'{:.4f}',
    '2025 Log loss':'{:.4f}','2025 Winner accuracy %':'{:.1f}'}),
    hide_index=True,use_container_width=True)
st.caption('Lower Brier score and log loss are better. 2024 fit-data results are descriptive, NOT independent validation. '
           'The 2025 holdout should not be repeatedly used to optimize the model.')

base=scores(p25,y25)
full=scores(calibrated(p25,coeff,1.0),y25)
col1,col2,col3=st.columns(3)
col1.metric('Original 2025 Brier',f"{base['Brier score']:.4f}")
col2.metric('Full calibration Brier',f"{full['Brier score']:.4f}")
col3.metric('Full minus original',f"{full['Brier score']-base['Brier score']:+.4f}")

st.subheader('Do large calibration changes help?')
full_p=calibrated(p25,coeff,1.0)
change=np.abs(full_p-p25)
threshold=st.slider('Minimum absolute probability change (percentage points)',5,40,15,5)
mask=change>=threshold/100
if not mask.any():
    st.info('No 2025 games reached this adjustment threshold.')
else:
    raw_err=(p25[mask]-y25[mask])**2
    cal_err=(full_p[mask]-y25[mask])**2
    better=int(np.sum(cal_err<raw_err-1e-12))
    worse=int(np.sum(cal_err>raw_err+1e-12))
    st.write(f'**{mask.sum()} of {len(y25)} games** had an adjustment of at least {threshold} percentage points. '
             f'Calibration reduced squared error on **{better}**, increased it on **{worse}**.')
    st.write(f"Original subgroup Brier: **{raw_err.mean():.4f}** · "
             f"Fully calibrated subgroup Brier: **{cal_err.mean():.4f}**")
    st.caption('This is a diagnostic subgroup selected by prediction change, not a separate holdout. '
               'The same 2025 games are counted in the overall results above.')

with st.expander('View probability adjustments for each 2025 game'):
    details=pd.DataFrame({'Original win %':100*p25,'Fully calibrated win %':100*full_p,
                          'Adjustment (pp)':100*(full_p-p25),'Home team won':y25.astype(bool),
                          'Original squared error':(p25-y25)**2,
                          'Calibrated squared error':(full_p-y25)**2})
    st.dataframe(details.round(3),hide_index=True,use_container_width=True)
    st.download_button('Download 2025 diagnostics CSV',details.to_csv(index=False),
                       'sds532_calibration_2025_diagnostics.csv','text/csv')

st.subheader('What the results mean')
delta=full['Brier score']-base['Brier score']
if delta < -0.002:
    st.success('Full calibration scored better than the original probabilities on 2025 Brier score. '
               'This is encouraging but does not prove future improvement.')
elif delta > 0.002:
    st.warning('Full calibration scored worse than the original probabilities on 2025 Brier score. '
               'Avoid promoting its stronger percentages until the calibration method is revised and independently retested.')
else:
    st.info('Full calibration and original probabilities were close on 2025 Brier score. '
            'The evidence is too weak to justify changing the public model based on this comparison alone.')
st.caption('Important: selecting the best-looking method from this 2025 table and deploying it would reuse the holdout for model selection. '
           'Freeze any revised method first, then validate it prospectively on new games or a genuinely untouched season. '
           'Neither historical win accuracy nor probability calibration establishes a profitable betting edge.')
