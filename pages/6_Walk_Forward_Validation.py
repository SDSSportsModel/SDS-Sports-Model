"""SDS 5.4 | Walk-forward validation and probability consistency diagnostics.

Add this file to pages/. Requires the existing root-level sds50_engine.py.
Read-only: does not modify the public forecast, Supabase, or sportsbook data.
"""
import math
import numpy as np
import pandas as pd
import streamlit as st
from sds50_engine import load_model_data, assemble, ridge_fit, forecast, normal_cdf, fit_platt, sigmoid, logit

st.set_page_config(page_title='SDS 5.4 | Validation Lab', page_icon='🧪', layout='wide')
st.title('🧪 SDS 5.4 | Walk-Forward Validation')
st.caption('Read-only historical validation • No changes to the live model or recorded snapshots')
st.warning('Research only. Historical predictive performance does not establish a profitable betting strategy.')

@st.cache_data(ttl=21600, show_spinner=False)
def get_data():
    schedules, pbp, status, n = load_model_data()
    return assemble(schedules, pbp), status, n

def predict_raw(rows, beta, sd):
    margins = forecast(rows, beta)
    return np.clip(np.array([normal_cdf(v/sd) for v in margins]), 1e-6, 1-1e-6), margins

def metrics(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
    y = np.asarray(y, float)
    return {'Brier': float(np.mean((p-y)**2)),
            'Log loss': float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
            'Winner accuracy %': float(100*np.mean((p>=.5)==y))}

with st.spinner('Loading NFL schedules and historical EPA features...'):
    try:
        data, data_status, current_games = get_data()
    except Exception as exc:
        st.error(f'Could not load the existing SDS engine: {type(exc).__name__}: {exc}')
        st.stop()

st.subheader('1. How the historical test works')
st.markdown('For each evaluation season, fit the **score model on earlier seasons**, fit probability calibration on the **immediately preceding season**, then evaluate on the next season. No evaluation-season results are used to fit either component.')
st.caption('2024 test: score model trained on 2022; calibrator fitted on 2023. 2025 test: score model trained on 2022–2023; calibrator fitted on 2024. These are two season-level forward tests, not a fully independent prospective study.')

# The calibration score model and evaluation score model are intentionally distinct:
# evaluation model may incorporate the calibration season, but neither sees evaluation outcomes.
folds = []
fold_predictions = []
for test_season in (2024, 2025):
    cal_season = test_season - 1
    base_train = data[data.completed & data.season.lt(cal_season)]
    cal_rows = data[data.completed & data.season.eq(cal_season)]
    eval_train = data[data.completed & data.season.lt(test_season)]
    test_rows = data[data.completed & data.season.eq(test_season)]
    if min(len(base_train),len(cal_rows),len(eval_train),len(test_rows)) < 100:
        st.warning(f'Skipped {test_season}: insufficient historical games for leakage-safe calibration.')
        continue
    try:
        base_beta, base_sd = ridge_fit(base_train, 'margin')
        cal_raw, _ = predict_raw(cal_rows, base_beta, base_sd)
        cal_y = (cal_rows.margin.to_numpy(float)>0).astype(float)
        platt = fit_platt(cal_raw, cal_y)
        eval_beta, eval_sd = ridge_fit(eval_train, 'margin')
        raw, predicted_margin = predict_raw(test_rows, eval_beta, eval_sd)
        full = sigmoid(platt[0] + platt[1]*logit(raw))
        y = (test_rows.margin.to_numpy(float)>0).astype(float)
        # Fixed blend weights, shown for diagnosis only. No 'winner' is auto-selected.
        for weight in (0, .25, .5, .75, 1):
            prob = (1-weight)*raw + weight*full
            m = metrics(y, prob)
            folds.append({'Test season':test_season, 'Calibration weight':f'{int(weight*100)}%',
                          'Games':len(y), **m})
        for i, r in enumerate(test_rows.itertuples(index=False)):
            fold_predictions.append({'Test season':test_season,'game_id':str(r.game_id),
                                     'home':str(r.home),'away':str(r.away),
                                     'predicted_margin':float(predicted_margin[i]),
                                     'actual_margin':float(r.margin),
                                     'raw_home_prob':float(raw[i]),'full_home_prob':float(full[i]),
                                     'home_won':int(y[i])})
    except Exception as exc:
        st.error(f'{test_season} fold failed: {type(exc).__name__}: {exc}')

if not folds:
    st.stop()
results = pd.DataFrame(folds)
preds = pd.DataFrame(fold_predictions)
st.subheader('2. Walk-forward test results')
st.dataframe(results.style.format({'Brier':'{:.4f}','Log loss':'{:.4f}','Winner accuracy %':'{:.1f}'}),
             hide_index=True, use_container_width=True)
st.caption('Lower Brier and log loss are better. Calibration weights are predeclared comparisons, not a new production setting. Each game is evaluated once per test season.')

summary_rows = []
for weight in ('0%', '25%', '50%', '75%', '100%'):
    g = results[results['Calibration weight'].eq(weight)]
    if g.empty:
        continue
    summary_rows.append({'Calibration weight':weight, 'Games':int(g.Games.sum()),
                         'Weighted Brier':float(np.average(g.Brier, weights=g.Games)),
                         'Weighted log loss':float(np.average(g['Log loss'], weights=g.Games)),
                         'Weighted winner accuracy %':float(np.average(g['Winner accuracy %'],weights=g.Games))})
summary = pd.DataFrame(summary_rows)
st.markdown('**Combined historical view (descriptive, not a model-selection verdict)**')
st.dataframe(summary, hide_index=True, use_container_width=True)

st.subheader('3. Probability vs. projected-score consistency')
st.markdown('A model that predicts a narrow victory but an overwhelming win probability needs closer inspection. We compare the calibrated probability with the **normal-distribution probability implied by the same score-margin model**.')
try:
    # Current production-style fit, used for DIAGNOSTICS only; this does not change predictions.
    train = data[data.completed & data.season.between(2022, 2025)]
    future = data[data.season.eq(2026) & (~data.completed)].copy()
    if len(train) < 300 or future.empty:
        st.info('Not enough training data or no future games available for current-season consistency checks.')
    else:
        beta, sd = ridge_fit(train, 'margin')
        # Fit production-style 2024 calibration exactly as defined by the engine.
        from sds50_engine import fit_2024_calibrator
        cal_b = fit_2024_calibrator(data)
        raw, margin = predict_raw(future, beta, sd)
        cal = sigmoid(cal_b[0] + cal_b[1]*logit(raw))
        diagnostics = future[['week','game_id','away','home']].copy().reset_index(drop=True)
        diagnostics['Predicted home margin'] = np.round(margin, 1)
        diagnostics['Margin-implied home win %'] = np.round(raw*100,1)
        diagnostics['Calibrated home win %'] = np.round(cal*100,1)
        diagnostics['Difference (pp)'] = np.round(np.abs(cal-raw)*100,1)
        threshold = st.slider('Flag probability differences of at least (percentage points)',5,30,15,1)
        diagnostics['Check'] = np.where(diagnostics['Difference (pp)']>=threshold,'Review','Within threshold')
        flagged = diagnostics[diagnostics.Check.eq('Review')].sort_values('Difference (pp)',ascending=False)
        a,b = st.columns(2)
        a.metric('Future games checked',len(diagnostics))
        b.metric('Games flagged',len(flagged))
        if flagged.empty:
            st.success('No upcoming games exceed the selected discrepancy threshold.')
        else:
            st.dataframe(flagged, hide_index=True, use_container_width=True)
        with st.expander('All future games'):
            st.dataframe(diagnostics, hide_index=True, use_container_width=True)
        st.caption(f'This diagnostic uses the production-style 2022–2025 margin fit and 2024 Platt calibrator; residual SD = {sd:.2f} points. The margin-implied probability is a normal-residual approximation, not an independently validated probability model.')
except Exception as exc:
    st.error(f'Current-season diagnostic unavailable: {type(exc).__name__}: {exc}')

st.subheader('4. Next validation checkpoint')
st.info('Freeze any chosen calibration rule before collecting more 2026 results. Use pregame timestamped forecasts and evaluate them only after games finish. Do not tune repeatedly on the same holdout.')
st.download_button('Download walk-forward game-level diagnostics',preds.to_csv(index=False),
                   'sds54_walk_forward_diagnostics.csv','text/csv')
