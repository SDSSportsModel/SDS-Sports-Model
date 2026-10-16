"""SDS Sports Model — public-facing Streamlit website (sample data, no live feeds)."""
from pathlib import Path
import pandas as pd
import streamlit as st
from odds_engine import MarketQuote, implied_probability, no_vig_two_way

st.set_page_config(page_title="SDS Sports Model | Betting Analytics", page_icon="🏈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');
html,body,[class*="css"],.stApp {font-family:'DM Sans',sans-serif;}
.stApp {background:#09121c;color:#edf5f8;}
[data-testid="stSidebar"] {background:#101e2a;border-right:1px solid #263948;}
h1,h2,h3 {font-family:'Outfit',sans-serif!important;letter-spacing:-.035em;}
h1 {font-size:3.1rem!important;}
[data-testid="stMetric"] {background:#122532;border:1px solid #29414d;padding:18px;border-radius:16px;}
[data-testid="stMetricLabel"] {color:#a7c2cd;}
[data-testid="stMetricValue"] {color:#eafbf5;}
.stButton>button[kind="primary"] {background:#24d6a3;color:#071c1b;border:0;font-weight:800;}
.stTabs [data-baseweb="tab-list"] {gap:16px;}
.stTabs [data-baseweb="tab"] {font-weight:700;}
.hero {padding:27px 30px;border-radius:22px;background:linear-gradient(115deg,#173f43 0%,#112637 56%,#18203c 100%);border:1px solid #2d5b5d;margin-bottom:18px;}
.hero .eyebrow {color:#55e5b5;letter-spacing:.14em;font-size:.8rem;font-weight:800;}
.hero h1 {margin:.2em 0 .2em;color:#f5fffc;}
.hero p {font-size:1.04rem;color:#b9d1d5;max-width:760px;}
.pill {display:inline-block;padding:5px 11px;border-radius:30px;border:1px solid #54727b;color:#b7d4d7;font-size:.78rem;margin:8px 6px 0 0;}
.disclaimer {padding:14px 17px;border:1px solid #786441;border-radius:12px;background:#332b1c;color:#f6dcae;}
</style>""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "sample_markets.csv"
REQUIRED = {"game", "market", "selection", "line", "american_odds", "model_probability"}

@st.cache_data
def load_sample():
    return pd.read_csv(DATA_PATH)

def score_markets(df):
    rows = []
    for _, r in df.iterrows():
        try:
            q = MarketQuote(game_id=str(r['game']), market=str(r['market']), selection=str(r['selection']),
                            line=None if pd.isna(r['line']) else float(r['line']),
                            american_odds=int(r['american_odds']), model_probability=float(r['model_probability']))
            out = q.evaluate()
            rows.append({"Matchup":r['game'],"Market":r['market'],"Selection":r['selection'],
                         "Line":r['line'],"Odds":int(r['american_odds']),
                         "Model %":100*out['model_probability'],"Book %":100*out['implied_probability'],
                         "Edge (pp)":100*out['probability_edge'],"Expected ROI %":100*out['expected_roi'],
                         "¼ Kelly %":100*out['quarter_kelly_fraction']})
        except (ValueError, TypeError, KeyError, OverflowError):
            continue
    return pd.DataFrame(rows)

# Public NFL data is a separate information panel, never treated as a trained projection.
@st.cache_data(ttl=900, show_spinner=False)
def get_public_nfl_stats():
    from nfl_stats import fetch_stats
    return fetch_stats()

with st.sidebar:
    st.markdown("## 🏈 SDS SPORTS MODEL")
    st.caption("BETTING INTELLIGENCE / V1.0")
    st.divider()
    st.markdown("**Data source**")
    source = st.radio("Choose a dataset", ["Illustrative sample", "Upload my projections"], label_visibility="collapsed")
    uploaded = st.file_uploader("Upload CSV with market probabilities", type="csv") if source == "Upload my projections" else None
    st.caption("Public NFL stats are retrieved from nflverse; odds and predictions are not live.")
    st.divider()
    st.markdown("**Screening controls**")
    min_edge = st.slider("Minimum probability edge (pp)", -10.0, 20.0, 0.0, 0.5)
    min_roi = st.slider("Minimum expected ROI (%)", -25.0, 50.0, 0.0, 1.0)
    st.divider()
    st.caption("For educational and analytical use. No bets are placed through this website.")

with st.expander("🏈 REAL NFL STATS • 2026 season", expanded=True):
    st.caption("Public nflverse data. Updates depend on the upstream release schedule; not live in-game tracking or model forecasts.")
    a, b = st.columns([3, 1])
    with a:
        st.markdown("**Schedules, team averages, and player leaderboards**")
    with b:
        if st.button("Refresh stats", help="Clear cached data and request the latest available nflverse release"):
            get_public_nfl_stats.clear()
    try:
        with st.spinner("Loading NFL statistics from nflverse…"):
            stats = get_public_nfl_stats()
        from nfl_stats import weekly_team_summary, upcoming_games, player_leaderboard
        st.success("NFL data connected")
        st.caption("Retrieved at: " + stats['fetched_at'] + " UTC. Individual datasets may have earlier upstream update times.")
        t1,t2,t3 = st.tabs(["Upcoming games", "Team statistics", "Player leaders"])
        with t1:
            schedule = upcoming_games(stats['schedules'])
            if schedule.empty: st.info("No upcoming regular-season games found in the available schedule.")
            else: st.dataframe(schedule, hide_index=True, use_container_width=True)
        with t2:
            summary = weekly_team_summary(stats['teams'])
            if summary.empty: st.info("No current-season team statistics available yet.")
            else:
                team = st.selectbox("Team", ["All teams"] + sorted(summary['team'].dropna().unique().tolist()))
                st.dataframe(summary if team == "All teams" else summary[summary['team'].eq(team)], hide_index=True, use_container_width=True)
        with t3:
            metric = st.selectbox("Leaderboard", ["passing_yards", "rushing_yards", "receiving_yards", "receptions", "passing_tds", "rushing_tds", "receiving_tds"])
            leaders = player_leaderboard(stats['players'], stat=metric)
            if leaders.empty: st.info("This metric is not available in the current source data.")
            
try:
    if leaders.empty:
        st.info("No player data available.")
    else:
    display_cols = [
        col for col in [
            "player_name",
            "recent_team",
            metric,
            "position"
        ]
        if col in leaders.columns
    ]

    st.dataframe(
        leaders[display_cols],
        hide_index=True,
        use_container_width=True,
        column_config={
            "player_name": "Player",
            "recent_team": "Team",
            metric: st.column_config.NumberColumn(
                metric.replace("_", " ").title(),
                format="%d"
            ),
            "position": "Pos"
        }
    )

    except Exception as exc:
        st.warning("NFL statistics are temporarily unavailable. The rest of the site remains usable.")
        st.caption(f"Technical detail: {type(exc).__name__}: {exc}")

st.markdown("""<div class="hero"><div class="eyebrow">NFL ANALYTICS PLATFORM</div>
<h1>Find the edge. Understand the numbers.</h1>
<p>Explore moneylines, spreads, totals, player props and touchdown markets in one clean dashboard. Compare your probability estimates with sportsbook pricing.</p>
<span class="pill">5 market families</span><span class="pill">Expected value analysis</span><span class="pill">Transparent calculations</span></div>""", unsafe_allow_html=True)

if source == "Upload my projections":
    if uploaded is None:
        st.info("Upload a CSV to populate the dashboard. Download the sample template below to see the expected columns.")
        st.download_button("Download CSV template", load_sample().to_csv(index=False), "market_template.csv", "text/csv")
        st.stop()
    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
    missing = REQUIRED - set(raw.columns)
    if missing:
        st.error("Missing columns: " + ", ".join(sorted(missing)))
        st.stop()
    raw = raw.copy()
    raw['american_odds'] = pd.to_numeric(raw['american_odds'], errors='coerce')
    raw['model_probability'] = pd.to_numeric(raw['model_probability'], errors='coerce')
    raw['line'] = pd.to_numeric(raw['line'], errors='coerce')
    raw = raw.dropna(subset=['american_odds', 'model_probability'])
    raw = raw[(raw['american_odds'] != 0) & raw['model_probability'].between(0, 1)]
    st.success(f"Loaded {len(raw)} valid rows from your uploaded dataset.")
else:
    raw = load_sample()
    st.markdown('<div class="disclaimer">⚠️ <b>DEMO MODE:</b> All matchups, odds and probabilities shown below are fictional examples, not current FanDuel lines or trained-model predictions. Do not use these values to place bets.</div>', unsafe_allow_html=True)

scored = score_markets(raw)
if scored.empty:
    st.warning("No valid markets available. Check your odds and probability inputs.")
    st.stop()

all_markets = sorted(scored['Market'].unique())
all_games = sorted(scored['Matchup'].unique())
filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    selected_markets = st.multiselect("Betting markets", all_markets, default=all_markets)
with filter_col2:
    selected_games = st.multiselect("Games", all_games, default=all_games)
view = scored[scored['Market'].isin(selected_markets) & scored['Matchup'].isin(selected_games)]
view = view[(view['Edge (pp)'] >= min_edge) & (view['Expected ROI %'] >= min_roi)].sort_values('Expected ROI %', ascending=False)

metrics = st.columns(4)
metrics[0].metric("Markets displayed", f"{len(view):,}")
metrics[1].metric("Positive EV markets", f"{(view['Expected ROI %'] > 0).sum():,}")
metrics[2].metric("Highest estimated edge", f"{view['Edge (pp)'].max():+.1f} pp" if not view.empty else "—")
metrics[3].metric("Highest estimated ROI", f"{view['Expected ROI %'].max():+.1f}%" if not view.empty else "—")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Market board", "🧮 Bet calculator", "📈 Insights", "ℹ️ About the model"])
with tab1:
    st.subheader("Market opportunity board")
    st.caption("Sorted by expected ROI. Green highlights are mathematical estimates, not guarantees.")
    st.dataframe(view.style.format({"Line":"{:.1f}","Model %":"{:.1f}%","Book %":"{:.1f}%","Edge (pp)":"{:+.1f}","Expected ROI %":"{:+.1f}%","¼ Kelly %":"{:.2f}%"}).background_gradient(subset=['Expected ROI %'], cmap='YlGn'), use_container_width=True, hide_index=True)
    st.download_button("Download filtered analysis (CSV)", view.to_csv(index=False), "nfl_market_analysis.csv", "text/csv", disabled=view.empty)
with tab2:
    st.subheader("Analyze any market")
    c1, c2, c3 = st.columns(3)
    with c1: odds = st.number_input("American odds", value=-110, step=5, help="Negative or positive American odds, excluding zero")
    with c2: prob_pct = st.number_input("Your projected win probability (%)", min_value=0.0, max_value=100.0, value=55.0, step=0.5)
    with c3: bankroll = st.number_input("Example bankroll ($)", min_value=0.0, value=1000.0, step=50.0)
    if odds == 0:
        st.error("American odds cannot equal zero.")
    else:
        q = MarketQuote("Custom", "Custom", "Your selection", None, int(odds), prob_pct / 100).evaluate()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Book implied probability", f"{q['implied_probability']*100:.1f}%")
        c2.metric("Probability edge", f"{q['probability_edge']*100:+.1f} pp")
        c3.metric("Expected ROI", f"{q['expected_roi']*100:+.1f}%")
        c4.metric("¼ Kelly illustrative stake", f"${bankroll*q['quarter_kelly_fraction']:.2f}")
        st.caption("Kelly sizing is highly sensitive to probability estimation error; the calculation is not a recommendation to wager.")
    with st.expander("Compare two-way prices without the sportsbook margin"):
        a,b = st.columns(2)
        with a: odds_a = st.number_input("Side A odds", value=-115, step=5)
        with b: odds_b = st.number_input("Side B odds", value=-105, step=5)
        if odds_a and odds_b:
            pa,pb = no_vig_two_way(int(odds_a), int(odds_b))
            st.write(f"Normalized market probabilities: **Side A {pa:.1%}** · **Side B {pb:.1%}**")
with tab3:
    st.subheader("Estimated edge by market")
    if view.empty:
        st.info("Adjust the filters to view charts.")
    else:
        chart = view.groupby('Market', as_index=False).agg(**{'Average expected ROI %':('Expected ROI %','mean'), 'Market count':('Market','size')})
        st.bar_chart(chart.set_index('Market')['Average expected ROI %'], horizontal=True)
        st.caption("Sample averages reflect only the filtered dataset, not historical performance.")
        st.subheader("Top 10 estimated opportunities")
        top = view.head(10).copy()
        top['Label'] = top['Selection'] + ' · ' + top['Market']
        st.bar_chart(top.set_index('Label')['Expected ROI %'], horizontal=True)
with tab4:
    st.subheader("What this version can and cannot do")
    st.markdown("""**Available today:** a public-friendly interface, CSV upload, multi-market screening, odds conversion, two-way no-vig pricing, expected ROI, fractional Kelly math, charts, and export.\n\n**Now included:** nflverse NFL schedules, team statistics and player leaderboards with hourly caching.\n\n**Not yet implemented:** trained NFL forecasts, injury-adjusted player distributions, real-time sportsbook feeds, historical backtesting, or verified profitable signals. You must supply your own calibrated probabilities and odds to analyze real markets.\n\n**CSV schema:** `game, market, selection, line, american_odds, model_probability` — probability is a decimal between 0 and 1.\n\n**Risk:** Model estimates can be wrong; markets move and correlations matter. Do not interpret positive estimated EV as a guaranteed profit.""")
    st.download_button("Download example CSV", load_sample().to_csv(index=False), "sample_markets.csv", "text/csv")

st.divider()
st.caption("SDS Sports Model • Independent analytics demo • Not affiliated with the NFL or FanDuel • No wagering functionality")
