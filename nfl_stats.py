"""NFL public stats ingestion. Source: nflverse via nflreadpy; no live-play guarantee."""
from datetime import datetime, timezone
import pandas as pd
import nflreadpy as nfl

SEASONS = [2024, 2025, 2026]

def _pandas(frame):
    return frame.to_pandas()

def fetch_stats():
    """Fetch current and previous seasons; raise errors rather than silently invent data."""
    schedules = _pandas(nfl.load_schedules(SEASONS))
    teams = _pandas(nfl.load_team_stats(SEASONS))
    players = _pandas(nfl.load_player_stats(SEASONS))
    return {'schedules': schedules, 'teams': teams, 'players': players,
            'fetched_at': datetime.now(timezone.utc).isoformat()}

def weekly_team_summary(teams, season=2026):
    """Season-to-date aggregates from completed weeks; raw stats, not predictions."""
    data = teams.loc[teams['season'].eq(season)].copy()
    if data.empty: return data
    if 'season_type' in data: data = data[data['season_type'].eq('REG')]
    # game-level nflverse stats are generally team/week rows
    keys = [k for k in ['team'] if k in data.columns]
    if not keys: return pd.DataFrame()
    count = data.groupby('team').size().rename('Games').reset_index()
    metrics = [x for x in ['passing_yards','rushing_yards','passing_tds','rushing_tds','completions','attempts','carries'] if x in data]
    if metrics:
        averages = data.groupby('team')[metrics].mean().round(1).reset_index()
        return count.merge(averages, on='team').sort_values('team')
    return count.sort_values('team')

def upcoming_games(schedules, season=2026):
    games = schedules.loc[schedules['season'].eq(season)].copy()
    if games.empty: return games
    if 'game_type' in games: games = games[games['game_type'].eq('REG')]
    if 'gameday' not in games: return pd.DataFrame()
    games['gameday'] = pd.to_datetime(games['gameday'], errors='coerce')
    today = pd.Timestamp.now().normalize()
    games = games.loc[games['gameday'].ge(today)].sort_values('gameday')
    cols = [x for x in ['week','gameday','away_team','home_team','away_score','home_score','spread_line','total_line'] if x in games]
    return games[cols].head(32).reset_index(drop=True)

def player_leaderboard(players, season=2026, stat='passing_yards'):
    data = players.loc[players['season'].eq(season)].copy()
    if 'season_type' in data: data = data[data['season_type'].eq('REG')]
    if data.empty or stat not in data: return pd.DataFrame()
    keys = [k for k in ['player_id','player_name','recent_team','position'] if k in data]
    if 'player_id' not in keys: keys = [k for k in ['player_name','recent_team','position'] if k in data]
    if not keys: return pd.DataFrame()
    grouped = data.groupby(keys, dropna=False)[stat].sum().reset_index()
    return grouped.sort_values(stat, ascending=False).head(30).reset_index(drop=True)
