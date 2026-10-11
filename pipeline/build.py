#!/usr/bin/env python3
"""VosCFB data pipeline.

For each season: download the cfbfastR play-by-play (.rds) from sportsdataverse,
compute per-team-game offense/defense EPA, kicking-game grades, and tempo,
then write site/index.html with the data inlined.

Usage:  python pipeline/build.py 2025 2026
        python pipeline/build.py 2026 --cache   (reuse a downloaded .rds)
"""
import sys, os, json, re, io, contextlib, urllib.request
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from rds import read_rds
import kicks
import kicking
import model

SRC = "https://github.com/sportsdataverse/cfbfastR-data/raw/main/data/rds/pbp_players_pos_{y}.rds"
CACHE = os.path.join(ROOT, "data", "cache")

COLS = ['season', 'week', 'game_id', 'start_date', 'home', 'away', 'neutral_site', 'conference_game', 'season_type',
        'home_team_division', 'away_team_division', 'home_team_conference', 'away_team_conference',
        'pos_team', 'def_pos_team', 'pos_team_score', 'def_pos_team_score', 'EPA', 'success', 'rush', 'pass',
        'play_type', 'play_text', 'yards_gained', 'down', 'distance', 'yards_to_goal', 'ep_before', 'period',
        'clock_minutes', 'clock_seconds', 'half', 'game_play_number', 'id_play']


def fetch(y, cache):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"pbp_{y}.rds")
    if not (cache and os.path.exists(p)):
        print(f"downloading {y} play-by-play ...", flush=True)
        urllib.request.urlretrieve(SRC.format(y=y), p)
    df = read_rds(p)
    return df[[c for c in COLS if c in df.columns]].copy()


def pace_table(a):
    """Seconds per offensive snap within drives (drive start to next drive start)."""
    a = a.assign(scrim=((a.rush == 1) | (a['pass'] == 1)), t=a.clock_minutes.fillna(0) * 60 + a.clock_seconds.fillna(0))
    res = {}
    for gid, g in a.groupby('game_id', sort=False):
        pos, per, t, sc = g.pos_team.values, g.period.values, g.t.values, g.scrim.values
        starts = [0] + [i for i in range(1, len(g)) if pos[i] != pos[i - 1] and isinstance(pos[i], str)] + [len(g)]
        drives = []
        for s, e in zip(starts[:-1], starts[1:]):
            idx = [i for i in range(s, e) if sc[i]]
            drives.append((pos[s], per[idx[0]], t[idx[0]], len(idx)) if idx else None)
        for d, n in zip(drives, drives[1:]):
            if d is None or n is None or d[1] != n[1]: continue
            dur = d[2] - n[2]; k = d[3]
            if dur <= 0 or dur > 900 or dur / k > 80 or dur / k < 5: continue
            r = res.setdefault((int(gid), d[0]), [0, 0]); r[0] += int(dur); r[1] += k
    return res


def season(y, cache):
    d = fetch(y, cache)
    d['row_order'] = range(len(d))
    d['text'] = d.play_text.astype(object).fillna('').astype(str)

    # kicking-game grades (all games incl. FCS, baselines need the volume)
    kv, info = kicks.grade(d[['season', 'week', 'game_id', 'home', 'away', 'pos_team', 'def_pos_team', 'pos_team_score',
                              'def_pos_team_score', 'half', 'period', 'clock_minutes', 'clock_seconds', 'play_type',
                              'play_text', 'down', 'distance', 'yards_to_goal', 'ep_before', 'row_order']].copy())
    print(y, 'kicks', info, flush=True)
    tg = kicks.team_games(kv).pivot_table(index=['game_id', 'team'], columns='unit', values='net').fillna(0)
    pk, pkinfo = kicking.team_game_values(d[['game_id', 'home', 'away', 'pos_team', 'pos_team_score', 'def_pos_team_score',
                                            'play_type', 'play_text', 'period', 'row_order']])
    print(y, 'place kicking', pkinfo, flush=True)

    # FBS-vs-FBS games only for the ratings
    f = d[(d.home_team_division == 'fbs') & (d.away_team_division == 'fbs')].sort_values('row_order')
    games = f.groupby('game_id').agg(date=('start_date', 'first'), home=('home', 'first'), away=('away', 'first'),
                                     neutral=('neutral_site', 'first'), confg=('conference_game', 'first'),
                                     stype=('season_type', 'first'), hconf=('home_team_conference', 'first'),
                                     aconf=('away_team_conference', 'first'))
    hp = f.pos_team == f.home
    f = f.assign(hs=np.where(hp, f.pos_team_score, f.def_pos_team_score), as_=np.where(hp, f.def_pos_team_score, f.pos_team_score))
    games = games.join(f.groupby('game_id')[['hs', 'as_']].max())
    m = ((f.rush == 1) | (f['pass'] == 1)) & f.EPA.notna() & f.pos_team.notna()
    # drop garbage time (by quarter and lead) and cap single-play EPA
    lead = (f.pos_team_score - f.def_pos_team_score).abs(); q = f.period.fillna(1)
    gt = np.zeros(len(f), dtype=bool)
    for qq, th in model.GT.items():
        gt |= ((q >= qq) if qq == 4 else (q == qq)).values & (lead > th).values
    f = f.assign(epac=f.EPA.clip(-model.EPA_CAP, model.EPA_CAP))
    off = f[m & ~gt].groupby(['game_id', 'pos_team']).agg(n=('epac', 'size'), epa=('epac', 'sum'), succ=('success', 'sum'))
    pace = pace_table(f)
    conf = {}
    for r in games.itertuples(): conf[r.home] = str(r.hconf); conf[r.away] = str(r.aconf)

    rows, last = [], ''
    for gid, g in games.iterrows():
        date = (pd.Timestamp(g.date) - pd.Timedelta(hours=6)).strftime('%Y-%m-%d')
        for team, opp, loc in ((g.home, g.away, 'N' if g.neutral else 'H'), (g.away, g.home, 'N' if g.neutral else 'A')):
            if (gid, team) not in off.index or (gid, opp) not in off.index: continue
            o, de = off.loc[(gid, team)], off.loc[(gid, opp)]
            st = tg.loc[(gid, team)] if (gid, team) in tg.index else None
            tp, op = (g.hs, g.as_) if team == g.home else (g.as_, g.hs)
            pc = pace.get((int(gid), team), [0, 0])
            rows.append([date, team, opp, loc, int(bool(g.confg)), 1 if g.stype == 'postseason' else 0,
                         int(tp) if pd.notna(tp) else None, int(op) if pd.notna(op) else None,
                         int(o.n), round(float(o.epa), 2), int(o.succ), int(de.n), round(float(de.epa), 2), int(de.succ),
                         round(float(st['KO']), 2) if st is not None else 0.0, round(float(st['P']), 2) if st is not None else 0.0,
                         pc[0], pc[1], round(float(pk.get((gid, team), 0.0)), 2)])
            last = max(last, date)
    teams = sorted({r[1] for r in rows})
    print(y, len(rows), 'team-games,', len(teams), 'teams, through', last, flush=True)
    return {'teams': {t: conf.get(t, '—') for t in teams}, 'games': rows}, last


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    cache = '--cache' in sys.argv
    years = args or ['2026']
    ddir = os.path.join(ROOT, 'site', 'data'); os.makedirs(ddir, exist_ok=True)
    for y in years:
        data, last = season(int(y), cache)
        json.dump({'thru': last, **data}, open(os.path.join(ddir, f'{y}.json'), 'w'), separators=(',', ':'))
    # manifest of every season on disk + overall data-through date
    seasons = sorted(f[:-5] for f in os.listdir(ddir) if re.fullmatch(r'\d{4}\.json', f))
    # preseason prior for each season = last season's final ratings, decayed
    for y in seasons:
        prev = str(int(y) - 1)
        fp = os.path.join(ddir, f'{y}.json'); d = json.load(open(fp))
        d['prior'] = model.prior_from(json.load(open(os.path.join(ddir, f'{prev}.json')))['games']) if prev in seasons else {}
        json.dump(d, open(fp, 'w'), separators=(',', ':'))
    thru = max(json.load(open(os.path.join(ddir, f'{y}.json')))['thru'] for y in seasons)
    current = seasons[-1]
    cur = json.load(open(os.path.join(ddir, f'{current}.json')))
    tal = open(os.path.join(ROOT, 'data', 'talent.json')).read()
    tpl = open(os.path.join(ROOT, 'site', 'template.html')).read()
    pretty = pd.Timestamp(thru).strftime('%b %-d, %Y')
    html = (tpl.replace('__DATA__', json.dumps({current: cur}, separators=(',', ':')))
               .replace('__SEASONS__', json.dumps(seasons)).replace('__TAL__', tal))
    open(os.path.join(ROOT, 'site', 'index.html'), 'w').write(html)
    print('seasons', seasons, '| wrote site/index.html, data through', pretty)


if __name__ == '__main__':
    main()
