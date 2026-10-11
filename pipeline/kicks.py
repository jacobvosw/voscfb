"""Special-teams grades: kickoff & punt units (both directions) in points vs FBS average.

Value of a kick (kicking team's perspective):
  - if points are scored before the next scrimmage snap: net points (kicking - receiving)
  - else: +EP(ball) if kicking team has it, -EP(ball) if receiving team has it,
    using a league-wide 1st-and-10 EP curve by yards-to-goal.
Grade = value - league baseline (kickoffs: by kick type; punts: by line of scrimmage).
"""
import re, sys
import numpy as np, pandas as pd

NON_SCRIM = {'Timeout', 'End Period', 'End of Half', 'End of Game', 'End of Regulation',
             'Kickoff', 'Kickoff Return (Offense)', 'Kickoff Return Touchdown', 'Coin Toss',
             'Extra Point Good', 'Extra Point Missed', 'Two Point Pass', 'Two Point Rush',
             'Defensive 2pt Conversion', 'placeholder', 'Uncategorized'}

def prep(df):
    d = df.sort_values('row_order').reset_index(drop=True)
    d['text'] = d.play_text.astype(object).fillna('').astype(str)
    d['pt'] = d.play_type.astype(object).fillna('').astype(str)
    hp = d.pos_team == d.home
    d['hs'] = np.where(hp, d.pos_team_score, d.def_pos_team_score)
    d['as_'] = np.where(hp, d.def_pos_team_score, d.pos_team_score)
    d['is_ko'] = d.text.str.contains(r'\bkickoff\b', case=False) & ~d.text.str.contains(r'kickoff return \(offense\)', case=False)
    d['is_punt'] = d.text.str.contains(r'\bpunt\b', case=False) & ~d.is_ko & ~d.pt.str.contains('Field Goal')
    d['is_scrim'] = ~d.pt.isin(NON_SCRIM) & ~d.is_ko & ~d.is_punt & d.yards_to_goal.notna() & d.pos_team.notna()
    return d

def ep_curve(d):
    m = d.is_scrim & (d.down == 1) & ((d.distance == 10) | (d.distance >= d.yards_to_goal)) & d.ep_before.notna()
    s = d[m].groupby(d.yards_to_goal.round())['ep_before'].mean().reindex(range(1, 100)).interpolate(limit_direction='both')
    s = s.rolling(5, center=True, min_periods=1).mean()
    return lambda ytg: float(s.loc[int(min(99, max(1, round(ytg))))])

def kicker_team_votes(d):
    """(game, kicker-name) -> team, from punts/FG/holder mentions and plain kickoffs."""
    votes = {}
    def add(g, name, team):
        if name and isinstance(team, str): votes.setdefault((g, name), []).append(team)
    for r in d[d.is_punt | d.pt.str.contains('Field Goal')].itertuples():
        for nm in re.findall(r'#\d+ ([A-Z][\w.\'\- ]+?) (?:punt|field goal)', r.text): add(r.game_id, nm, r.pos_team)
        for nm in re.findall(r'H: #\d+ ([A-Z][\w.\'\- ]+?),', r.text): add(r.game_id, nm, r.pos_team)
    for r in d[d.is_ko & d.pt.eq('Kickoff')].itertuples():
        m = re.search(r'#\d+ ([A-Z][\w.\'\- ]+?) kickoff', r.text)
        if m: add(r.game_id, m.group(1), r.def_pos_team)
    return {k: max(set(v), key=v.count) for k, v in votes.items()}

def kick_values(d, EP):
    votes = kicker_team_votes(d)
    out = []
    games = d.groupby('game_id', sort=False)
    for g, gd in games:
        gd = gd.reset_index(drop=True)
        teams = {gd.home.iloc[0], gd.away.iloc[0]}
        for i in np.flatnonzero((gd.is_ko | gd.is_punt).values):
            r = gd.iloc[i]
            if r.is_punt:
                kteam = r.pos_team
            else:
                m = re.search(r'#\d+ ([A-Z][\w.\'\- ]+?) kickoff', r.text)
                kteam = votes.get((g, m.group(1))) if m else None
                if kteam is None:
                    kteam = r.def_pos_team if r.pt == 'Kickoff' else None
            if kteam not in teams: continue
            rteam = (teams - {kteam}).pop()
            prev = gd.iloc[i-1] if i > 0 else r
            sc = lambda row, t: row.hs if t == row.home else row.as_
            secs = lambda row: (row.clock_minutes or 0) * 60 + (row.clock_seconds or 0)
            ksec = secs(r)
            # touchdown on the kick itself (or on an immediately-following return-TD row)
            td_row = None
            TD = r'touchdown|for a TD|\bTD\b|Yd (?:Punt|Kickoff|Kick) Return'
            if re.search(TD, r.text, re.I): td_row = r
            elif i + 1 < len(gd) and re.search(TD, gd.pt.iat[i+1] + ' ' + gd.text.iat[i+1], re.I) and gd.period.iat[i+1] == r.period \
                    and abs(secs(gd.iloc[i+1]) - ksec) <= 20 and not gd.is_scrim.iat[i+1]:
                td_row = gd.iloc[i+1]
            if re.search(r'SAFETY', r.text): continue
            if td_row is not None:
                try:
                    k_scored = (sc(td_row, kteam) - sc(prev, kteam)) > 0 and (sc(td_row, rteam) - sc(prev, rteam)) <= 0
                except Exception:
                    k_scored = False
                val = 7.0 if k_scored else -7.0
            else:
                # next real snap in the same half, ignoring out-of-order rows stamped later than the kick
                nxt = None
                kept = bool(re.search(r'fumble|muff|recovered|onside|NO PLAY|re-kick|rekick', r.text, re.I))
                for j in range(i + 1, min(len(gd), i + 15)):
                    c = gd.iloc[j]
                    if c.half != r.half: break
                    if c.is_ko or c.is_punt: break
                    if not c.is_scrim: continue
                    if c.period < r.period: continue
                    if c.pos_team == kteam and not kept: continue
                    if c.period == r.period and secs(c) > ksec + 2: continue
                    nxt = c; break
                if nxt is None:
                    # fallback: rows can be out of order -- take the snap closest in game time after the kick
                    same = gd[(gd.index != i) & gd.is_scrim & (gd.period == r.period)]
                    tt = same.clock_minutes.fillna(0) * 60 + same.clock_seconds.fillna(0)
                    ok = ((same.index > i) & (tt <= ksec + 1)) | ((same.index < i) & (tt < ksec))
                    cand = same[ok & ((same.pos_team != kteam) | kept)]
                    if len(cand):
                        nxt = cand.loc[(cand.clock_minutes.fillna(0) * 60 + cand.clock_seconds.fillna(0)).idxmax()]
                if nxt is not None:
                    val = EP(nxt.yards_to_goal) * (1 if nxt.pos_team == kteam else -1)
                else:
                    # last resort for kicks recovered by the kicking team with no later snap recorded
                    m = re.search(r'fumbled by ' + re.escape(rteam) + r'.*recovered by (\w+).*? at ([A-Z]+)(\d+)', r.text)
                    land = re.search(r'to the ([A-Z]+)\d+', r.text)
                    if not (m and land and not rteam.upper().startswith(m.group(1)[:3])): continue
                    n = int(m.group(3))
                    ytg = 100 - n if m.group(2) == land.group(1) and r.is_ko else n
                    val = EP(ytg)
            dist = re.search(r'(?:kickoff|punt) (\d+) yards?', r.text, re.I)
            los = r.yards_to_goal
            if r.is_punt and (pd.isna(los) or los <= 0):
                lm = re.search(r'punt (\d+) yards? to the [A-Z]+(\d+)', r.text)
                if not lm: continue
                los = min(99, int(lm.group(1)) + int(lm.group(2)))
            onside = r.is_ko and (bool(re.search('onside', r.text, re.I)) or (dist is not None and int(dist.group(1)) < 25))
            out.append(dict(game_id=g, week=r.week, season=r.season, home=r.home, away=r.away,
                            unit='KO' if r.is_ko else 'P', onside=onside, kteam=kteam, rteam=rteam,
                            los=los, val=val, period=r.period,
                            clock=f"{int(r.clock_minutes or 0)}:{int(r.clock_seconds or 0):02d}", text=r.text))
    return pd.DataFrame(out)

def grade(df):
    d = prep(df); EP = ep_curve(d)
    k = kick_values(d, EP)
    # baselines
    ko = k.unit == 'KO'
    b5 = (k.los / 5).round() * 5
    base_ko = k[ko & ~k.onside].groupby(b5)['val'].agg(['mean', 'size'])
    glob_ko = k[ko & ~k.onside & (k.los.round() == 65)].val.mean()
    def bko(los):
        los = round(los / 5) * 5
        return base_ko.loc[los, 'mean'] if los in base_ko.index and base_ko.loc[los, 'size'] >= 50 else glob_ko
    onside_base = k[ko & k.onside].val.mean()
    p = k.unit == 'P'
    pb = k[p].groupby((k.los / 5).round() * 5)['val'].mean().rolling(3, center=True, min_periods=1).mean()
    k['base'] = np.where(ko, np.where(k.onside, onside_base, k.los.map(bko)),
                         ((k.los / 5).round() * 5).map(pb))
    k['grade'] = k.val - k.base
    return k, dict(glob_ko=glob_ko, onside_base=onside_base, n=len(k))




def team_games(k):
    rows = []
    for (g, unit), gd in k.groupby(['game_id', 'unit']):
        h, a = gd.home.iloc[0], gd.away.iloc[0]
        for team in (h, a):
            cover = gd.loc[gd.kteam == team, 'grade'].sum()
            ret = -gd.loc[gd.rteam == team, 'grade'].sum()
            rows.append(dict(game_id=g, unit=unit, team=team, opp=a if team == h else h,
                             week=gd.week.iloc[0], cover=cover, ret=ret, net=cover + ret,
                             n_cover=(gd.kteam == team).sum(), n_ret=(gd.rteam == team).sum()))
    return pd.DataFrame(rows)

