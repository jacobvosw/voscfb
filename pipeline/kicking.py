"""Place-kicking value: field goals and extra points, in points over expected.

Field goal:  3*made - 3*P(make | distance), with P fitted league-wide each season
             (logistic in distance). A blocked or missed kick returned for a
             touchdown costs an extra 7.
Extra point: 1*made - league PAT make rate.

Each team-game gets  own kicking value - opponent kicking value,  so a team is
credited for its kicker and for its field-goal defense (blocks, pressure, and
the opponent's misses), the same way the punt and kickoff grades work.
"""
import re
import numpy as np
import pandas as pd

DIST = [re.compile(r'(\d+)\s*(?:yd|yard|yds)\.?\s*(?:FG|field goal)', re.I),
        re.compile(r'field goal attempt from (\d+) yards?', re.I)]
XP_GOOD = re.compile(r'kick attempt good|\(\s*[^()]*\b(?:Kick|KICK)\)', re.I)
XP_MISS = re.compile(r'kick attempt (?:failed|missed|blocked|no good)|PAT (?:MISSED|BLOCKED|FAILED)|'
                     r'Kick (?:Failed|MISSED|Blocked)|KICK (?:FAILED|MISSED|BLOCKED)', re.I)
TD_TEXT = re.compile(r'TOUCHDOWN|for a TD|\bTD\b|Yd (?:Run|pass|Pass|Return|Interception|Fumble)', re.I)


def _dist(text):
    for rx in DIST:
        m = rx.search(text)
        if m:
            v = int(m.group(1))
            if 10 <= v <= 75:
                return v
    return None


def _logistic(x, y, iters=50):
    X = np.column_stack([np.ones_like(x, dtype=float), x.astype(float)])
    b = np.array([4.0, -0.08])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(2)
        b = b + np.linalg.solve(H, X.T @ (y - p))
    return lambda d: 1 / (1 + np.exp(-(b[0] + b[1] * np.asarray(d, float))))


def team_game_values(d):
    """d: play-level frame with game_id, home, away, pos_team, pos_team_score,
    def_pos_team_score, play_type, play_text, period, row_order.
    Returns {(game_id, team): net place-kicking points} and a summary dict."""
    d = d.sort_values('row_order')
    pt = d.play_type.astype(object).fillna('').astype(str)
    tx = d.play_text.astype(object).fillna('').astype(str)

    # ---- field goals ----
    fg = d[pt.str.contains('Field Goal')].assign(pt=pt, tx=tx)
    fg = fg.drop_duplicates(['game_id', 'period', 'tx'])
    fg = fg.assign(dist=fg.tx.map(_dist), made=fg.pt.str.startswith('Field Goal Good').astype(int))
    fg = fg[fg.dist.notna() & fg.pos_team.notna()]
    P = _logistic(fg.dist.values, fg.made.values)
    fg = fg.assign(val=3 * fg.made - 3 * P(fg.dist.values)
                   - 7 * (fg.pt.str.contains('Touchdown') & (fg.made == 0)).astype(int))

    # ---- extra points (on touchdown rows) ----
    hp = d.pos_team == d.home
    hs = np.where(hp, d.pos_team_score, d.def_pos_team_score)
    as_ = np.where(hp, d.def_pos_team_score, d.pos_team_score)
    s = d.assign(hs=hs, as_=as_, tx=tx)
    s = s.assign(dh=s.groupby('game_id').hs.diff().fillna(s.hs), da=s.groupby('game_id').as_.diff().fillna(s.as_))
    td = s[s.tx.str.contains(TD_TEXT) & ~s.tx.str.contains('defensive PAT', case=False)]
    good = td.tx.str.contains(XP_GOOD)
    miss = td.tx.str.contains(XP_MISS)
    xp = td[good ^ miss].assign(made=good[good ^ miss].astype(int))
    xp = xp.assign(team=np.where(xp.dh >= 6, xp.home, np.where(xp.da >= 6, xp.away, None)))
    xp = xp[xp.team.notna()].drop_duplicates(['game_id', 'period', 'tx'])
    rate = xp.made.mean() if len(xp) else 0.97
    xp = xp.assign(val=xp.made - rate)

    own = {}
    for g, t, v in zip(fg.game_id, fg.pos_team, fg.val):
        own[(g, t)] = own.get((g, t), 0.0) + v
    for g, t, v in zip(xp.game_id, xp.team, xp.val):
        own[(g, t)] = own.get((g, t), 0.0) + v
    games = d.groupby('game_id')[['home', 'away']].first()
    net = {}
    for g, r in games.iterrows():
        h, a = own.get((g, r.home), 0.0), own.get((g, r.away), 0.0)
        net[(g, r.home)] = h - a
        net[(g, r.away)] = a - h
    info = dict(fg_att=len(fg), fg_pct=round(fg.made.mean(), 3), p30=round(float(P(30)), 3),
                p45=round(float(P(45)), 3), p55=round(float(P(55)), 3), xp_att=len(xp), xp_rate=round(rate, 3))
    return net, info
