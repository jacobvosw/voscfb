"""VosCFB rating model settings and season-to-season priors.

The browser (site/template.html) runs the in-season solve. This module mirrors it
in Python to compute each season's preseason prior from the previous season's
final ratings. Settings were chosen by walk-forward backtest against closing
spreads for 2025 and 2026 (see README).
"""
from collections import defaultdict

HFA_PTS = 3.0       # home-field advantage, points
K = 0.5             # shrinkage toward the prior/average, in games
PRIOR_W = 3.0       # weight of last season's ratings, in games (full-season windows only)
PRIOR_DECAY = 0.9   # share of last season's rating carried forward
ST_W = 1.5          # weight of special teams in the rating
SCALE = 1.1         # efficiency-to-points calibration
MOV_W = 0.4         # share of the rating from adjusted scoring margin
MOV_CAP = 28        # cap on a single game's margin
GT = {2: 38, 3: 28, 4: 22}   # garbage time: lead larger than this in that quarter
EPA_CAP = 4.5       # cap on a single play's EPA


def st(r):
    return r[14] + r[15] + (r[18] if len(r) > 18 else 0)


def rate(rows, prior=None, iters=40):
    """Opponent-adjusted (aO, aD, aS, mov) per team, matching the site's solve."""
    T = defaultdict(list)
    for r in rows:
        T[r[1]].append(r)
    totP = sum(r[8] for r in rows); totE = sum(r[9] for r in rows)
    mu = totE / totP; P = totP / len(rows)
    hfa = HFA_PTS / (2 * P)
    aO = {k: 0.0 for k in T}; aD = dict(aO); aS = dict(aO); M = dict(aO)
    pw = PRIOR_W if prior else 0.0
    pr = lambda k, i: (prior.get(k, (0, 0, 0, 0))[i] if prior else 0.0)
    for _ in range(iters):
        nO, nD, nS, nM = {}, {}, {}, {}
        for k, games in T.items():
            so = sd = ss = sm = 0.0; nm = 0
            for r in games:
                loc = 1 if r[3] == 'H' else -1 if r[3] == 'A' else 0
                so += (r[9] / r[8] - mu - loc * hfa) - aD.get(r[2], 0)
                sd += (r[12] / r[11] - mu + loc * hfa) - aO.get(r[2], 0)
                ss += st(r) + aS.get(r[2], 0)
                if r[6] is not None and r[7] is not None:
                    sm += max(-MOV_CAP, min(MOV_CAP, r[6] - r[7])) - loc * HFA_PTS + M.get(r[2], 0); nm += 1
            d = len(games) + K + pw
            nO[k] = (so + pw * pr(k, 0)) / d
            nD[k] = (sd + pw * pr(k, 1)) / d
            nS[k] = (ss + pw * pr(k, 2)) / d
            nM[k] = (sm + pw * pr(k, 3)) / (nm + K + pw)
        aO, aD, aS, M = nO, nD, nS, nM
    return {k: (aO[k], aD[k], aS[k], M[k]) for k in T}


def prior_from(prev_rows):
    """Last season's final ratings, decayed, as next season's prior."""
    R = rate(prev_rows)
    return {k: [round(PRIOR_DECAY * x, 4) for x in v] for k, v in R.items()}
