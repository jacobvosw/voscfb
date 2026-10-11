# VosCFB

Opponent-adjusted college football efficiency ratings built from ESPN play-by-play.

Live at https://voscfb.com

## How it works

- `pipeline/build.py` downloads the season's play-by-play from
  [sportsdataverse/cfbfastR-data](https://github.com/sportsdataverse/cfbfastR-data),
  computes per-team-game offense/defense EPA, kicking-game grades (`pipeline/kicks.py`)
  and tempo, and writes per-season `site/data/<year>.json` files plus `site/index.html` with the data inlined.
- All ratings math (opponent adjustment, date windows, talent fits) runs in the browser
  in `site/template.html`.
- `data/talent.json` holds the by-side recruiting composites from 247Sports rosters.
  It is static; refresh it once per offseason.
- `.github/workflows/refresh.yml` rebuilds the current season every Sunday and Tuesday
  morning and deploys to GitHub Pages.

## Run locally

```
pip install numpy pandas
python pipeline/build.py 2019 2020 2021 2022 2023 2024 2025 2026   # downloads ~450 MB of source data
python pipeline/build.py 2026 --cache     # reuse data/cache/
open site/index.html
```

## One-time setup

1. Repo Settings → Pages → Source: **GitHub Actions**.
2. Settings → Actions → General → Workflow permissions: **Read and write**.
3. DNS at the registrar: `A` records for the apex to GitHub Pages
   (185.199.108.153, .109.153, .110.153, .111.153) and `CNAME www → <user>.github.io`.
   Then Settings → Pages → Custom domain `voscfb.com`, Enforce HTTPS.
4. Actions tab → Refresh ratings → Run workflow, to confirm the first deploy.

## Model and accuracy

Settings live in `pipeline/model.py` (mirrored in `site/template.html`): 3-point home field,
garbage-time plays removed and single-play EPA capped at ±4.5, last season's ratings as a
3-game prior (90% carryover, plus 2.5 margin points per point of roster talent above FBS average) on full-season views, special teams weighted 1.5×, and the
Rating is 60% adjusted efficiency (×1.1) + 40% opponent-adjusted scoring margin (capped at 28).

Chosen by walk-forward backtest against closing spreads: each week rated only from earlier
games, predictions compared with the line and the final score.

| | Vegas avg miss | Model avg miss | Gap |
|---|---|---|---|
| 2025 (659 games, mid-Sep on) | 11.8 | 12.3 | +0.55 |
| 2026 through Oct 10 (225 games) | 11.3 | 12.1 | +0.84 |

Acceptance standard: model average miss within 2–3 points of the line's.
