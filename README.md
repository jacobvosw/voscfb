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
