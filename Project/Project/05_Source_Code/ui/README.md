# AeroSphinx UI

Frontend for the AeroSphinx atmospheric-intelligence repository.

Stack: React 19 · TypeScript 7 (strict) · Vite 8 · Tailwind CSS v4 ·
React Router 7 · TanStack Query · Zod · ECharts · Vitest / React Testing
Library · Playwright.

Python remains the scientific source of truth. The UI never computes a
scientific value: it renders artifacts produced by the Python package.

## Run locally

Two processes are required.

```bash
# 1. Read-only artifact API (standard library only)
python api/server.py --port 8787

# 2. Frontend (proxies /api → :8787)
cd ui
npm install
npm run dev          # http://127.0.0.1:5173
```

If the API is not running, every data-driven module renders an explicit
**Unavailable** state with a Retry control — it never falls back to fabricated
numbers.

## Scripts

| Command | Purpose |
| --- | --- |
| `npm run dev` | Vite dev server on :5173 |
| `npm run build` | Type-check (`tsc -b`) then production build |
| `npm run typecheck` | Strict type check only |
| `npm test` | Vitest unit / component suite |
| `npm run test:coverage` | Coverage report |
| `npx playwright test` | End-to-end (needs both servers running) |
| `node scripts/screenshots.mjs` | Capture docs/screenshots (needs both servers) |

## Modules

`/` Command Center · `/research-lab` · `/data-observatory` ·
`/validation-matrix` · `/defect-register` · `/model-intelligence` ·
`/forecast-studio` · `/atmospheric-explorer` · `/reports` · `/settings`

## API contract (implemented)

`api/server.py` serves the repository's real artifacts:

| Endpoint | Source |
| --- | --- |
| `GET /api/health` | process liveness |
| `GET /api/status` | git + dataset + coverage summary |
| `GET /api/dataset` | `data/processed/fused_dataset.csv` |
| `GET /api/experiments` | `results/legacy_chapter/metrics.json` |
| `GET /api/coverage` | `data/processed/fused_dataset.csv` |
| `GET /api/series?start&end` | `data/processed/fused_dataset.csv` |
| `GET /api/manifests` | `results/**/manifest.json` |

Missing artifacts return `{"available": false, "reason": "..."}`. Blank or
`NaN` CSV cells are emitted as `null`, never as `0`.

## Backend capabilities still missing

These are rendered as unavailable states rather than faked:

- `POST /api/forecast` — inference for Forecast Studio. Requires
  `{ station, issue_time, horizon, model }` and must return the value, a
  **calibrated** interval (or none), and the availability record used.
- `GET /api/manifests/{id}` — full manifest body (inputs, hashes, config).
- A documented gridded product for the spatial map in Atmospheric Explorer.
- Diagnostic artifacts (residuals, per-fold errors, seasonal breakdown) for
  Model Intelligence, if these are to be shown.

## Scientific rules honoured

- No forward-filling; missing stays missing (gaps, not zeros).
- No future observations as forecast inputs; eligibility is enforced.
- Track B is labelled retrospective, never operational.
- No uncalibrated intervals are rendered.
- Negative results (e.g. fusion does not beat ground-only) are shown, not hidden.
