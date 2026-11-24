import type { ReactElement } from "react";
import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider } from "@/lib/theme";

export function renderWithProviders(
  ui: ReactElement,
  { route = "/" }: { route?: string } = {},
) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>
      </ThemeProvider>
    </QueryClientProvider>,
  );
}

export const mockStatus = {
  generated: "2026-11-13T10:00:00+00:00",
  repo: {
    branch: "main",
    sha: "d89915c0000000000000000000000000000000000",
    short_sha: "d89915c",
    dirty: true,
  },
  dataset: {
    available: true,
    rows: 2979,
    columns: 42,
    start: "2016-01-01",
    end: "2024-12-31",
    missingness: {
      aod_observed: { missing: 2840, fraction: 0.9533 },
      merra_observed: { missing: 2807, fraction: 0.9423 },
      station_met_observed: { missing: 0, fraction: 0 },
    },
  },
  coverage: {
    available: true,
    rows: 2979,
    sources: {
      station_met: { observed: 2979, fraction: 1 },
      merra: { observed: 172, fraction: 0.0577 },
      aod: { observed: 139, fraction: 0.0467 },
    },
    all_sources_observed: { observed: 139, fraction: 0.0467 },
  },
  experiments: { available: true, generated: "2026-11-12" },
};

export const mockExperiments = {
  available: true,
  generated: "2026-11-12",
  configuration: "phase6_fusion",
  station: { name: "Faridabad", lat: 28.4089, lon: 77.3178 },
  coverage: { n_rows: 2979, n_test: 33 },
  split: { n_train: 166, n_test: 33 },
  metrics: {
    RandomForest: { MAE: 17.821, RMSE: 22.4, R2: 0.12, Mean_Bias: -0.4 },
    XGBoost: { MAE: 16.941, RMSE: 21.9, R2: 0.15, Mean_Bias: 0.2 },
    Ensemble: { MAE: 17.298, RMSE: 22.1, R2: 0.017, Mean_Bias: 0.1 },
  },
  baselines: {
    Ground_history_only: { MAE: 16.617, RMSE: 21.0, R2: 0.18 },
    AOD_only: { MAE: 19.26, RMSE: 24.1, R2: -0.1 },
    Persistence: { MAE: 19.918, RMSE: 25.0, R2: -0.2 },
  },
  features_used: ["PM25_Lag_1", "month_sin", "AOD"],
  features_unavailable: ["PBLH"],
  hyperparameters: { RandomForest: { n_estimators: 300 } },
};

export const mockSeries = {
  available: true,
  count: 4,
  rows: [
    { date: "2024-06-15", pm25: 98.2, aod: null, t2m: 39.1, rh: 42, ws: 2.1, sources: { station_met: true, merra: false, aod: false } },
    { date: "2024-06-16", pm25: 110.4, aod: 0.71, t2m: 38.2, rh: 45, ws: 1.8, sources: { station_met: true, merra: true, aod: true } },
    { date: "2024-06-17", pm25: null, aod: null, t2m: null, rh: null, ws: null, sources: { station_met: false, merra: false, aod: false } },
    { date: "2024-06-18", pm25: 120.1, aod: 0.65, t2m: 37.7, rh: 50, ws: 1.4, sources: { station_met: true, merra: true, aod: true } },
  ],
};

export const mockManifests = {
  available: true,
  count: 1,
  manifests: [
    {
      path: "results/runs/F-fusion-1/manifest.json",
      run_id: "F-fusion-1",
      git_sha: "d89915c",
      generated: "2026-11-12T10:00:00Z",
      track: "B",
    },
  ],
};

export const mockMatrix = {
  available: true,
  generated: "2026-11-22T10:00:00Z",
  run_id: "matrix-1",
  configs: {},
  metrics: [
    {
      track: "A",
      config: "M0",
      model: "persistence",
      n: 120,
      MAE: 12.0,
      RMSE: 18.0,
      MedianAE: 9.0,
      R2: 0.1,
      Mean_Bias: 0.0,
      folds: 3,
    },
    {
      track: "B",
      config: "M4",
      model: "ridge",
      n: 60,
      MAE: 10.0,
      RMSE: 15.0,
      MedianAE: 8.0,
      R2: 0.2,
      Mean_Bias: 0.5,
      folds: 2,
    },
  ],
  common: [],
  by_season: [],
  by_year: [],
  ablations: [
    {
      track: "A",
      config: "M1",
      reference: "M0",
      delta_MAE: -0.65,
      n: 2615,
      ci_low: -1.14,
      ci_high: -0.17,
      year_ci_low: -1.2,
      year_ci_high: 0.06,
      p_holm: 0.008,
      significant: true,
      reported: true,
    },
    {
      track: "B",
      config: "M4",
      reference: "M1",
      delta_MAE: 4.7,
      n: 49,
      ci_low: 0.3,
      ci_high: 10.41,
      year_ci_low: 0.3,
      year_ci_high: 10.41,
      p_holm: 0.34,
      significant: false,
      reported: true,
    },
  ],
};

export const mockAblations = {
  available: true,
  rows: [
    {
      track: "B",
      config: "M4",
      reference: "M1",
      delta_MAE: 4.7,
      n: 49,
      ci_low: 0.3,
      ci_high: 10.41,
      p_holm: 0.34,
      significant: false,
      reported: true,
    },
  ],
  common: [],
};

export const mockSufficiency = {
  available: true,
  rows: [
    {
      season: "winter",
      source: "aod",
      n_season_rows: 742,
      n_observed: 41,
      observed_fraction: 0.055,
      status: "WARNING",
      reason: "41 observed rows supports a comparison but is below stability",
    },
    {
      season: "post_monsoon",
      source: "merra",
      n_season_rows: 479,
      n_observed: 0,
      observed_fraction: 0,
      status: "NOT_EVALUABLE",
      reason: "the source is never observed in this season",
    },
  ],
  criteria: { min_rows_for_inference: 30, min_rows_for_sufficient: 90 },
  counts: { SUFFICIENT: 0, WARNING: 1, INSUFFICIENT: 0, NOT_EVALUABLE: 1 },
};

export const mockAlignment = {
  available: true,
  coverage: [
    {
      source: "aod",
      observed_rows: 139,
      missing_rows: 2840,
      observed_fraction: 0.0467,
      first_observed: "2024-01-07",
      last_observed: "2024-06-18",
    },
  ],
  missingness: [
    {
      source: "aod",
      n_observed: 139,
      n_unobserved: 2840,
      target_mean_when_observed: 100.0,
      target_mean_when_unobserved: 93.0,
      target_mean_difference: 7.0,
    },
  ],
  report: { rows: 2979, validation_passed: true },
};