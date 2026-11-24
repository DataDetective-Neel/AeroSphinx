import { z } from "zod";

/**
 * Typed client for the read-only AeroSphinx artifact API (api/server.py).
 *
 * Every response is validated with Zod. A field that the backend does not
 * provide is `null` or `available: false` — never a fabricated default.
 */

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function getJson<S extends z.ZodTypeAny>(
  path: string,
  schema: S,
  signal?: AbortSignal,
): Promise<z.output<S>> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { signal });
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new ApiError(
      "Cannot reach the AeroSphinx API. Start it with `python api/server.py`.",
      0,
    );
  }
  if (!res.ok) {
    throw new ApiError(`API ${path} failed with ${res.status}`, res.status);
  }
  const raw: unknown = await res.json();
  const parsed = schema.safeParse(raw);
  if (!parsed.success) {
    throw new ApiError(
      `API ${path} returned an unexpected shape: ${parsed.error.message}`,
      500,
    );
  }
  return parsed.data;
}

/* ---------------------------------- Schemas --------------------------------- */

export const HealthSchema = z.object({
  status: z.string(),
  generated: z.string(),
});

export const RepoSchema = z.object({
  branch: z.string().nullable(),
  sha: z.string().nullable(),
  short_sha: z.string().nullable(),
  dirty: z.boolean(),
});

export const DatasetSummarySchema = z.object({
  available: z.boolean(),
  reason: z.string().optional(),
  rows: z.number().optional(),
  columns: z.number().optional(),
  start: z.string().nullable().optional(),
  end: z.string().nullable().optional(),
  missingness: z
    .record(z.string(), z.object({ missing: z.number(), fraction: z.number() }))
    .optional(),
});

export const CoverageSchema = z.object({
  available: z.boolean(),
  rows: z.number().optional(),
  sources: z
    .object({
      station_met: z.object({ observed: z.number(), fraction: z.number() }),
      merra: z.object({ observed: z.number(), fraction: z.number() }),
      aod: z.object({ observed: z.number(), fraction: z.number() }),
    })
    .optional(),
  all_sources_observed: z
    .object({ observed: z.number(), fraction: z.number() })
    .optional(),
});

export const StatusSchema = z.object({
  generated: z.string(),
  repo: RepoSchema,
  dataset: DatasetSummarySchema,
  coverage: CoverageSchema,
  experiments: z.object({
    available: z.boolean(),
    generated: z.string().nullable(),
  }),
});

const MetricBlock = z.record(z.string(), z.number());

export const ExperimentsSchema = z.object({
  available: z.boolean(),
  reason: z.string().optional(),
  generated: z.string().nullable().optional(),
  configuration: z.string().nullable().optional(),
  station: z
    .object({ name: z.string(), lat: z.number(), lon: z.number() })
    .nullable()
    .optional(),
  coverage: z.record(z.string(), z.union([z.number(), z.string()])).nullable().optional(),
  split: z.record(z.string(), z.union([z.number(), z.string()])).nullable().optional(),
  metrics: z.record(z.string(), MetricBlock).nullable().optional(),
  baselines: z.record(z.string(), MetricBlock).nullable().optional(),
  features_used: z.array(z.string()).nullable().optional(),
  features_unavailable: z.array(z.string()).nullable().optional(),
  hyperparameters: z
    .record(z.string(), z.record(z.string(), z.union([z.number(), z.string()])))
    .nullable()
    .optional(),
});

export const SeriesPointSchema = z.object({
  date: z.string(),
  pm25: z.number().nullable(),
  aod: z.number().nullable(),
  t2m: z.number().nullable(),
  rh: z.number().nullable(),
  ws: z.number().nullable(),
  sources: z.object({
    station_met: z.boolean().nullable(),
    merra: z.boolean().nullable(),
    aod: z.boolean().nullable(),
  }),
});

export const SeriesSchema = z.object({
  available: z.boolean(),
  reason: z.string().optional(),
  count: z.number().optional(),
  rows: z.array(SeriesPointSchema).optional(),
});

export const ManifestsSchema = z.object({
  available: z.boolean(),
  count: z.number(),
  manifests: z.array(
    z.object({
      path: z.string(),
      run_id: z.string().nullable(),
      git_sha: z.string().nullable(),
      generated: z.string().nullable(),
      track: z.string().nullable(),
    }),
  ),
});

export const ArtifactRowSchema = z.record(
  z.string(),
  z.union([z.string(), z.number(), z.boolean(), z.null()]),
);
export type ArtifactRow = z.infer<typeof ArtifactRowSchema>;

export const ExperimentMatrixSchema = z.object({
  available: z.boolean(),
  reason: z.string().optional(),
  generated: z.string().nullable().optional(),
  run_id: z.string().nullable().optional(),
  configs: z.record(z.string(), z.unknown()).optional(),
  metrics: z.array(ArtifactRowSchema).optional(),
  common: z.array(ArtifactRowSchema).optional(),
  by_season: z.array(ArtifactRowSchema).optional(),
  by_year: z.array(ArtifactRowSchema).optional(),
  ablations: z.array(ArtifactRowSchema).optional(),
});

export const AblationsSchema = z.object({
  available: z.boolean(),
  reason: z.string().optional(),
  rows: z.array(ArtifactRowSchema).optional(),
  common: z.array(ArtifactRowSchema).optional(),
});

export const SufficiencySchema = z.object({
  available: z.boolean(),
  reason: z.string().optional(),
  rows: z.array(ArtifactRowSchema).optional(),
  criteria: z.record(z.string(), z.unknown()).nullable().optional(),
  counts: z.record(z.string(), z.number()).nullable().optional(),
  by_season: z
    .record(z.string(), z.record(z.string(), z.number()))
    .nullable()
    .optional(),
  not_evaluable: z
    .array(z.record(z.string(), z.string()))
    .nullable()
    .optional(),
});

export const ForecastProjectionSchema = z.object({
  path: z.string(),
  rows: z.array(ArtifactRowSchema),
});

export const ForecastsSchema = z.object({
  available: z.boolean(),
  reason: z.string().optional(),
  operational: ForecastProjectionSchema.optional(),
  recursive: ForecastProjectionSchema.optional(),
});

export const AlignmentSchema = z.object({
  available: z.boolean(),
  reason: z.string().optional(),
  coverage: z.array(ArtifactRowSchema).optional(),
  missingness: z.array(ArtifactRowSchema).optional(),
  report: z.record(z.string(), z.unknown()).optional(),
});

export type Status = z.infer<typeof StatusSchema>;
export type ExperimentMatrix = z.infer<typeof ExperimentMatrixSchema>;
export type Ablations = z.infer<typeof AblationsSchema>;
export type Sufficiency = z.infer<typeof SufficiencySchema>;
export type Alignment = z.infer<typeof AlignmentSchema>;
export type Experiments = z.infer<typeof ExperimentsSchema>;
export type SeriesResponse = z.infer<typeof SeriesSchema>;
export type SeriesPoint = z.infer<typeof SeriesPointSchema>;
export type Manifests = z.infer<typeof ManifestsSchema>;
export type DatasetSummary = z.infer<typeof DatasetSummarySchema>;
export type Coverage = z.infer<typeof CoverageSchema>;

/* ---------------------------------- Fetchers -------------------------------- */

export const api = {
  health: (signal?: AbortSignal) =>
    getJson("/health", HealthSchema, signal),
  status: (signal?: AbortSignal) =>
    getJson("/status", StatusSchema, signal),
  experiments: (signal?: AbortSignal) =>
    getJson("/experiments", ExperimentsSchema, signal),
  coverage: (signal?: AbortSignal) =>
    getJson("/coverage", CoverageSchema, signal),
  dataset: (signal?: AbortSignal) =>
    getJson("/dataset", DatasetSummarySchema, signal),
  series: (start: string, end: string, signal?: AbortSignal) =>
    getJson(
      `/series?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`,
      SeriesSchema,
      signal,
    ),
  manifests: (signal?: AbortSignal) =>
    getJson("/manifests", ManifestsSchema, signal),
  experimentMatrix: (signal?: AbortSignal) =>
    getJson("/experiment-matrix", ExperimentMatrixSchema, signal),
  ablations: (signal?: AbortSignal) =>
    getJson("/ablations", AblationsSchema, signal),
  seasonalSufficiency: (signal?: AbortSignal) =>
    getJson("/seasonal-sufficiency", SufficiencySchema, signal),
  alignment: (signal?: AbortSignal) =>
    getJson("/alignment", AlignmentSchema, signal),
  forecasts: (signal?: AbortSignal) =>
    getJson("/forecasts", ForecastsSchema, signal),
};