import { useQuery } from "@tanstack/react-query";
import { api } from "./api";

export const queryKeys = {
  status: ["status"] as const,
  experiments: ["experiments"] as const,
  coverage: ["coverage"] as const,
  dataset: ["dataset"] as const,
  manifests: ["manifests"] as const,
  experimentMatrix: ["experiment-matrix"] as const,
  ablations: ["ablations"] as const,
  seasonalSufficiency: ["seasonal-sufficiency"] as const,
  alignment: ["alignment"] as const,
  forecasts: ["forecasts"] as const,
  series: (start: string, end: string) => ["series", start, end] as const,
};

export function useStatus() {
  return useQuery({
    queryKey: queryKeys.status,
    queryFn: ({ signal }) => api.status(signal),
    staleTime: 30_000,
  });
}

export function useExperiments() {
  return useQuery({
    queryKey: queryKeys.experiments,
    queryFn: ({ signal }) => api.experiments(signal),
    staleTime: 60_000,
  });
}

export function useCoverage() {
  return useQuery({
    queryKey: queryKeys.coverage,
    queryFn: ({ signal }) => api.coverage(signal),
    staleTime: 60_000,
  });
}

export function useDataset() {
  return useQuery({
    queryKey: queryKeys.dataset,
    queryFn: ({ signal }) => api.dataset(signal),
    staleTime: 60_000,
  });
}

export function useManifests() {
  return useQuery({
    queryKey: queryKeys.manifests,
    queryFn: ({ signal }) => api.manifests(signal),
    staleTime: 60_000,
  });
}

export function useExperimentMatrix() {
  return useQuery({
    queryKey: queryKeys.experimentMatrix,
    queryFn: ({ signal }) => api.experimentMatrix(signal),
    staleTime: 60_000,
  });
}

export function useAblations() {
  return useQuery({
    queryKey: queryKeys.ablations,
    queryFn: ({ signal }) => api.ablations(signal),
    staleTime: 60_000,
  });
}

export function useSeasonalSufficiency() {
  return useQuery({
    queryKey: queryKeys.seasonalSufficiency,
    queryFn: ({ signal }) => api.seasonalSufficiency(signal),
    staleTime: 60_000,
  });
}

export function useAlignment() {
  return useQuery({
    queryKey: queryKeys.alignment,
    queryFn: ({ signal }) => api.alignment(signal),
    staleTime: 60_000,
  });
}

export function useForecasts() {
  return useQuery({
    queryKey: queryKeys.forecasts,
    queryFn: ({ signal }) => api.forecasts(signal),
    staleTime: 60_000,
  });
}

export function useSeries(start: string, end: string, enabled = true) {
  return useQuery({
    queryKey: queryKeys.series(start, end),
    queryFn: ({ signal }) => api.series(start, end, signal),
    staleTime: 60_000,
    enabled,
  });
}
