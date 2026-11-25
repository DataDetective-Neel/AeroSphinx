import { useMemo, useState } from "react";
import { LineChart } from "@/components/ui/LineChart";
import { useSeries, useStatus } from "@/lib/queries";
import {
  Card,
  EmptyState,
  ErrorState,
  LoadingBlock,
  PageHeader,
  StatusBadge,
} from "@/components/ui/primitives";
import { STATION } from "@/lib/reference";
import { useTheme } from "@/lib/theme";

const VARIABLES = [
  { key: "pm25", label: "PM2.5", unit: "µg/m³" },
  { key: "aod", label: "AOD", unit: "" },
  { key: "t2m", label: "Temperature", unit: "°C" },
  { key: "rh", label: "Humidity", unit: "%" },
  { key: "ws", label: "Wind speed", unit: "m/s" },
] as const;

type VarKey = (typeof VARIABLES)[number]["key"];

export function AtmosphericExplorer() {
  const status = useStatus();
  const { resolvedTheme } = useTheme();
  const dataset = status.data?.dataset;
  const [variable, setVariable] = useState<VarKey>("pm25");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");

  const effStart = start || dataset?.start || "";
  const effEnd = end || dataset?.end || "";

  const series = useSeries(effStart, effEnd, !!effStart && !!effEnd);

  const points = useMemo(
    () => (series.data?.rows ?? []).filter((r) => r[variable] !== null),
    [series.data, variable],
  );

  const meta = VARIABLES.find((v) => v.key === variable)!;
  const axisColor = resolvedTheme === "dark" ? "#9AAABC" : "#64748B";
  const lineColor = resolvedTheme === "dark" ? "#38BDF8" : "#0284C7";

  const option = {
    backgroundColor: "transparent",
    grid: { left: 48, right: 20, top: 24, bottom: 40 },
    tooltip: {
      trigger: "axis",
      backgroundColor: resolvedTheme === "dark" ? "#1A2633" : "#FFFFFF",
      borderColor: resolvedTheme === "dark" ? "#293747" : "#DCE3EB",
      textStyle: { color: resolvedTheme === "dark" ? "#E8EEF5" : "#172334" },
    },
    xAxis: {
      type: "category",
      data: points.map((p) => p.date),
      axisLine: { lineStyle: { color: axisColor } },
      axisLabel: { color: axisColor, fontSize: 10 },
    },
    yAxis: {
      type: "value",
      name: meta.unit,
      axisLabel: { color: axisColor, fontSize: 10 },
      splitLine: {
        lineStyle: {
          color: resolvedTheme === "dark" ? "#293747" : "#EDF2F7",
        },
      },
    },
    series: [
      {
        type: "line",
        showSymbol: points.length < 400,
        symbolSize: 3,
        data: points.map((p) => p[variable]),
        lineStyle: { width: 1.5, color: lineColor },
        itemStyle: { color: lineColor },
        // Gaps are preserved, never connected across missing days.
        connectNulls: false,
      },
    ],
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Atmospheric Explorer"
        tagline="Station time series from the fused frame. Missing days are gaps, not interpolated values."
      />

      <div className="grid gap-4 lg:grid-cols-[300px_1fr]">
        <div className="space-y-4">
          <Card title="Query">
            <div className="space-y-3">
              <div>
                <label htmlFor="station" className="text-xs font-medium text-fg-muted">
                  Station
                </label>
                <select
                  id="station"
                  value={STATION.name}
                  disabled
                  className="mt-1 w-full rounded-md border border-line bg-surface-raised px-3 py-1.5 text-sm"
                >
                  <option>{STATION.name}</option>
                </select>
              </div>
              <div>
                <label htmlFor="start" className="text-xs font-medium text-fg-muted">
                  Start date
                </label>
                <input
                  id="start"
                  type="date"
                  value={effStart}
                  min={dataset?.start ?? undefined}
                  max={dataset?.end ?? undefined}
                  onChange={(e) => setStart(e.target.value)}
                  className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-1.5 text-sm"
                />
              </div>
              <div>
                <label htmlFor="end" className="text-xs font-medium text-fg-muted">
                  End date
                </label>
                <input
                  id="end"
                  type="date"
                  value={effEnd}
                  min={dataset?.start ?? undefined}
                  max={dataset?.end ?? undefined}
                  onChange={(e) => setEnd(e.target.value)}
                  className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-1.5 text-sm"
                />
              </div>
              <div>
                <label htmlFor="variable" className="text-xs font-medium text-fg-muted">
                  Variable
                </label>
                <select
                  id="variable"
                  value={variable}
                  onChange={(e) => setVariable(e.target.value as VarKey)}
                  className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-1.5 text-sm"
                >
                  {VARIABLES.map((v) => (
                    <option key={v.key} value={v.key}>
                      {v.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </Card>

          <Card title="Station metadata">
            <dl className="space-y-1.5 text-xs">
              <div className="flex justify-between">
                <dt className="text-fg-muted">Latitude</dt>
                <dd className="font-mono">{STATION.lat}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-fg-muted">Longitude</dt>
                <dd className="font-mono">{STATION.lon}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-fg-muted">Rows in range</dt>
                <dd className="font-mono">{series.data?.count ?? "—"}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-fg-muted">Non-null {meta.label}</dt>
                <dd className="font-mono">{points.length}</dd>
              </div>
            </dl>
          </Card>
        </div>

        <div className="space-y-4">
          <Card
            title={`${meta.label}${meta.unit ? ` (${meta.unit})` : ""}`}
            description={`${effStart} → ${effEnd} · gaps preserved`}
          >
            {series.isLoading && <LoadingBlock rows={4} />}
            {series.isError && (
              <ErrorState
                message={(series.error as Error).message}
                onRetry={() => series.refetch()}
              />
            )}
            {series.data && !series.data.available && (
              <EmptyState
                title="Series unavailable"
                description={series.data.reason}
              />
            )}
            {points.length > 0 && (
              <LineChart
                option={option}
                height={360}
                ariaLabel={`${meta.label} time series`}
              />
            )}
            {series.data?.available && points.length === 0 && (
              <EmptyState
                title={`No ${meta.label} observations in this range`}
                description="Widen the date range or choose another variable."
              />
            )}
          </Card>

          <Card title="Spatial view">
            <div className="rounded-md border border-dashed border-line bg-surface-raised p-4">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium">Not available</p>
                <StatusBadge status="UNAVAILABLE" />
              </div>
              <p className="mt-1 text-xs text-fg-muted">
                The interface maps only what the data supports. This repository
                provides one station's time series and its overpass-only AOD
                values; it does not contain a gridded PM2.5 field. A citywide
                pollution surface is therefore not rendered.
              </p>
              <p className="mt-2 text-xs text-fg-muted">
                To enable a map, the backend must expose a documented gridded
                product with per-cell timestamps and a resolution claim.
              </p>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}