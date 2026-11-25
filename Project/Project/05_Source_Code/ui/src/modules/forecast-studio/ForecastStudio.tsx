import { useMemo, useState } from "react";
import { useForecasts, useSeries, useStatus } from "@/lib/queries";
import {
  Card,
  EmptyState,
  ErrorState,
  LoadingBlock,
  PageHeader,
  StatusBadge,
} from "@/components/ui/primitives";
import { STATION } from "@/lib/reference";

type ForecastStatus = "VALIDATED" | "DEGRADED" | "UNSUPPORTED" | "ABSTAINED";

const MODELS: Array<{ id: string; name: string; needs: string[] }> = [
  { id: "persistence", name: "Persistence", needs: ["pm25_history"] },
  { id: "temporal_ml", name: "Temporal ML", needs: ["pm25_history"] },
  { id: "fusion", name: "Full fusion", needs: ["pm25_history", "aod", "merra"] },
];

export function ForecastStudio() {
  const status = useStatus();
  const forecasts = useForecasts();
  const [issueDate, setIssueDate] = useState<string>("");
  const [horizon, setHorizon] = useState(1);
  const [model, setModel] = useState<string>("persistence");
  const [submitted, setSubmitted] = useState(false);

  const range = status.data?.dataset;
  const series = useSeries(range?.start ?? "", range?.end ?? "", !!range);

  const point = useMemo(
    () => series.data?.rows?.find((r) => r.date === issueDate) ?? null,
    [series.data, issueDate],
  );

  const selectedModel = MODELS.find((m) => m.id === model)!;

  // Conservative eligibility: a model requiring an environmental source that
  // is not observed on the issue date is UNSUPPORTED. No environmental value is
  // ever assumed available.
  const eligibility = useMemo(() => {
    const reasons: string[] = [];
    let st: ForecastStatus = "VALIDATED";
    if (!point) {
      return {
        status: "ABSTAINED" as ForecastStatus,
        reasons: ["Choose an issue date with a ground observation."],
      };
    }
    const needsAod = selectedModel.needs.includes("aod");
    const needsMerra = selectedModel.needs.includes("merra");
    if (needsAod && !point.sources.aod) {
      st = "UNSUPPORTED";
      reasons.push("INSAT-3D AOD is not observed on the issue date.");
    }
    if (needsMerra && !point.sources.merra) {
      st = "UNSUPPORTED";
      reasons.push("MERRA-2 reanalysis is not observed on the issue date.");
    }
    if (horizon >= 3) {
      if (st === "VALIDATED") st = "DEGRADED";
      reasons.push(
        `Horizon ${horizon} compounds error; the published comparison only supports one day.`,
      );
    }
    if (st === "VALIDATED" && model === "persistence") {
      st = "DEGRADED";
      reasons.push(
        "No learned model has been shown to beat persistence at p < 0.05 (D-006).",
      );
    }
    if (reasons.length === 0)
      reasons.push("All required features are observed at the issue timestamp.");
    return { status: st, reasons };
  }, [point, selectedModel, horizon, model]);

  const canSubmit =
    !!point && eligibility.status !== "UNSUPPORTED" && eligibility.status !== "ABSTAINED";

  return (
    <div className="space-y-6">
      <PageHeader
        title="Forecast Studio"
        tagline="Eligibility is evaluated against the availability contract. Future observations are never used as inputs."
      />

      <div className="grid gap-4 lg:grid-cols-[340px_1fr]">
        <Card title="Forecast request">
          <div className="space-y-4">
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
                <option>
                  {STATION.name} ({STATION.lat}, {STATION.lon})
                </option>
              </select>
              <p className="mt-1 text-[11px] text-fg-muted">
                Only one station is currently configured.
              </p>
            </div>

            <div>
              <label htmlFor="issue" className="text-xs font-medium text-fg-muted">
                Issue date
              </label>
              <select
                id="issue"
                value={issueDate}
                onChange={(e) => {
                  setIssueDate(e.target.value);
                  setSubmitted(false);
                }}
                className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-1.5 text-sm"
              >
                <option value="">Select a date…</option>
                {(series.data?.rows ?? [])
                  .filter((r) => r.pm25 !== null)
                  .slice(-120)
                  .reverse()
                  .map((r) => (
                    <option key={r.date} value={r.date}>
                      {r.date}
                    </option>
                  ))}
              </select>
            </div>

            <div>
              <label htmlFor="horizon" className="text-xs font-medium text-fg-muted">
                Horizon: day {horizon}
              </label>
              <input
                id="horizon"
                type="range"
                min={1}
                max={5}
                value={horizon}
                onChange={(e) => {
                  setHorizon(Number(e.target.value));
                  setSubmitted(false);
                }}
                className="mt-2 w-full"
              />
            </div>

            <div>
              <label htmlFor="model" className="text-xs font-medium text-fg-muted">
                Model
              </label>
              <select
                id="model"
                value={model}
                onChange={(e) => {
                  setModel(e.target.value);
                  setSubmitted(false);
                }}
                className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-1.5 text-sm"
              >
                {MODELS.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>

            <button
              onClick={() => setSubmitted(true)}
              disabled={!canSubmit}
              className="w-full rounded-md bg-accent px-4 py-2 text-sm font-medium text-surface transition-colors enabled:hover:bg-accent/90 disabled:cursor-not-allowed disabled:bg-surface-raised disabled:text-fg-muted"
            >
              Evaluate eligibility
            </button>
          </div>
        </Card>

        <div className="space-y-4">
          <Card
            title="Feature availability at issue time"
            description="Read directly from the fused frame — no assumed availability"
          >
            {series.isLoading && <LoadingBlock rows={2} />}
            {series.isError && (
              <ErrorState
                message={(series.error as Error).message}
                onRetry={() => series.refetch()}
              />
            )}
            {!issueDate && (
              <EmptyState title="Select an issue date" description="Availability is inspected per timestamp." />
            )}
            {issueDate && point && (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-line text-left text-[11px] uppercase tracking-wider text-fg-muted">
                      <th className="py-2 pr-3">Feature</th>
                      <th className="py-2 pr-3">Value</th>
                      <th className="py-2">Availability</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-line/60">
                      <td className="py-2 pr-3">PM2.5 (ground)</td>
                      <td className="py-2 pr-3 font-mono text-xs">
                        {point.pm25 ?? "—"}
                      </td>
                      <td className="py-2">
                        <StatusBadge
                          status={point.pm25 !== null ? "PASS" : "UNAVAILABLE"}
                          label={point.pm25 !== null ? "observed" : "missing"}
                        />
                      </td>
                    </tr>
                    <tr className="border-b border-line/60">
                      <td className="py-2 pr-3">INSAT-3D AOD</td>
                      <td className="py-2 pr-3 font-mono text-xs">
                        {point.aod ?? "—"}
                      </td>
                      <td className="py-2">
                        <StatusBadge
                          status={point.sources.aod ? "PASS" : "UNAVAILABLE"}
                          label={point.sources.aod ? "observed" : "not observed"}
                        />
                      </td>
                    </tr>
                    <tr className="border-b border-line/60">
                      <td className="py-2 pr-3">MERRA-2</td>
                      <td className="py-2 pr-3 font-mono text-xs">
                        {point.t2m !== null ? `T2M ${point.t2m}` : "—"}
                      </td>
                      <td className="py-2">
                        <StatusBadge
                          status={point.sources.merra ? "PASS" : "UNAVAILABLE"}
                          label={point.sources.merra ? "observed" : "not observed"}
                        />
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          <Card
            title="Eligibility"
            actions={<StatusBadge status={eligibility.status} />}
          >
            <ul className="space-y-1.5 text-sm text-fg-muted">
              {eligibility.reasons.map((r, i) => (
                <li key={i}>• {r}</li>
              ))}
            </ul>
            {!canSubmit && (
              <p className="mt-3 text-xs font-medium text-danger">
                This request cannot be submitted: its inputs are not available
                under the availability contract.
              </p>
            )}
          </Card>

          <Card title="Forecast result">
            {!submitted ? (
              <p className="text-sm text-fg-muted">
                No request submitted yet.
              </p>
            ) : (
              <div className="rounded-md border border-dashed border-warning/40 bg-warning/5 p-4">
                <p className="text-sm font-medium text-warning">
                  Forecast values unavailable
                </p>
                <p className="mt-1 text-xs text-fg-muted">
                  The repository produces forecasts through offline Python
                  scripts. There is no inference endpoint, so no value is
                  rendered here rather than inventing one.
                </p>
                <p className="mt-2 text-xs text-fg-muted">
                  Required backend contract:{" "}
                  <code className="text-fg">
                    POST /api/forecast {"{ station, issue_time, horizon, model }"}
                  </code>{" "}
                  → forecast value, calibrated interval (omit if uncalibrated),
                  and the feature-availability record used.
                </p>
              </div>
            )}
          </Card>
        </div>
      </div>

      <Card
        title="Persisted projections"
        description="Read-only rows from results/. These are retrospective projections, not operational forecasts (D-016)."
      >
        {forecasts.isLoading && <LoadingBlock rows={3} />}
        {forecasts.isError && (
          <ErrorState
            message={(forecasts.error as Error).message}
            onRetry={() => forecasts.refetch()}
          />
        )}
        {!forecasts.isLoading && !forecasts.isError && !forecasts.data?.available && (
          <EmptyState
            title="No persisted projection"
            description={forecasts.data?.reason ?? "No forecast projection artifact is present."}
          />
        )}
        {forecasts.data?.available && (
          <div className="space-y-6">
            <div>
              <h3 className="mb-2 text-sm font-medium text-fg">
                Operational forward projection ({forecasts.data.operational?.path})
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-line text-left text-[11px] uppercase tracking-wider text-fg-muted">
                      <th className="py-2 pr-3">Date</th>
                      <th className="py-2 pr-3">Horizon</th>
                      <th className="py-2 pr-3">Predicted PM2.5</th>
                      <th className="py-2 pr-3">AQI class</th>
                      <th className="py-2">Meteorology</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(forecasts.data.operational?.rows ?? []).map((r, i) => (
                      <tr key={i} className="border-b border-line/60">
                        <td className="py-2 pr-3 font-mono text-xs">{String(r.Date ?? r.date ?? "—")}</td>
                        <td className="py-2 pr-3 font-mono text-xs">{String(r.horizon_day ?? "—")}</td>
                        <td className="py-2 pr-3 font-mono text-xs">
                          {typeof r.predicted_PM25 === "number"
                            ? r.predicted_PM25.toFixed(1)
                            : String(r.predicted_PM25 ?? "—")}
                        </td>
                        <td className="py-2 pr-3">{String(r.predicted_AQI_class ?? "—")}</td>
                        <td className="py-2 text-xs text-fg-muted">
                          {String(r.meteorology_source ?? "—")}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h3 className="mb-2 text-sm font-medium text-fg">
                Five-day recursive projection ({forecasts.data.recursive?.path})
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-line text-left text-[11px] uppercase tracking-wider text-fg-muted">
                      <th className="py-2 pr-3">Date</th>
                      <th className="py-2 pr-3">Horizon</th>
                      <th className="py-2 pr-3">Actual PM2.5</th>
                      <th className="py-2 pr-3">Predicted PM2.5</th>
                      <th className="py-2 pr-3">Abs. error</th>
                      <th className="py-2">AQI class</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(forecasts.data.recursive?.rows ?? []).map((r, i) => (
                      <tr key={i} className="border-b border-line/60">
                        <td className="py-2 pr-3 font-mono text-xs">{String(r.Date ?? r.date ?? "—")}</td>
                        <td className="py-2 pr-3 font-mono text-xs">{String(r.horizon_day ?? "—")}</td>
                        <td className="py-2 pr-3 font-mono text-xs">
                          {typeof r.actual_PM25 === "number"
                            ? r.actual_PM25.toFixed(1)
                            : String(r.actual_PM25 ?? "—")}
                        </td>
                        <td className="py-2 pr-3 font-mono text-xs">
                          {typeof r.predicted_PM25 === "number"
                            ? r.predicted_PM25.toFixed(1)
                            : String(r.predicted_PM25 ?? "—")}
                        </td>
                        <td className="py-2 pr-3 font-mono text-xs">
                          {typeof r.absolute_error === "number"
                            ? r.absolute_error.toFixed(1)
                            : String(r.absolute_error ?? "—")}
                        </td>
                        <td className="py-2">{String(r.predicted_AQI_class ?? "—")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}