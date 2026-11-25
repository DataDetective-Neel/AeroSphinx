import { useState } from "react";
import {
  useExperiments,
  useExperimentMatrix,
  useSeasonalSufficiency,
} from "@/lib/queries";
import type { ArtifactRow } from "@/lib/api";
import { EXPERIMENT_CATALOG, type ExperimentDef } from "@/lib/reference";
import {
  Card,
  EmptyState,
  ErrorState,
  LoadingBlock,
  Metric,
  PageHeader,
  StatusBadge,
  formatNumber,
  formatUtc,
} from "@/components/ui/primitives";

function cell(row: ArtifactRow, key: string) {
  const value = row[key];
  return value === null || value === undefined ? "—" : String(value);
}

function numeric(row: ArtifactRow, key: string) {
  const value = row[key];
  return typeof value === "number" ? value : null;
}

function interval(row: ArtifactRow) {
  const low = numeric(row, "ci_low");
  const high = numeric(row, "ci_high");
  if (low === null || high === null) return null;
  const signed = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
  return `[${signed(low)}, ${signed(high)}]`;
}

/** A paired difference is only called out when its interval excludes zero. */
function verdict(row: ArtifactRow) {
  if (cell(row, "reported").toLowerCase() === "false") {
    return "suppressed (n too small)";
  }
  // The API passes CSV cells through unchanged, so a boolean can arrive as
  // the string "True"; tolerate both spellings rather than depending on it.
  const significant = cell(row, "significant").toLowerCase() === "true";
  const p = numeric(row, "p_holm");
  if (p === null) return significant ? "significant" : "not significant";
  return `${significant ? "significant" : "not significant"} · Holm p ${p.toFixed(3)}`;
}

export function ResearchLab() {
  const [track, setTrack] = useState<"A" | "B">("A");
  const [selected, setSelected] = useState<string>("A2");
  const experiments = useExperiments();
  const matrix = useExperimentMatrix();
  const sufficiency = useSeasonalSufficiency();
  const mx = matrix.data;
  const sf = sufficiency.data;
  const defs = EXPERIMENT_CATALOG.filter((d) => d.track === track);
  const current: ExperimentDef | undefined =
    EXPERIMENT_CATALOG.find((d) => d.id === selected) ?? defs[0];

  function switchTrack(next: "A" | "B") {
    setTrack(next);
    setSelected(next === "A" ? "A2" : "B5");
  }

  const m = experiments.data;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Research Lab"
        tagline="Track A and Track B experiments with the metrics actually produced by the Python package."
      />

      <div className="inline-flex rounded-lg border border-line bg-surface p-1">
        {(["A", "B"] as const).map((t) => (
          <button
            key={t}
            onClick={() => switchTrack(t)}
            aria-pressed={track === t}
            className={`rounded-md px-4 py-1.5 text-sm font-medium transition-colors ${
              track === t
                ? "bg-accent text-surface"
                : "text-fg-muted hover:text-fg"
            }`}
          >
            Track {t}
          </button>
        ))}
      </div>

      <p className="text-xs text-fg-muted">
        {track === "A"
          ? "Track A uses CPCB ground history only. Comparisons below are one-day-horizon metrics on the shared chronological holdout."
          : "Track B is retrospective estimation on matched dates. It may not be described as operational forecasting."}
      </p>

      <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
        <div className="space-y-1.5">
          {defs.map((d) => (
            <button
              key={d.id}
              onClick={() => setSelected(d.id)}
              aria-pressed={selected === d.id}
              className={`w-full rounded-md border px-3 py-2 text-left text-sm transition-colors ${
                selected === d.id
                  ? "border-accent/50 bg-accent/10"
                  : "border-line bg-surface hover:bg-surface-raised"
              }`}
            >
              <span className="font-mono text-[11px] text-fg-muted">{d.id}</span>
              <span className="ml-2 font-medium">{d.name}</span>
              <span className="mt-0.5 block text-xs text-fg-muted">
                {d.question}
              </span>
            </button>
          ))}
        </div>

        <div className="space-y-4">
          <Card
            title={current ? `${current.id} · ${current.name}` : "Select an experiment"}
            description={current?.question}
            actions={
              current ? (
                <StatusBadge
                  status={current.mode === "retrospective" ? "WARNING" : "PASS"}
                  label={current.mode}
                />
              ) : undefined
            }
          >
            {current && (
              <>
                <p className="mb-3 text-xs text-fg-muted">
                  <span className="font-medium text-fg">Inputs:</span>{" "}
                  {current.inputs.join(", ")}
                </p>
                <p className="text-xs text-fg-muted">
                  Metrics for this experiment are read from{" "}
                  <code className="text-fg">results/legacy_chapter/metrics.json</code>. If a
                  model was not fitted for this configuration, it is shown as
                  unavailable rather than estimated.
                </p>
              </>
            )}
          </Card>

          <Card
            title="Model comparison (legacy pipeline)"
            description="One-day horizon on the legacy 33-day holdout, 2024-05-17 → 2024-06-18"
          >
            {experiments.isLoading && <LoadingBlock rows={4} />}
            {experiments.isError && (
              <ErrorState
                message={(experiments.error as Error).message}
                onRetry={() => experiments.refetch()}
              />
            )}
            {m && !m.available && (
              <EmptyState
                title="Metrics unavailable"
                description={m.reason ?? "results/legacy_chapter/metrics.json not found"}
              />
            )}
            {m?.available && (
              <div className="overflow-x-auto">
                <table className="w-full text-sm min-w-0">
                  <thead>
                    <tr className="border-b border-line text-left text-[11px] uppercase tracking-wider text-fg-muted">
                      <th className="py-2 pr-3">Model</th>
                      <th className="py-2 pr-3">MAE</th>
                      <th className="py-2 pr-3">RMSE</th>
                      <th className="py-2 pr-3">R²</th>
                      <th className="py-2">Bias</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(m.metrics ?? {}).map(([name, mm]) => (
                      <tr key={name} className="border-b border-line/60">
                        <td className="py-2 pr-3 font-medium">{name}</td>
                        <td className="py-2 pr-3 font-mono text-xs">
                          {formatNumber(mm.MAE)}
                        </td>
                        <td className="py-2 pr-3 font-mono text-xs">
                          {formatNumber(mm.RMSE)}
                        </td>
                        <td className="py-2 pr-3 font-mono text-xs">
                          {formatNumber(mm.R2, 3)}
                        </td>
                        <td className="py-2 font-mono text-xs">
                          {formatNumber(mm.Mean_Bias)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card
              title="Legacy ablation (Track B)"
              description="33 test rows, superseded by the matrix ablations below"
            >
              {m?.baselines ? (
                <ul className="space-y-1.5 text-sm">
                  {Object.entries(m.baselines).map(([name, b]) => (
                    <li key={name} className="flex justify-between gap-3">
                      <span className="text-fg-muted">{name}</span>
                      <span className="font-mono text-xs">
                        MAE {formatNumber(b.MAE)}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <EmptyState title="Ablation unavailable" />
              )}
            </Card>

            <Card
              title="Track A: learned model vs persistence"
              description="Paired difference from the matrix artefacts, per track"
            >
              {(() => {
                const row = (mx?.ablations ?? []).find(
                  (candidate) =>
                    cell(candidate, "track") === track &&
                    cell(candidate, "config") === "M1" &&
                    cell(candidate, "reference") === "M0",
                );
                if (!row) {
                  return (
                    <EmptyState
                      title="No paired comparison for this track"
                      description={mx?.reason}
                    />
                  );
                }
                return (
                  <>
                    <dl className="grid grid-cols-2 gap-3">
                      <Metric
                        label="Δ MAE (ridge − persistence)"
                        value={formatNumber(numeric(row, "delta_MAE"))}
                      />
                      <Metric label="Pairs" value={cell(row, "n")} />
                      <Metric label="95% interval (days)" value={interval(row) ?? "—"} />
                      <Metric
                        label="95% interval (years)"
                        value={
                          numeric(row, "year_ci_low") === null
                            ? "—"
                            : `[${numeric(row, "year_ci_low")!.toFixed(2)}, ${numeric(
                                row,
                                "year_ci_high",
                              )!.toFixed(2)}]`
                        }
                      />
                    </dl>
                    <p className="mt-3 text-xs text-fg-muted">
                      {verdict(row)}. A negative Δ means the learned model is
                      better. The day-level interval resamples contiguous blocks;
                      the year-level interval resamples whole calendar years and is
                      the conservative test, because eight years is a small number
                      of resampling units.
                    </p>
                  </>
                );
              })()}
            </Card>
          </div>

          <Card
            title="Seasonal experiment matrix (M0–M8)"
            description="Identical eligible dates and rolling-origin folds within each track"
            actions={
              mx?.available ? (
                <span className="text-right font-mono text-[11px] leading-4 text-fg-muted">
                  {mx.run_id ?? "matrix"}
                  <br />
                  {mx.generated ? `generated ${formatUtc(mx.generated)}` : "generated —"}
                </span>
              ) : undefined
            }
          >
            {matrix.isLoading && <LoadingBlock rows={4} />}
            {matrix.isError && (
              <ErrorState
                message={(matrix.error as Error).message}
                onRetry={() => matrix.refetch()}
              />
            )}
            {mx && !mx.available && (
              <EmptyState
                title="Matrix unavailable"
                description={mx.reason ?? "run scripts/run_experiment_matrix.py"}
              />
            )}
            {mx?.available && (
              <div className="overflow-x-auto">
                <table className="w-full text-sm min-w-0">
                  <thead>
                    <tr className="border-b border-line text-left text-[11px] uppercase tracking-wider text-fg-muted">
                      <th className="py-2 pr-3">Config</th>
                      <th className="py-2 pr-3">Model</th>
                      <th className="py-2 pr-3">n</th>
                      <th className="py-2 pr-3">MAE</th>
                      <th className="py-2 pr-3">RMSE</th>
                      <th className="py-2 pr-3">MedAE</th>
                      <th className="py-2">R²</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(mx.metrics ?? [])
                      .filter((row) => cell(row, "track") === track)
                      .map((row) => (
                        <tr
                          key={`${cell(row, "track")}-${cell(row, "config")}`}
                          className="border-b border-line/60"
                        >
                          <td className="py-2 pr-3 font-mono text-xs">
                            {cell(row, "config")}
                          </td>
                          <td className="py-2 pr-3">{cell(row, "model")}</td>
                          <td className="py-2 pr-3 font-mono text-xs">
                            {cell(row, "n")}
                          </td>
                          <td className="py-2 pr-3 font-mono text-xs">
                            {formatNumber(numeric(row, "MAE"))}
                          </td>
                          <td className="py-2 pr-3 font-mono text-xs">
                            {formatNumber(numeric(row, "RMSE"))}
                          </td>
                          <td className="py-2 pr-3 font-mono text-xs">
                            {formatNumber(numeric(row, "MedianAE"))}
                          </td>
                          <td className="py-2 font-mono text-xs">
                            {formatNumber(numeric(row, "R2"), 3)}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
                {(mx.metrics ?? []).filter(
                  (row) => cell(row, "track") === track,
                ).length === 0 && (
                  <p className="mt-2 text-xs text-fg-muted">
                    No configurations recorded for Track {track}.
                  </p>
                )}
              </div>
            )}
          </Card>

          <Card
            title="Source ablation (paired)"
            description="delta MAE = config − reference on the same dates; negative is better"
          >
            {mx?.available &&
            (mx.ablations ?? []).filter(
              (row) => cell(row, "track") === track,
            ).length > 0 ? (
              <ul className="space-y-1.5 text-sm">
                {(mx.ablations ?? [])
                  .filter((row) => cell(row, "track") === track)
                  .map((row) => (
                    <li
                      key={`${cell(row, "config")}-${cell(row, "reference")}`}
                      className="flex justify-between gap-3"
                    >
                      <span className="text-fg-muted">
                        {cell(row, "config")} vs {cell(row, "reference")}
                      </span>
                      <span className="flex flex-col items-end font-mono text-xs">
                        <span>{formatNumber(numeric(row, "delta_MAE"))}</span>
                        <span className="text-[10px] text-fg-muted">
                          {interval(row) ?? "interval —"}
                        </span>
                        <span className="text-[10px] text-fg-muted">
                          {verdict(row)}
                        </span>
                      </span>
                    </li>
                  ))}
              </ul>
            ) : (
              <EmptyState
                title={
                  mx?.available
                    ? `No paired ablations for Track ${track}`
                    : "Ablation unavailable"
                }
                description={
                  mx?.available
                    ? track === "A"
                      ? "Paired source ablations are defined for Track B only: Track A has no environmental group to add or remove."
                      : "No paired rows were recorded for this track."
                    : mx?.reason
                }
              />
            )}
          </Card>

          <Card
            title="Per-season source sufficiency"
            description="Declared thresholds: 30 rows to compare, 90 rows for a stable seasonal estimate"
          >
            {sufficiency.isLoading && <LoadingBlock rows={3} />}
            {sufficiency.isError && (
              <ErrorState
                message={(sufficiency.error as Error).message}
                onRetry={() => sufficiency.refetch()}
              />
            )}
            {sf && !sf.available && (
              <EmptyState
                title="Sufficiency unavailable"
                description={sf.reason}
              />
            )}
            {sf?.available && (
              <div className="overflow-x-auto">
                <table className="w-full text-sm min-w-0">
                  <thead>
                    <tr className="border-b border-line text-left text-[11px] uppercase tracking-wider text-fg-muted">
                      <th className="py-2 pr-3">Season</th>
                      <th className="py-2 pr-3">Source</th>
                      <th className="py-2 pr-3">Observed</th>
                      <th className="py-2 pr-3">Rows</th>
                      <th className="py-2">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(sf.rows ?? []).map((row) => (
                      <tr
                        key={`${cell(row, "season")}-${cell(row, "source")}`}
                        className="border-b border-line/60"
                      >
                        <td className="py-2 pr-3">{cell(row, "season")}</td>
                        <td className="py-2 pr-3 font-mono text-xs">
                          {cell(row, "source")}
                        </td>
                        <td className="py-2 pr-3 font-mono text-xs">
                          {cell(row, "n_observed")}
                        </td>
                        <td className="py-2 pr-3 font-mono text-xs">
                          {cell(row, "n_season_rows")}
                        </td>
                        <td className="py-2">
                          <StatusBadge status={cell(row, "status")} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}