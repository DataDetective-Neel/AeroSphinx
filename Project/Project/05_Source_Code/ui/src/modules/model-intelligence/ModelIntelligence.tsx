import { useState } from "react";
import { useExperiments } from "@/lib/queries";
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

export function ModelIntelligence() {
  const experiments = useExperiments();
  const [selected, setSelected] = useState<string | null>(null);
  const m = experiments.data;

  if (experiments.isLoading) return <LoadingBlock rows={5} />;
  if (experiments.isError)
    return (
      <ErrorState
        message={(experiments.error as Error).message}
        onRetry={() => experiments.refetch()}
      />
    );
  if (!m?.available)
    return (
      <EmptyState
        title="Model registry unavailable"
        description={m?.reason ?? "results/legacy_chapter/metrics.json not found"}
      />
    );

  const names = Object.keys(m.metrics ?? {});
  const active = selected ?? names[0] ?? null;
  const mm = active ? m.metrics?.[active] : undefined;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Model Intelligence"
        tagline="Model metadata, feature contracts and metrics as recorded in the current run."
      />

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Card title="Configuration" description={m.configuration ?? "unnamed"}>
          <p className="font-mono text-xs text-fg-muted">
            generated {formatUtc(m.generated)}
          </p>
        </Card>
        <Card title="Station">
          <p className="text-sm">
            {m.station?.name ?? "—"}{" "}
            {m.station && (
              <span className="text-xs text-fg-muted">
                ({m.station.lat}, {m.station.lon})
              </span>
            )}
          </p>
        </Card>
        <Card title="Test rows">
          <p className="text-sm font-mono">{m.split?.n_test ?? "—"}</p>
        </Card>
        <Card title="Features used">
          <p className="text-sm font-mono">{m.features_used?.length ?? "—"}</p>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
        <div className="space-y-1.5">
          {names.map((name) => (
            <button
              key={name}
              onClick={() => setSelected(name)}
              aria-pressed={active === name}
              className={`w-full rounded-md border px-3 py-2 text-left text-sm transition-colors ${
                active === name
                  ? "border-accent/50 bg-accent/10"
                  : "border-line bg-surface hover:bg-surface-raised"
              }`}
            >
              <span className="font-medium">{name}</span>
              <span className="mt-0.5 block font-mono text-[11px] text-fg-muted">
                MAE {formatNumber(m.metrics?.[name].MAE)}
              </span>
            </button>
          ))}
        </div>

        <div className="space-y-4">
          <Card title={active ?? "Select a model"}>
            {mm ? (
              <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <Metric label="MAE" value={formatNumber(mm.MAE)} unit="µg/m³" />
                <Metric label="RMSE" value={formatNumber(mm.RMSE)} unit="µg/m³" />
                <Metric label="R²" value={formatNumber(mm.R2, 3)} />
                <Metric label="Mean bias" value={formatNumber(mm.Mean_Bias)} />
              </dl>
            ) : (
              <EmptyState title="No metrics for this model" />
            )}
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card title="Feature contract" description="Declared inputs of the current configuration">
              <div className="space-y-2 text-xs">
                <p className="font-medium text-fg-muted">Used</p>
                <ul className="space-y-0.5 font-mono">
                  {(m.features_used ?? []).map((f) => (
                    <li key={f} className="truncate">
                      {f}
                    </li>
                  ))}
                </ul>
                <p className="pt-2 font-medium text-fg-muted">Unavailable</p>
                {m.features_unavailable?.length ? (
                  <ul className="space-y-0.5 font-mono text-danger">
                    {m.features_unavailable.map((f) => (
                      <li key={f}>{f}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-fg-muted">None recorded.</p>
                )}
              </div>
            </Card>

            <Card title="Diagnostics coverage" description="What this run can and cannot support">
              <ul className="space-y-2 text-xs">
                <li className="flex items-center justify-between">
                  <span>Residual diagnostics</span>
                  <StatusBadge status="UNAVAILABLE" label="not in this artifact" />
                </li>
                <li className="flex items-center justify-between">
                  <span>Fold-level errors</span>
                  <StatusBadge status="UNAVAILABLE" label="not in this artifact" />
                </li>
                <li className="flex items-center justify-between">
                  <span>Seasonal performance</span>
                  <StatusBadge status="UNAVAILABLE" label="not in this artifact" />
                </li>
                <li className="flex items-center justify-between">
                  <span>Extrapolation behaviour</span>
                  <StatusBadge status="UNAVAILABLE" label="not in this artifact" />
                </li>
              </ul>
              <p className="mt-3 text-xs text-fg-muted">
                These diagnostics are produced by separate evaluation scripts
                (results/diagnostics). They are not fabricated here.
              </p>
            </Card>
          </div>

          <Card title="Hyperparameters">
            {m.hyperparameters ? (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {Object.entries(m.hyperparameters).map(([model, params]) => (
                  <div key={model}>
                    <p className="text-xs font-medium">{model}</p>
                    <ul className="mt-1 space-y-0.5 font-mono text-[11px] text-fg-muted">
                      {Object.entries(params).map(([k, v]) => (
                        <li key={k}>
                          {k}: {String(v)}
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            ) : (
              <EmptyState title="Hyperparameters unavailable" />
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}