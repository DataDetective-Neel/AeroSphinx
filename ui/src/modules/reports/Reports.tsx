import { useState } from "react";
import { useExperiments, useManifests, useStatus } from "@/lib/queries";
import {
  Card,
  EmptyState,
  ErrorState,
  LoadingBlock,
  PageHeader,
  StatusBadge,
  formatUtc,
} from "@/components/ui/primitives";

function download(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function metricsToCsv(metrics: Record<string, Record<string, number>>): string {
  const models = Object.keys(metrics);
  const keys = Array.from(
    new Set(models.flatMap((m) => Object.keys(metrics[m]))),
  );
  const lines = [["model", ...keys].join(",")];
  for (const m of models) {
    lines.push([m, ...keys.map((k) => metrics[m][k] ?? "")].join(","));
  }
  return lines.join("\n");
}

export function Reports() {
  const experiments = useExperiments();
  const manifests = useManifests();
  const status = useStatus();
  const [selectedManifest, setSelectedManifest] = useState<string | null>(null);

  const m = experiments.data;
  const mf = manifests.data;
  const active =
    mf?.manifests.find((x) => x.path === selectedManifest) ??
    mf?.manifests[0] ??
    null;

  function exportJson() {
    if (!m?.available) return;
    download(
      `aerosphinx-experiment-${m.configuration ?? "current"}.json`,
      JSON.stringify(
        { generated: m.generated, experiments: m, source: "results/legacy_chapter/metrics.json" },
        null,
        2,
      ),
      "application/json",
    );
  }

  function exportCsv() {
    if (!m?.metrics) return;
    download(
      `aerosphinx-metrics-${m.configuration ?? "current"}.csv`,
      metricsToCsv(m.metrics),
      "text/csv",
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Reports & Reproducibility"
        tagline="Experiment manifests, provenance and exportable metrics. Missing manifest fields are shown as missing."
        actions={
          <div className="flex gap-2">
            <button
              onClick={exportJson}
              disabled={!m?.available}
              className="rounded-md border border-line px-3 py-1.5 text-xs font-medium hover:bg-surface-raised disabled:opacity-40"
            >
              Export JSON
            </button>
            <button
              onClick={exportCsv}
              disabled={!m?.metrics}
              className="rounded-md border border-line px-3 py-1.5 text-xs font-medium hover:bg-surface-raised disabled:opacity-40"
            >
              Export CSV
            </button>
            <button
              disabled
              title="No PDF renderer is exposed by the backend"
              className="cursor-not-allowed rounded-md border border-line px-3 py-1.5 text-xs font-medium opacity-40"
            >
              Export PDF
            </button>
          </div>
        }
      />

      <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
        <Card title="Experiment registry" description="Manifests found under results/">
          {manifests.isLoading && <LoadingBlock rows={4} />}
          {manifests.isError && (
            <ErrorState
              message={(manifests.error as Error).message}
              onRetry={() => manifests.refetch()}
            />
          )}
          {mf && !mf.available && (
            <EmptyState
              title="No manifests found"
              description="No results/**/manifest.json exists in this checkout."
            />
          )}
          {mf?.available && (
            <ul className="space-y-1.5">
              {mf.manifests.map((x) => (
                <li key={x.path}>
                  <button
                    onClick={() => setSelectedManifest(x.path)}
                    aria-pressed={active?.path === x.path}
                    className={`w-full rounded-md border px-3 py-2 text-left text-xs transition-colors ${
                      active?.path === x.path
                        ? "border-accent/50 bg-accent/10"
                        : "border-line bg-surface hover:bg-surface-raised"
                    }`}
                  >
                    <span className="block truncate font-mono">{x.path}</span>
                    <span className="mt-0.5 block text-fg-muted">
                      {x.run_id ?? "no run_id"} · {x.track ?? "unknown track"}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <div className="space-y-4">
          <Card title="Manifest viewer" description="Selected run metadata">
            {!active ? (
              <EmptyState title="Select a manifest" />
            ) : (
              <dl className="grid gap-3 sm:grid-cols-2">
                {[
                  ["Run ID", active.run_id],
                  ["Git SHA", active.git_sha],
                  ["Generated", active.generated],
                  ["Track", active.track],
                  ["Path", active.path],
                ].map(([label, value]) => (
                  <div key={label}>
                    <dt className="text-xs text-fg-muted">{label}</dt>
                    <dd className="font-mono text-xs">
                      {value ?? (
                        <span className="text-fg-muted">missing</span>
                      )}
                    </dd>
                  </div>
                ))}
                <div>
                  <dt className="text-xs text-fg-muted">Working tree</dt>
                  <dd>
                    {status.data?.repo.dirty ? (
                      <StatusBadge status="WARNING" label="uncommitted" />
                    ) : (
                      <StatusBadge status="PASS" label="clean" />
                    )}
                  </dd>
                </div>
              </dl>
            )}
          </Card>

          <Card
            title="Metrics snapshot"
            description="Source: results/legacy_chapter/metrics.json"
            actions={
              m?.generated ? (
                <StatusBadge status="PASS" label={`generated ${formatUtc(m.generated)}`} />
              ) : undefined
            }
          >
            {experiments.isLoading && <LoadingBlock rows={3} />}
            {m?.available && m.metrics ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm min-w-0">
                  <thead>
                    <tr className="border-b border-line text-left text-[11px] uppercase tracking-wider text-fg-muted">
                      <th className="py-2 pr-3">Model</th>
                      <th className="py-2 pr-3">MAE</th>
                      <th className="py-2 pr-3">RMSE</th>
                      <th className="py-2">R²</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(m.metrics).map(([name, mm]) => (
                      <tr key={name} className="border-b border-line/60">
                        <td className="py-2 pr-3 font-medium">{name}</td>
                        <td className="py-2 pr-3 font-mono text-xs">
                          {mm.MAE?.toFixed(2)}
                        </td>
                        <td className="py-2 pr-3 font-mono text-xs">
                          {mm.RMSE?.toFixed(2)}
                        </td>
                        <td className="py-2 font-mono text-xs">
                          {mm.R2?.toFixed(3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <EmptyState
                title="Metrics unavailable"
                description={m?.reason ?? "results/legacy_chapter/metrics.json not found"}
              />
            )}
          </Card>

          <Card title="Report preview">
            <pre className="max-h-64 overflow-auto rounded-md border border-line bg-surface-raised p-3 font-mono text-[11px] leading-relaxed min-w-0">
{`AeroSphinx — Experiment Report
Configuration : ${m?.configuration ?? "unavailable"}
Generated     : ${formatUtc(m?.generated)}
Station       : ${m?.station?.name ?? "unavailable"}
Train rows    : ${m?.split?.n_train ?? "unavailable"}
Test rows     : ${m?.split?.n_test ?? "unavailable"}
Git           : ${status.data?.repo.short_sha ?? "unavailable"}

Metrics (one-day horizon):
${Object.entries(m?.metrics ?? {})
  .map(
    ([k, v]) =>
      `  ${k.padEnd(14)} MAE ${v.MAE?.toFixed(2) ?? "—"}  R² ${v.R2?.toFixed(3) ?? "—"}`,
  )
  .join("\n") || "  unavailable"}

Note: Track B is retrospective estimation, not operational forecasting.`}
            </pre>
          </Card>
        </div>
      </div>
    </div>
  );
}