import { useDataset, useAlignment } from "@/lib/queries";
import type { ArtifactRow } from "@/lib/api";
import {
  Card,
  EmptyState,
  ErrorState,
  LoadingBlock,
  PageHeader,
  StatCard,
  StatusBadge,
  formatNumber,
  formatPct,
} from "@/components/ui/primitives";

function cell(row: ArtifactRow, key: string) {
  const value = row[key];
  return value === null || value === undefined ? "—" : String(value);
}

function numeric(row: ArtifactRow, key: string) {
  const value = row[key];
  return typeof value === "number" ? value : null;
}

const SOURCE_META = [
  {
    key: "aod_observed",
    name: "INSAT-3D AOD",
    provider: "ISRO / SAC",
    note: "139 dates, 2024 only. No QA/cloud flag dataset (D-004).",
    status: "WARNING" as const,
  },
  {
    key: "merra_observed",
    name: "MERRA-2",
    provider: "NASA GMAO",
    note: "172 dates, 2024 only. Midday single-cell values (D-010).",
    status: "WARNING" as const,
  },
  {
    key: "station_met_observed",
    name: "CPCB station meteorology",
    provider: "CPCB NAMP",
    note: "Full record. BP screened out (D-020); VWS 43% missing.",
    status: "PASS" as const,
  },
];

export function DataObservatory() {
  const dataset = useDataset();
  const alignment = useAlignment();
  const al = alignment.data;

  if (dataset.isLoading) return <LoadingBlock rows={5} />;
  if (dataset.isError)
    return (
      <ErrorState
        message={(dataset.error as Error).message}
        onRetry={() => dataset.refetch()}
      />
    );
  const d = dataset.data;
  if (!d?.available)
    return (
      <EmptyState
        title="Dataset unavailable"
        description={d?.reason ?? "fused_dataset.csv not found"}
      />
    );

  const missing = d.missingness ?? {};

  return (
    <div className="space-y-6">
      <PageHeader
        title="Data Observatory"
        tagline="Source registry, coverage and missingness computed from the fused frame. Missing values are never treated as zero."
      />

      <div className="grid gap-3 sm:grid-cols-3">
        <StatCard label="Rows" value={d.rows?.toLocaleString()} />
        <StatCard label="Columns" value={d.columns} />
        <StatCard
          label="Date range"
          value={
            <span className="text-base">
              {d.start} → {d.end}
            </span>
          }
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        {SOURCE_META.map((src) => {
          const m = missing[src.key];
          return (
            <Card
              key={src.key}
              title={src.name}
              description={src.provider}
              actions={<StatusBadge status={src.status} />}
            >
              <p className="mb-2 text-xs text-fg-muted">{src.note}</p>
              {m ? (
                <p className="text-xs">
                  <span className="text-fg-muted">Missing: </span>
                  <span className="font-mono">
                    {m.missing} / {d.rows} ({formatPct(m.fraction)})
                  </span>
                </p>
              ) : (
                <p className="text-xs text-fg-muted">Column not present.</p>
              )}
            </Card>
          );
        })}
      </div>

      <Card
        title="Daily alignment (persisted artefacts)"
        description="Coverage read from results/alignment — a missing value is never treated as zero"
      >
        {alignment.isLoading && <LoadingBlock rows={3} />}
        {alignment.isError && (
          <ErrorState
            message={(alignment.error as Error).message}
            onRetry={() => alignment.refetch()}
          />
        )}
        {al && !al.available && (
          <EmptyState
            title="Alignment artefacts unavailable"
            description={al.reason}
          />
        )}
        {al?.available && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm min-w-0">
              <thead>
                <tr className="border-b border-line text-left text-[11px] uppercase tracking-wider text-fg-muted">
                  <th className="py-2 pr-3">Source</th>
                  <th className="py-2 pr-3">Observed</th>
                  <th className="py-2 pr-3">Missing</th>
                  <th className="py-2 pr-3">Coverage</th>
                  <th className="py-2">First → last</th>
                </tr>
              </thead>
              <tbody>
                {(al.coverage ?? []).map((row) => (
                  <tr key={cell(row, "source")} className="border-b border-line/60">
                    <td className="py-2 pr-3 font-mono text-xs">
                      {cell(row, "source")}
                    </td>
                    <td className="py-2 pr-3 font-mono text-xs">
                      {cell(row, "observed_rows")}
                    </td>
                    <td className="py-2 pr-3 font-mono text-xs">
                      {cell(row, "missing_rows")}
                    </td>
                    <td className="py-2 pr-3 font-mono text-xs">
                      {formatPct(numeric(row, "observed_fraction"))}
                    </td>
                    <td className="py-2 font-mono text-xs">
                      {cell(row, "first_observed")} → {cell(row, "last_observed")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Card
        title="Missingness contrast"
        description="Target mean on observed versus unobserved dates: informative missingness, not random"
      >
        {al?.available ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm min-w-0">
              <thead>
                <tr className="border-b border-line text-left text-[11px] uppercase tracking-wider text-fg-muted">
                  <th className="py-2 pr-3">Source</th>
                  <th className="py-2 pr-3">Mean when observed</th>
                  <th className="py-2 pr-3">Mean when absent</th>
                  <th className="py-2">Difference</th>
                </tr>
              </thead>
              <tbody>
                {(al.missingness ?? []).map((row) => (
                  <tr key={cell(row, "source")} className="border-b border-line/60">
                    <td className="py-2 pr-3 font-mono text-xs">
                      {cell(row, "source")}
                    </td>
                    <td className="py-2 pr-3 font-mono text-xs">
                      {formatNumber(numeric(row, "target_mean_when_observed"))}
                    </td>
                    <td className="py-2 pr-3 font-mono text-xs">
                      {formatNumber(numeric(row, "target_mean_when_unobserved"))}
                    </td>
                    <td className="py-2 font-mono text-xs">
                      {formatNumber(numeric(row, "target_mean_difference"))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            title="Missingness contrast unavailable"
            description={al?.reason}
          />
        )}
      </Card>

      <Card
        title="Per-column missingness"
        description="Raw NaN counts from the fused frame — no substitution applied"
      >
        <div className="max-h-96 overflow-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-surface">
              <tr className="border-b border-line text-left text-[11px] uppercase tracking-wider text-fg-muted">
                <th className="py-2 pr-3">Column</th>
                <th className="py-2 pr-3">Missing</th>
                <th className="py-2">Fraction</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(missing)
                .sort((a, b) => b[1].fraction - a[1].fraction)
                .map(([col, v]) => (
                  <tr key={col} className="border-b border-line/50">
                    <td className="py-1.5 pr-3 font-mono text-xs">{col}</td>
                    <td className="py-1.5 pr-3 font-mono text-xs">{v.missing}</td>
                    <td className="py-1.5 font-mono text-xs">
                      {formatPct(v.fraction)}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        <Card title="Provenance & lineage" description="Recorded in run manifests">
          <ul className="space-y-2 text-xs text-fg-muted">
            <li>• Input SHA-256 hashes are recorded per run in the manifest.</li>
            <li>
              • Every feature has an availability class; a forward-filled value is
              never given the class of the column it fills.
            </li>
            <li>
              • The ground series defines the row calendar; sources cannot extend
              it.
            </li>
            <li>
              • Measurement time and availability time are distinct columns.
            </li>
          </ul>
        </Card>
        <Card title="Legacy preservation">
          <p className="text-xs text-fg-muted">
            The legacy single-script pipeline{" "}
            <code className="text-fg">legacy/aerosphinx_pipeline.py</code> is retained
            unchanged as the Phase 0 reproducibility baseline. It is not the
            research path and its forward-fill behaviour must not be reused for
            Track B experiments.
          </p>
        </Card>
      </div>
    </div>
  );
}