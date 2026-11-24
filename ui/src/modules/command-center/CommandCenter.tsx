import {
  useCoverage,
  useExperimentMatrix,
  useExperiments,
  useStatus,
} from "@/lib/queries";
import type { ArtifactRow } from "@/lib/api";
import { DEFECTS } from "@/lib/reference";
import {
  Card,
  EmptyState,
  ErrorState,
  LoadingBlock,
  PageHeader,
  StatCard,
  StatusBadge,
  formatPct,
  formatUtc,
} from "@/components/ui/primitives";


function numeric(row: ArtifactRow, key: string) {
  const value = row[key];
  return typeof value === "number" ? value : null;
}

function signed(value: number, digits = 2) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

/** "−0.65 [−1.14, −0.17]" or a dash when the artefact withholds the value. */
function paired(row: ArtifactRow | undefined) {
  if (!row) return null;
  const delta = numeric(row, "delta_MAE");
  const low = numeric(row, "ci_low");
  const high = numeric(row, "ci_high");
  if (delta === null) return null;
  const interval = low === null || high === null ? "" : ` [${signed(low)}, ${signed(high)}]`;
  return { delta, text: `${signed(delta)}${interval}` };
}

function findPair(rows: ArtifactRow[] | undefined, track: string, config: string, reference: string) {
  return (rows ?? []).find(
    (row) =>
      String(row.track) === track &&
      String(row.config) === config &&
      String(row.reference) === reference,
  );
}

export function CommandCenter() {
  const status = useStatus();
  const coverage = useCoverage();
  const experiments = useExperiments();
  const matrix = useExperimentMatrix();

  if (status.isLoading) return <LoadingBlock rows={5} />;
  if (status.isError)
    return (
      <ErrorState
        message={(status.error as Error).message}
        onRetry={() => status.refetch()}
      />
    );
  if (!status.data) return null;

  const s = status.data;
  const ds = s.dataset;
  const cov = coverage.data;
  const exp = experiments.data;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Command Center"
        tagline="Repository, datasets, coverage and research status — read from the live artifact API."
      />

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Ground rows"
          value={ds.available ? ds.rows?.toLocaleString() : "—"}
          hint={ds.available ? `${ds.start} → ${ds.end}` : ds.reason}
          tone="accent"
          to="/data-observatory"
        />
        <StatCard
          label="All-source overlap"
          value={cov?.available ? formatPct(cov.all_sources_observed?.fraction) : "—"}
          hint={
            cov?.available
              ? `${cov.all_sources_observed?.observed?.toLocaleString()} of ${cov.rows?.toLocaleString()} rows`
              : "Coverage unavailable"
          }
          tone="warning"
          to="/data-observatory"
        />
        <StatCard
          label="Defects tracked"
          value={String(DEFECTS.length)}
          hint={`D-001 … D-020 · ${
            DEFECTS.filter(
              (d) => d.severity === "Critical" && d.status !== "FIXED" && d.status !== "CLOSED",
            ).length
          } open critical`}
          to="/defect-register"
        />
        <StatCard
          label="Latest experiment"
          value={exp?.configuration ?? "—"}
          hint={
            exp?.generated ? `generated ${formatUtc(exp.generated)}` : "no metrics.json"
          }
          to="/research-lab"
        />
      </div>

      <div className="grid items-start gap-4 lg:grid-cols-3">
        <Card title="Repository" description="From git at request time">
          <dl className="space-y-2 text-sm">
            <div className="flex justify-between gap-3">
              <dt className="text-fg-muted">Branch</dt>
              <dd className="font-mono text-xs">{s.repo.branch ?? "unavailable"}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-fg-muted">Commit</dt>
              <dd className="font-mono text-xs">
                {s.repo.short_sha ?? "unavailable"}
              </dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-fg-muted">Working tree</dt>
              <dd>
                {s.repo.dirty ? (
                  <StatusBadge status="WARNING" label="uncommitted changes" />
                ) : (
                  <StatusBadge status="PASS" label="clean" />
                )}
              </dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-fg-muted">API generated</dt>
              <dd className="whitespace-nowrap text-right font-mono text-[11px]">
                {formatUtc(s.generated)}
              </dd>
            </div>
          </dl>
        </Card>

        <Card title="Source availability" description="Same-day observations in the fused frame">
          {cov?.available ? (
            <ul className="space-y-2 text-sm">
              {(
                [
                  ["Station meteorology", cov.sources?.station_met],
                  ["MERRA-2 reanalysis", cov.sources?.merra],
                  ["INSAT-3D AOD", cov.sources?.aod],
                ] as const
              ).map(([label, v]) => (
                <li key={label} className="flex items-center justify-between gap-3">
                  <span className="text-fg-muted">{label}</span>
                  <span className="whitespace-nowrap font-mono text-xs">
                    {(v?.observed ?? 0).toLocaleString()} ({formatPct(v?.fraction)})
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState title="Coverage unavailable" description="Start the API server." />
          )}
        </Card>

        <Card
          title="Track summaries"
          description="Paired differences read from the matrix artefacts"
        >
          {(() => {
            const ablations = matrix.data?.ablations;
            const trackA = paired(findPair(ablations, "A", "M1", "M0"));
            const trackB = paired(findPair(ablations, "B", "M4", "M1"));
            if (!matrix.data?.available || (!trackA && !trackB)) {
              return (
                <EmptyState
                  title="Matrix unavailable"
                  description={matrix.data?.reason ?? "No paired ablations in the artefact."}
                />
              );
            }
            return (
              <div className="space-y-3 text-sm">
                <div>
                  <p className="font-medium text-accent">Track A · historical CPCB</p>
                  <p className="text-xs text-fg-muted">
                    Ridge minus persistence over the full record:{" "}
                    <span className="font-mono text-fg">{trackA?.text ?? "—"}</span>{" "}
                    MAE. A gain of about 3%, and the year-resampled interval
                    reaches zero.
                  </p>
                </div>
                <div>
                  <p className="font-medium text-accent">Track B · multisource</p>
                  <p className="text-xs text-fg-muted">
                    Retrospective only. Full fusion minus the ground-only model:{" "}
                    <span className="font-mono text-fg">{trackB?.text ?? "—"}</span>{" "}
                    MAE on matched dates, and no environmental ablation is
                    distinguishable from zero.
                  </p>
                </div>
              </div>
            );
          })()}
        </Card>
      </div>

      <Card
        title="Research highlights"
        description="Sourced from the defect register and phase reports"
      >
        <ul className="space-y-2 text-sm text-fg-muted">
          <li>
            • The fused dataset preserves all ground rows and never fills an
            environmental value; missing values remain{" "}
            <code className="text-fg">NaN</code> with per-source observation age.
          </li>
          <li>
            • Satellite and reanalysis overlap the ground record on a small
            minority of days confined to 2024, so any Track B claim is a
            matched-date comparison, stated as such.
          </li>
          <li>
            • The five-day exercise is retrospective: future observed AOD and
            MERRA-2 are forbidden inputs under the availability contract.
          </li>
          <li>
            • A corrupt station pressure column (D-020) was found by
            cross-validating MERRA-2 against the station, not the station against
            itself.
          </li>
        </ul>
      </Card>
    </div>
  );
}