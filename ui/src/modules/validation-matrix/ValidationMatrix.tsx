import { useMemo, useState } from "react";
import {
  VALIDATION_RECORDS,
  type ValidationStatus,
} from "@/lib/reference";
import {
  Card,
  EmptyState,
  PageHeader,
  StatusBadge,
} from "@/components/ui/primitives";

const FILTERS: Array<ValidationStatus | "ALL"> = [
  "ALL",
  "PASS",
  "WARNING",
  "FAIL",
  "NOT_COMPARABLE",
  "INSUFFICIENT_EVIDENCE",
];

export function ValidationMatrix() {
  const [filter, setFilter] = useState<ValidationStatus | "ALL">("ALL");
  const [selected, setSelected] = useState<string | null>(
    VALIDATION_RECORDS[0]?.id ?? null,
  );

  const rows = useMemo(
    () =>
      filter === "ALL"
        ? VALIDATION_RECORDS
        : VALIDATION_RECORDS.filter((r) => r.status === filter),
    [filter],
  );
  const current = VALIDATION_RECORDS.find((r) => r.id === selected) ?? null;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Validation Matrix"
        tagline="Cross-source comparisons transcribed from the Phase 4 and Phase 5 validation reports. Statuses reflect the recorded evidence."
      />

      <div className="flex flex-wrap gap-2">
        {FILTERS.map((f) => {
          const count =
            f === "ALL"
              ? VALIDATION_RECORDS.length
              : VALIDATION_RECORDS.filter((r) => r.status === f).length;
          return (
            <button
              key={f}
              onClick={() => setFilter(f)}
              aria-pressed={filter === f}
              className={`rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
                filter === f
                  ? "border-accent/50 bg-accent/10 text-accent"
                  : "border-line text-fg-muted hover:bg-surface-raised"
              }`}
            >
              {f.replaceAll("_", " ")} ({count})
            </button>
          );
        })}
      </div>

      <Card>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-line text-left text-[11px] uppercase tracking-wider text-fg-muted">
                <th className="py-2 pr-3">ID</th>
                <th className="py-2 pr-3">Source 1</th>
                <th className="py-2 pr-3">Source 2</th>
                <th className="py-2 pr-3">Variable pair</th>
                <th className="py-2 pr-3">n</th>
                <th className="py-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr
                  key={r.id}
                  onClick={() => setSelected(r.id)}
                  className={`cursor-pointer border-b border-line/60 transition-colors hover:bg-surface-raised ${
                    selected === r.id ? "bg-accent/5" : ""
                  }`}
                >
                  <td className="py-2 pr-3 font-mono text-xs">{r.id}</td>
                  <td className="py-2 pr-3">{r.source1}</td>
                  <td className="py-2 pr-3">{r.source2}</td>
                  <td className="py-2 pr-3 text-xs">{r.variablePair}</td>
                  <td className="py-2 pr-3 font-mono text-xs">
                    {r.sampleSize ?? "—"}
                  </td>
                  <td className="py-2">
                    <StatusBadge status={r.status} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {rows.length === 0 && (
          <div className="mt-3">
            <EmptyState title="No records match this filter" />
          </div>
        )}
      </Card>

      {current && (
        <Card
          title={`${current.id} · ${current.variablePair}`}
          actions={<StatusBadge status={current.status} />}
        >
          <div className="grid gap-5 md:grid-cols-2">
            <dl className="space-y-2 text-sm">
              <div>
                <dt className="text-xs text-fg-muted">Source 1</dt>
                <dd className="font-medium">{current.source1}</dd>
              </div>
              <div>
                <dt className="text-xs text-fg-muted">Source 2</dt>
                <dd className="font-medium">{current.source2}</dd>
              </div>
              <div>
                <dt className="text-xs text-fg-muted">Methodology</dt>
                <dd>{current.methodology}</dd>
              </div>
              <div>
                <dt className="text-xs text-fg-muted">Tolerance</dt>
                <dd className="font-mono text-xs">{current.tolerance}</dd>
              </div>
              <div>
                <dt className="text-xs text-fg-muted">Sample size</dt>
                <dd className="font-mono text-xs">
                  {current.sampleSize ?? "unavailable"}
                </dd>
              </div>
            </dl>
            <div>
              <p className="text-xs font-medium text-fg-muted">Evidence</p>
              <p className="mt-1 text-sm">{current.evidence}</p>
              <p className="mt-3 text-xs font-medium text-fg-muted">Notes</p>
              <p className="mt-1 text-sm text-fg-muted">{current.notes}</p>
            </div>
          </div>
        </Card>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        <Card title="B-039 · Station BP investigation">
          <p className="text-sm text-fg-muted">
            <span className="font-medium text-fg">Hypothesis (open):</span> the
            station pressure column is affected by a systematic transformation
            rather than sporadic instrument faults. The diurnal phase is
            inverted (max 10h, min 17h) while the station temperature column is
            correctly phased. This is presented as a hypothesis, not an
            established finding.
          </p>
        </Card>
        <Card title="B-040 · Cross-source validation">
          <p className="text-sm text-fg-muted">
            The corrupt pressure column was found by validating MERRA-2 against
            the station — a different group from the one being validated.
            Whether every source should be cross-validated against every other
            remains an open question (B-040).
          </p>
        </Card>
      </div>
    </div>
  );
}