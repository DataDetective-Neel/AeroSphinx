import { useMemo, useState } from "react";
import {
  DEFECTS,
  RESEARCH_QUESTIONS,
  type Defect,
} from "@/lib/reference";
import {
  Card,
  EmptyState,
  PageHeader,
  StatusBadge,
} from "@/components/ui/primitives";

type View = "defects" | "questions";
type SortKey = "id" | "severity" | "status";

const PAGE_SIZE = 8;
const SEVERITIES = ["ALL", "Critical", "High", "Medium", "Low"] as const;

export function DefectRegister() {
  const [view, setView] = useState<View>("defects");
  const [search, setSearch] = useState("");
  const [severity, setSeverity] = useState<(typeof SEVERITIES)[number]>("ALL");
  const [sortKey, setSortKey] = useState<SortKey>("id");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<Defect | null>(DEFECTS[0] ?? null);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return DEFECTS.filter((d) => {
      const matchesQ =
        !q ||
        d.id.toLowerCase().includes(q) ||
        d.title.toLowerCase().includes(q) ||
        d.impact.toLowerCase().includes(q);
      const matchesSev = severity === "ALL" || d.severity === severity;
      return matchesQ && matchesSev;
    }).sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      const cmp = String(av).localeCompare(String(bv));
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [search, severity, sortKey, sortDir]);

  const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const currentPage = Math.min(page, pages);
  const paged = filtered.slice(
    (currentPage - 1) * PAGE_SIZE,
    currentPage * PAGE_SIZE,
  );

  const questions = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return RESEARCH_QUESTIONS;
    return RESEARCH_QUESTIONS.filter(
      (r) =>
        r.id.toLowerCase().includes(q) || r.question.toLowerCase().includes(q),
    );
  }, [search]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir("asc");
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Defect Register"
        tagline="Engineering defects and open research questions are kept deliberately separate."
      />

      <div className="inline-flex rounded-lg border border-line bg-surface p-1">
        {(
          [
            ["defects", `Engineering defects (${DEFECTS.length})`],
            ["questions", `Research questions (${RESEARCH_QUESTIONS.length})`],
          ] as const
        ).map(([v, label]) => (
          <button
            key={v}
            onClick={() => setView(v)}
            aria-pressed={view === v}
            className={`rounded-md px-4 py-1.5 text-sm font-medium transition-colors ${
              view === v ? "bg-accent text-surface" : "text-fg-muted hover:text-fg"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <input
          type="search"
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(1);
          }}
          placeholder="Search by ID, title or impact…"
          aria-label="Search register"
          className="min-w-52 flex-1 rounded-md border border-line bg-surface px-3 py-1.5 text-sm placeholder:text-fg-muted focus:outline-none"
        />
        {view === "defects" && (
          <select
            value={severity}
            onChange={(e) => {
              setSeverity(e.target.value as (typeof SEVERITIES)[number]);
              setPage(1);
            }}
            aria-label="Filter by severity"
            className="rounded-md border border-line bg-surface px-3 py-1.5 text-sm"
          >
            {SEVERITIES.map((s) => (
              <option key={s} value={s}>
                {s === "ALL" ? "All severities" : s}
              </option>
            ))}
          </select>
        )}
      </div>

      {view === "defects" ? (
        <>
          <Card>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-line text-left text-[11px] uppercase tracking-wider text-fg-muted">
                    {(
                      [
                        ["id", "ID"],
                        ["severity", "Severity"],
                        ["status", "Status"],
                      ] as const
                    ).map(([k, label]) => (
                      <th key={k} className="py-2 pr-3">
                        <button
                          onClick={() => toggleSort(k)}
                          className="inline-flex items-center gap-1 hover:text-fg"
                        >
                          {label}
                          {sortKey === k && (sortDir === "asc" ? "▲" : "▼")}
                        </button>
                      </th>
                    ))}
                    <th className="py-2">Title</th>
                  </tr>
                </thead>
                <tbody>
                  {paged.map((d) => (
                    <tr
                      key={d.id}
                      onClick={() => setSelected(d)}
                      className={`cursor-pointer border-b border-line/60 transition-colors hover:bg-surface-raised ${
                        selected?.id === d.id ? "bg-accent/5" : ""
                      }`}
                    >
                      <td className="py-2 pr-3 font-mono text-xs">{d.id}</td>
                      <td className="py-2 pr-3">
                        <StatusBadge
                          status={d.severity === "Low" ? "PASS" : d.severity === "Medium" ? "WARNING" : "FAIL"}
                          label={d.severity}
                        />
                      </td>
                      <td className="py-2 pr-3">
                        <StatusBadge status={d.status} />
                      </td>
                      <td className="py-2">{d.title}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {filtered.length === 0 && (
              <div className="mt-3">
                <EmptyState title="No defects match the filters" />
              </div>
            )}
            <div className="mt-3 flex items-center justify-between text-xs text-fg-muted">
              <span>
                {filtered.length === 0
                  ? "0 results"
                  : `${(currentPage - 1) * PAGE_SIZE + 1}–${Math.min(
                      currentPage * PAGE_SIZE,
                      filtered.length,
                    )} of ${filtered.length}`}
              </span>
              <div className="flex gap-1.5">
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="rounded-md border border-line px-2.5 py-1 disabled:opacity-40"
                >
                  Prev
                </button>
                <button
                  onClick={() => setPage((p) => Math.min(pages, p + 1))}
                  disabled={currentPage === pages}
                  className="rounded-md border border-line px-2.5 py-1 disabled:opacity-40"
                >
                  Next
                </button>
              </div>
            </div>
          </Card>

          {selected && (
            <Card
              title={`${selected.id} · ${selected.title}`}
              actions={<StatusBadge status={selected.status} />}
            >
              <div className="grid gap-5 md:grid-cols-2">
                <dl className="space-y-2 text-sm">
                  <div>
                    <dt className="text-xs text-fg-muted">Location</dt>
                    <dd className="font-mono text-xs">{selected.location}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-fg-muted">Severity</dt>
                    <dd>{selected.severity}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-fg-muted">Resolved in</dt>
                    <dd>{selected.resolvedIn}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-fg-muted">Impact</dt>
                    <dd className="text-fg-muted">{selected.impact}</dd>
                  </div>
                </dl>
                <div>
                  <p className="text-xs font-medium text-fg-muted">Evidence</p>
                  <p className="mt-1 text-sm">{selected.evidence}</p>
                  <p className="mt-3 text-xs font-medium text-fg-muted">
                    Resolution
                  </p>
                  <p className="mt-1 text-sm text-fg-muted">
                    {selected.resolution}
                  </p>
                </div>
              </div>
            </Card>
          )}
        </>
      ) : (
        <Card>
          <ul className="divide-y divide-line">
            {questions.map((q) => (
              <li key={q.id} className="flex flex-col gap-1 py-3 sm:flex-row sm:items-start sm:justify-between sm:gap-4">
                <div>
                  <p className="font-mono text-[11px] text-fg-muted">{q.id}</p>
                  <p className="text-sm font-medium">{q.question}</p>
                  <p className="text-xs text-fg-muted">{q.note}</p>
                </div>
                <div className="shrink-0">
                  <StatusBadge status={q.status} />
                </div>
              </li>
            ))}
          </ul>
          {questions.length === 0 && (
            <EmptyState title="No questions match the search" />
          )}
        </Card>
      )}
    </div>
  );
}