import type { ReactNode } from "react";

export function PageHeader({
  title,
  tagline,
  actions,
}: {
  title: string;
  tagline?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h2 className="text-xl font-semibold tracking-tight md:text-2xl">
          {title}
        </h2>
        {tagline && <p className="mt-1 text-sm text-fg-muted">{tagline}</p>}
      </div>
      {actions}
    </div>
  );
}

export function Card({
  title,
  description,
  actions,
  children,
  className = "",
}: {
  title?: string;
  description?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-lg border border-line bg-surface p-4 md:p-5 ${className}`}
    >
      {(title || actions) && (
        <div className="mb-3 flex items-start justify-between gap-3">
          <div>
            {title && (
              <h3 className="text-sm font-semibold tracking-tight">{title}</h3>
            )}
            {description && (
              <p className="mt-0.5 text-xs text-fg-muted">{description}</p>
            )}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

export function StatCard({
  label,
  value,
  hint,
  tone = "default",
  to,
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: "default" | "success" | "warning" | "danger" | "accent";
  to?: string;
}) {
  const toneClass = {
    default: "text-fg",
    success: "text-success",
    warning: "text-warning",
    danger: "text-danger",
    accent: "text-accent",
  }[tone];
  const body = (
    <div className="rounded-lg border border-line bg-surface p-4 transition-colors hover:border-accent/40">
      <p className="text-[11px] font-medium uppercase tracking-wider text-fg-muted">
        {label}
      </p>
      <p className={`mt-1.5 text-2xl font-semibold tracking-tight ${toneClass}`}>
        {value}
      </p>
      {hint && <p className="mt-1 text-xs text-fg-muted">{hint}</p>}
    </div>
  );
  if (to) {
    return (
      <a href={to} className="block focus:outline-none">
        {body}
      </a>
    );
  }
  return body;
}

export function StatusBadge({
  status,
  label,
}: {
  status: string;
  label?: string;
}) {
  const s = status.toUpperCase();
  let cls = "border-line bg-surface-raised text-fg-muted";
  if (["PASS", "FIXED", "CLOSED", "RESOLVED", "ANSWERED", "DECIDED", "VALIDATED", "AVAILABLE", "SUFFICIENT"].includes(s))
    cls = "border-success/30 bg-success/10 text-success";
  else if (["WARNING", "PARTIALLY_RESOLVED", "PARTIALLY_ADDRESSED", "QUANTIFIED", "ADDRESSED", "DEGRADED", "ABSTAINED"].includes(s))
    cls = "border-warning/30 bg-warning/10 text-warning";
  else if (["FAIL", "OPEN", "UNSUPPORTED", "UNAVAILABLE", "NOT_COMPARABLE", "INSUFFICIENT_EVIDENCE", "INSUFFICIENT", "NOT_EVALUABLE"].includes(s))
    cls = "border-danger/30 bg-danger/10 text-danger";

  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium ${cls}`}
    >
      {label ?? status.replaceAll("_", " ")}
    </span>
  );
}

export function EmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="rounded-lg border border-dashed border-line bg-surface p-8 text-center">
      <p className="text-sm font-medium">{title}</p>
      {description && (
        <p className="mx-auto mt-1 max-w-md text-xs text-fg-muted">{description}</p>
      )}
      {action && <div className="mt-3">{action}</div>}
    </div>
  );
}

export function ErrorState({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div
      className="rounded-lg border border-danger/40 bg-danger/5 p-4"
      role="alert"
    >
      <p className="text-sm font-medium text-danger">Could not load data</p>
      <p className="mt-1 text-xs text-fg-muted">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-3 rounded-md border border-line px-3 py-1.5 text-xs font-medium hover:bg-surface-raised"
        >
          Retry
        </button>
      )}
    </div>
  );
}

export function UnavailableState({ reason }: { reason: string }) {
  return (
    <div className="rounded-lg border border-dashed border-line bg-surface-raised p-4">
      <p className="text-xs font-medium text-fg-muted">Unavailable</p>
      <p className="mt-1 text-xs text-fg-muted">{reason}</p>
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return (
    <div
      className={`animate-pulse rounded-md bg-surface-raised ${className}`}
      aria-hidden="true"
    />
  );
}

export function LoadingBlock({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-2" aria-label="Loading" role="status">
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className="h-10 w-full" />
      ))}
      <span className="sr-only">Loading…</span>
    </div>
  );
}

export function Metric({
  label,
  value,
  unit,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
}) {
  return (
    <div>
      <dt className="text-xs text-fg-muted">{label}</dt>
      <dd className="text-sm font-medium">
        {value}
        {unit && <span className="ml-0.5 text-xs text-fg-muted">{unit}</span>}
      </dd>
    </div>
  );
}

export function formatNumber(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

/**
 * A platform timestamp is always UTC. Render it as "2026-11-23 16:17 UTC" so a
 * reader never has to decode an ISO string, and never mistakes it for local
 * time (the feed is UTC; the reader may be anywhere).
 */
export function formatUtc(iso: string | null | undefined): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const pad = (value: number) => String(value).padStart(2, "0");
  return (
    `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())}` +
    ` ${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())} UTC`
  );
}

export function formatPct(fraction: number | null | undefined, digits = 1) {
  if (fraction === null || fraction === undefined) return "—";
  return `${(fraction * 100).toFixed(digits)}%`;
}