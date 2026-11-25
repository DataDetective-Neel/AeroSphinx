import { useEffect, useState } from "react";
import { useTheme } from "@/lib/theme";
import { useStatus } from "@/lib/queries";
import { STATION } from "@/lib/reference";
import {
  Card,
  PageHeader,
  StatusBadge,
  formatUtc,
} from "@/components/ui/primitives";

interface Prefs {
  timezone: string;
  defaultStation: string;
  defaultStart: string;
  defaultEnd: string;
  showGrid: boolean;
  animateCharts: boolean;
  dateFormat: "iso" | "locale";
}

const DEFAULT_PREFS: Prefs = {
  timezone: "Asia/Kolkata",
  defaultStation: STATION.name,
  defaultStart: "",
  defaultEnd: "",
  showGrid: true,
  animateCharts: false,
  dateFormat: "iso",
};

const STORAGE_KEY = "aerosphinx.prefs";

function loadPrefs(): Prefs {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_PREFS;
    const parsed = JSON.parse(raw) as Partial<Prefs>;
    return { ...DEFAULT_PREFS, ...parsed };
  } catch {
    return DEFAULT_PREFS;
  }
}

export function Settings() {
  const { theme, setTheme } = useTheme();
  const status = useStatus();
  const [prefs, setPrefs] = useState<Prefs>(loadPrefs);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
  }, [prefs]);

  function update<K extends keyof Prefs>(key: K, value: Prefs[K]) {
    setPrefs((p) => ({ ...p, [key]: value }));
    setSaved(true);
  }

  function reset() {
    setPrefs(DEFAULT_PREFS);
    setTheme("system");
    setSaved(true);
  }

  const datesValid =
    !prefs.defaultStart ||
    !prefs.defaultEnd ||
    prefs.defaultStart <= prefs.defaultEnd;

  const apiOk = status.isSuccess;
  const apiError = status.isError ? (status.error as Error).message : null;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Settings"
        tagline="Preferences are stored locally in your browser. Nothing here changes scientific behaviour."
      />

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Appearance">
          <div className="space-y-3">
            <div>
              <label htmlFor="theme" className="text-xs font-medium text-fg-muted">
                Theme
              </label>
              <select
                id="theme"
                value={theme}
                onChange={(e) => {
                  setTheme(e.target.value as "light" | "dark" | "system");
                  setSaved(true);
                }}
                className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-1.5 text-sm"
              >
                <option value="light">Light</option>
                <option value="dark">Dark</option>
                <option value="system">System</option>
              </select>
            </div>
            <div>
              <label htmlFor="dateFormat" className="text-xs font-medium text-fg-muted">
                Date format
              </label>
              <select
                id="dateFormat"
                value={prefs.dateFormat}
                onChange={(e) =>
                  update("dateFormat", e.target.value as "iso" | "locale")
                }
                className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-1.5 text-sm"
              >
                <option value="iso">ISO (2024-05-17)</option>
                <option value="locale">Locale (17/05/2024)</option>
              </select>
            </div>
          </div>
        </Card>

        <Card title="Data defaults">
          <div className="space-y-3">
            <div>
              <label htmlFor="tz" className="text-xs font-medium text-fg-muted">
                Timezone
              </label>
              <select
                id="tz"
                value={prefs.timezone}
                onChange={(e) => update("timezone", e.target.value)}
                className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-1.5 text-sm"
              >
                <option value="Asia/Kolkata">Asia/Kolkata (IST)</option>
                <option value="UTC">UTC</option>
                <option value="America/New_York">America/New_York</option>
                <option value="Europe/London">Europe/London</option>
              </select>
            </div>
            <div>
              <label htmlFor="station" className="text-xs font-medium text-fg-muted">
                Default station
              </label>
              <select
                id="station"
                value={prefs.defaultStation}
                onChange={(e) => update("defaultStation", e.target.value)}
                className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-1.5 text-sm"
              >
                <option value={STATION.name}>{STATION.name}</option>
              </select>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label htmlFor="dstart" className="text-xs font-medium text-fg-muted">
                  Default start
                </label>
                <input
                  id="dstart"
                  type="date"
                  value={prefs.defaultStart}
                  onChange={(e) => update("defaultStart", e.target.value)}
                  className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-1.5 text-sm"
                />
              </div>
              <div>
                <label htmlFor="dend" className="text-xs font-medium text-fg-muted">
                  Default end
                </label>
                <input
                  id="dend"
                  type="date"
                  value={prefs.defaultEnd}
                  onChange={(e) => update("defaultEnd", e.target.value)}
                  className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-1.5 text-sm"
                />
              </div>
            </div>
            {!datesValid && (
              <p className="text-xs font-medium text-danger" role="alert">
                The default start date must not be after the default end date.
              </p>
            )}
          </div>
        </Card>

        <Card title="Charts">
          <div className="space-y-3">
            <label className="flex items-center justify-between gap-3 text-sm">
              <span>Show grid lines</span>
              <input
                type="checkbox"
                checked={prefs.showGrid}
                onChange={(e) => update("showGrid", e.target.checked)}
                className="h-4 w-4"
              />
            </label>
            <label className="flex items-center justify-between gap-3 text-sm">
              <span>Animate charts</span>
              <input
                type="checkbox"
                checked={prefs.animateCharts}
                onChange={(e) => update("animateCharts", e.target.checked)}
                className="h-4 w-4"
              />
            </label>
            <p className="text-xs text-fg-muted">
              Animation is off by default so charts never suggest motion in data
              that is static.
            </p>
          </div>
        </Card>

        <Card title="Application & API">
          <dl className="space-y-2 text-sm">
            <div className="flex justify-between">
              <dt className="text-fg-muted">App version</dt>
              <dd className="font-mono text-xs">0.1.0</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-fg-muted">API connection</dt>
              <dd>
                {apiOk ? (
                  <StatusBadge status="PASS" label="connected" />
                ) : (
                  <StatusBadge status="UNAVAILABLE" label="unavailable" />
                )}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-fg-muted">Last sync</dt>
              <dd className="font-mono text-[11px]">
                {formatUtc(status.data?.generated)}
              </dd>
            </div>
          </dl>
          {apiError && (
            <p className="mt-2 text-xs text-danger">{apiError}</p>
          )}
          {!apiOk && !apiError && status.isLoading && (
            <p className="mt-2 text-xs text-fg-muted">Checking…</p>
          )}
          <p className="mt-3 text-xs text-fg-muted">
            Start the API with{" "}
            <code className="text-fg">python api/server.py</code> and run the
            frontend with <code className="text-fg">npm run dev</code>.
          </p>
        </Card>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={reset}
          className="rounded-md border border-danger/40 px-4 py-2 text-sm font-medium text-danger hover:bg-danger/5"
        >
          Reset preferences
        </button>
        <span className="text-xs text-fg-muted" role="status">
          {saved ? "Saved locally." : "Changes save automatically."}
        </span>
      </div>
    </div>
  );
}