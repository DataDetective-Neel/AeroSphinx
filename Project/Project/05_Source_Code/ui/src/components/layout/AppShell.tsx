import { useEffect, useRef, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useTheme } from "@/lib/theme";

interface NavItem {
  to: string;
  label: string;
  icon: string;
}

const NAV: NavItem[] = [
  { to: "/", label: "Command Center", icon: "M3 12l9-9 9 9M5 10v10h14V10" },
  { to: "/research-lab", label: "Research Lab", icon: "M9 3v6l-6 10h18L15 9V3M9 3h6" },
  { to: "/data-observatory", label: "Data Observatory", icon: "M12 3a9 9 0 100 18 9 9 0 000-18zM3 12h18M12 3c3 3 3 15 0 18" },
  { to: "/validation-matrix", label: "Validation Matrix", icon: "M3 3h18v18H3zM3 9h18M9 3v18" },
  { to: "/defect-register", label: "Defect Register", icon: "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 104 0M12 11v4M12 17h.01" },
  { to: "/model-intelligence", label: "Model Intelligence", icon: "M12 2l3 7h7l-5.5 4.5L18 21l-6-4-6 4 1.5-7.5L2 9h7z" },
  { to: "/forecast-studio", label: "Forecast Studio", icon: "M3 17l6-6 4 4 8-8M14 7h7v7" },
  { to: "/atmospheric-explorer", label: "Atmospheric Explorer", icon: "M2 12h20M12 2v20M4 4l16 16M20 4L4 20" },
  { to: "/reports", label: "Reports & Reproducibility", icon: "M6 2h9l5 5v15H6zM15 2v5h5M9 13h6M9 17h6" },
  { to: "/settings", label: "Settings", icon: "M12 15a3 3 0 100-6 3 3 0 000 6zM19.4 15a1.6 1.6 0 00.3 1.8l.1.1a2 2 0 11-2.8 2.8l-.1-.1a1.6 1.6 0 00-1.8-.3 1.6 1.6 0 00-1 1.5V21a2 2 0 11-4 0v-.1a1.6 1.6 0 00-1-1.5 1.6 1.6 0 00-1.8.3l-.1.1a2 2 0 11-2.8-2.8l.1-.1a1.6 1.6 0 00.3-1.8 1.6 1.6 0 00-1.5-1H3a2 2 0 110-4h.1a1.6 1.6 0 001.5-1 1.6 1.6 0 00-.3-1.8l-.1-.1a2 2 0 112.8-2.8l.1.1a1.6 1.6 0 001.8.3H9a1.6 1.6 0 001-1.5V3a2 2 0 114 0v.1a1.6 1.6 0 001 1.5 1.6 1.6 0 001.8-.3l.1-.1a2 2 0 112.8 2.8l-.1.1a1.6 1.6 0 00-.3 1.8V9a1.6 1.6 0 001.5 1H21a2 2 0 110 4h-.1a1.6 1.6 0 00-1.5 1z" },
];

function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const next = theme === "light" ? "dark" : theme === "dark" ? "system" : "light";
  const label =
    theme === "light" ? "Light" : theme === "dark" ? "Dark" : "System";
  return (
    <button
      onClick={() => setTheme(next)}
      className="rounded-md border border-line px-2.5 py-1.5 text-xs font-medium text-fg-muted transition-colors hover:bg-surface-raised hover:text-fg"
      aria-label={`Theme: ${label}. Click to switch to ${next}.`}
    >
      {label}
    </button>
  );
}

export function AppShell() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();
  const current = NAV.find((n) => n.to === location.pathname);
  const mainRef = useRef<HTMLElement>(null);

  // A drawer that cannot be dismissed from the keyboard traps the reader.
  useEffect(() => {
    if (!mobileOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobileOpen(false);
    };
    window.addEventListener("keydown", onKey);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = previousOverflow;
    };
  }, [mobileOpen]);

  // Moving between modules should start at the top of the new page, not at the
  // scroll position of the previous one.
  useEffect(() => {
    mainRef.current?.scrollTo?.({ top: 0 });
    window.scrollTo({ top: 0 });
  }, [location.pathname]);

  return (
    <div className="flex min-h-screen bg-bg text-fg">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 w-64 shrink-0 overflow-y-auto border-r border-line bg-surface transition-transform md:static md:translate-x-0 ${
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex h-16 items-center gap-2.5 border-b border-line px-4">
          <div
            className="flex h-9 w-9 items-center justify-center rounded-md bg-accent"
            aria-hidden="true"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-surface">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5" />
            </svg>
          </div>
          <div className="leading-tight">
            <p className="text-sm font-semibold tracking-tight">AeroSphinx</p>
            <p className="text-[10px] uppercase tracking-wider text-fg-muted">
              Atmospheric Intelligence
            </p>
          </div>
        </div>

        <nav className="p-2" aria-label="Primary navigation">
          <ul className="space-y-0.5">
            {NAV.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  onClick={() => setMobileOpen(false)}
                  className={({ isActive }) =>
                    `flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors ${
                      isActive
                        ? "bg-accent/10 font-medium text-accent"
                        : "text-fg-muted hover:bg-surface-raised hover:text-fg"
                    }`
                  }
                >
                  <svg
                    width="17"
                    height="17"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="shrink-0"
                    aria-hidden="true"
                  >
                    <path d={item.icon} />
                  </svg>
                  <span className="truncate">{item.label}</span>
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      </aside>

      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/40 md:hidden"
          onClick={() => setMobileOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Main */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex h-16 items-center justify-between gap-3 border-b border-line bg-surface/90 px-4 backdrop-blur">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setMobileOpen((v) => !v)}
              className="rounded-md border border-line p-1.5 text-fg-muted hover:bg-surface-raised md:hidden"
              aria-label="Toggle navigation"
              aria-expanded={mobileOpen}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 6h18M3 12h18M3 18h18" />
              </svg>
            </button>
            <div>
              <p className="text-[11px] uppercase tracking-wider text-fg-muted">
                AeroSphinx
              </p>
              <h1 className="text-sm font-semibold tracking-tight">
                {current?.label ?? "Not found"}
              </h1>
            </div>
          </div>
          <ThemeToggle />
        </header>

        <main
          id="main-content"
          ref={mainRef}
          tabIndex={-1}
          className="mx-auto w-full max-w-7xl flex-1 p-4 md:p-6 lg:p-8 focus:outline-none"
        >
          <Outlet />
        </main>

        <footer className="border-t border-line px-4 py-3 text-center text-[11px] text-fg-muted">
          AeroSphinx · Understand the atmosphere. Evaluate the evidence. · Data
          from CPCB, ISRO/SAC, NASA GMAO.
        </footer>
      </div>
    </div>
  );
}
