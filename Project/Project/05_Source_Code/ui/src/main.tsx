import { StrictMode, Suspense, lazy } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider } from "@/lib/theme";
import { AppShell } from "@/components/layout/AppShell";
import { EmptyState, LoadingBlock } from "@/components/ui/primitives";
import "./styles.css";

/**
 * Every module is loaded on demand.
 *
 * The application used to ship as one 1,029 kB chunk: the ten modules, the
 * ECharts renderer and the query client all arrived before the first screen
 * could paint. Each route now pulls its own module, so the command centre
 * loads a fraction of that and a chart library is fetched only when a page
 * that draws a chart is opened.
 */
const module = <T,>(loader: () => Promise<T>, pick: (loaded: T) => React.ComponentType) =>
  lazy(async () => ({ default: pick(await loader()) }));

const CommandCenter = module(
  () => import("@/modules/command-center/CommandCenter"),
  (m) => m.CommandCenter,
);
const ResearchLab = module(
  () => import("@/modules/research-lab/ResearchLab"),
  (m) => m.ResearchLab,
);
const DataObservatory = module(
  () => import("@/modules/data-observatory/DataObservatory"),
  (m) => m.DataObservatory,
);
const ValidationMatrix = module(
  () => import("@/modules/validation-matrix/ValidationMatrix"),
  (m) => m.ValidationMatrix,
);
const DefectRegister = module(
  () => import("@/modules/defect-register/DefectRegister"),
  (m) => m.DefectRegister,
);
const ModelIntelligence = module(
  () => import("@/modules/model-intelligence/ModelIntelligence"),
  (m) => m.ModelIntelligence,
);
const ForecastStudio = module(
  () => import("@/modules/forecast-studio/ForecastStudio"),
  (m) => m.ForecastStudio,
);
const AtmosphericExplorer = module(
  () => import("@/modules/atmospheric-explorer/AtmosphericExplorer"),
  (m) => m.AtmosphericExplorer,
);
const Reports = module(
  () => import("@/modules/reports/Reports"),
  (m) => m.Reports,
);
const Settings = module(
  () => import("@/modules/settings/Settings"),
  (m) => m.Settings,
);

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function RouteFallback() {
  return (
    <div role="status" aria-live="polite" aria-busy="true">
      <LoadingBlock rows={4} />
    </div>
  );
}

/** Wrap a lazily loaded module so a route transition always has a fallback. */
function page(node: React.ReactNode) {
  return <Suspense fallback={<RouteFallback />}>{node}</Suspense>;
}

const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: page(<CommandCenter />) },
      { path: "research-lab", element: page(<ResearchLab />) },
      { path: "data-observatory", element: page(<DataObservatory />) },
      { path: "validation-matrix", element: page(<ValidationMatrix />) },
      { path: "defect-register", element: page(<DefectRegister />) },
      { path: "model-intelligence", element: page(<ModelIntelligence />) },
      { path: "forecast-studio", element: page(<ForecastStudio />) },
      { path: "atmospheric-explorer", element: page(<AtmosphericExplorer />) },
      { path: "reports", element: page(<Reports />) },
      { path: "settings", element: page(<Settings />) },
      {
        path: "*",
        element: (
          <EmptyState
            title="Page not found"
            description="This route does not exist. Use the navigation to return to a module."
          />
        ),
      },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <RouterProvider router={router} />
      </ThemeProvider>
    </QueryClientProvider>
  </StrictMode>,
);
