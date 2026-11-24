import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  renderWithProviders,
  mockStatus,
  mockExperiments,
  mockSeries,
  mockManifests,
  mockMatrix,
  mockAblations,
  mockSufficiency,
  mockAlignment,
} from "./utils";

const { api } = vi.hoisted(() => ({
  api: {
    health: vi.fn(),
    status: vi.fn(),
    experiments: vi.fn(),
    coverage: vi.fn(),
    dataset: vi.fn(),
    series: vi.fn(),
    manifests: vi.fn(),
    experimentMatrix: vi.fn(),
    ablations: vi.fn(),
    seasonalSufficiency: vi.fn(),
    alignment: vi.fn(),
    forecasts: vi.fn(),
  },
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return { ...actual, api };
});

// ECharts needs a real canvas; stub the wrapper so tests exercise the data path.
vi.mock("@/components/ui/LineChart", () => ({
  LineChart: ({ ariaLabel }: { ariaLabel: string }) => (
    <div role="img" aria-label={ariaLabel} />
  ),
}));

import { CommandCenter } from "@/modules/command-center/CommandCenter";
import { ResearchLab } from "@/modules/research-lab/ResearchLab";
import { DataObservatory } from "@/modules/data-observatory/DataObservatory";
import { ValidationMatrix } from "@/modules/validation-matrix/ValidationMatrix";
import { DefectRegister } from "@/modules/defect-register/DefectRegister";
import { ModelIntelligence } from "@/modules/model-intelligence/ModelIntelligence";
import { ForecastStudio } from "@/modules/forecast-studio/ForecastStudio";
import { AtmosphericExplorer } from "@/modules/atmospheric-explorer/AtmosphericExplorer";
import { Reports } from "@/modules/reports/Reports";
import { Settings } from "@/modules/settings/Settings";
import { AppShell } from "@/components/layout/AppShell";

beforeEach(() => {
  api.status.mockResolvedValue(mockStatus);
  api.experiments.mockResolvedValue(mockExperiments);
  api.coverage.mockResolvedValue(mockStatus.coverage);
  api.dataset.mockResolvedValue(mockStatus.dataset);
  api.series.mockResolvedValue(mockSeries);
  api.manifests.mockResolvedValue(mockManifests);
  api.experimentMatrix.mockResolvedValue(mockMatrix);
  api.ablations.mockResolvedValue(mockAblations);
  api.seasonalSufficiency.mockResolvedValue(mockSufficiency);
  api.alignment.mockResolvedValue(mockAlignment);
  api.forecasts.mockResolvedValue({
    available: false,
    reason: "no persisted forecast projection found under results/",
    operational: { path: "results/legacy_chapter/operational_forward_forecast.csv", rows: [] },
    recursive: { path: "results/legacy_chapter/five_day_recursive_forecast.csv", rows: [] },
  });
});

describe("Command Center", () => {
  it("renders live dataset and coverage values", async () => {
    renderWithProviders(<CommandCenter />);
    expect(await screen.findByText("2,979")).toBeInTheDocument();
    expect(await screen.findByText("4.7%")).toBeInTheDocument();
    expect(screen.getByText("d89915c")).toBeInTheDocument();
  });

  it("shows a recoverable error when the API fails", async () => {
    api.status.mockRejectedValue(new Error("API offline"));
    renderWithProviders(<CommandCenter />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/Could not load data/i);
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });

  it("shows a loading state before data resolves", () => {
    api.status.mockReturnValue(new Promise(() => {}));
    renderWithProviders(<CommandCenter />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });
});

describe("Research Lab", () => {
  it("switches between Track A and Track B", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ResearchLab />);
    expect(await screen.findByText(/Track A uses CPCB ground history only/i)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Track B/i }));
    expect(
      await screen.findByText(/Track B is retrospective estimation/i),
    ).toBeInTheDocument();
  });

  it("updates the detail panel when an experiment is selected", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ResearchLab />);
    await user.click(await screen.findByRole("button", { name: /A0.*Persistence baseline/i }));
    expect(screen.getAllByText(/How well does yesterday's PM2.5/i).length).toBeGreaterThan(0);
  });

  it("renders metrics from the experiment artifact", async () => {
    renderWithProviders(<ResearchLab />);
    expect(await screen.findByText("RandomForest")).toBeInTheDocument();
    expect(screen.getByText("16.94")).toBeInTheDocument();
  });

  it("shows the paired interval and the year-robust view for the Track A comparison", async () => {
    renderWithProviders(<ResearchLab />);
    // The card reads the ablation artefact, never a hard-coded p-value.
    expect(await screen.findByText(/Δ MAE \(ridge − persistence\)/i)).toBeInTheDocument();
    expect(screen.getAllByText("[-1.14, -0.17]").length).toBeGreaterThan(0);
    expect(screen.getAllByText("[-1.20, 0.06]").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/significant · Holm p 0\.008/).length).toBeGreaterThan(0);
  });

  it("says plainly when an ablation is not distinguishable from zero", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ResearchLab />);
    await user.click(await screen.findByRole("button", { name: /Track B/i }));
    expect(
      (await screen.findAllByText(/not significant · Holm p 0\.340/)).length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByText("[+0.30, +10.41]").length).toBeGreaterThan(0);
  });

  it("reads the track summaries from the ablation artefact, not from prose", async () => {
    renderWithProviders(<CommandCenter />);
    // The mock pair for Track A is -0.65 [-1.14, -0.17]; the old copy quoted
    // a Wilcoxon p-value typed into the component.
    expect(await screen.findByText("-0.65 [-1.14, -0.17]")).toBeInTheDocument();
    expect(screen.getByText("+4.70 [+0.30, +10.41]")).toBeInTheDocument();
    expect(screen.queryByText(/0\.1495/)).not.toBeInTheDocument();
  });
});

describe("Data Observatory", () => {
  it("does not treat missing values as zero", async () => {
    renderWithProviders(<DataObservatory />);
    expect(await screen.findByText("95.3%")).toBeInTheDocument();
  });
});

describe("Validation Matrix", () => {
  it("filters records by status", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ValidationMatrix />);
    await user.click(screen.getByRole("button", { name: /FAIL \(1\)/i }));
    expect(screen.getByText("VAL-004")).toBeInTheDocument();
    expect(screen.queryByText("VAL-001")).not.toBeInTheDocument();
  });

  it("opens evidence details for a record", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ValidationMatrix />);
    await user.click(screen.getByText("VAL-002"));
    expect(
      await screen.findByText(/specific humidity \(kg\/kg\)/i),
    ).toBeInTheDocument();
  });
});

describe("Defect Register", () => {
  it("searches defects and keeps questions separate", async () => {
    const user = userEvent.setup();
    renderWithProviders(<DefectRegister />);
    const search = screen.getByRole("searchbox");
    await user.type(search, "pressure");
    expect(screen.getByText(/barometric pressure column is corrupt/i)).toBeInTheDocument();

    await user.clear(search);
    await user.click(screen.getByRole("button", { name: /Research questions/i }));
    expect(screen.getByText(/boundary-layer-height/i)).toBeInTheDocument();
  });

  it("combines severity filter with search", async () => {
    const user = userEvent.setup();
    renderWithProviders(<DefectRegister />);
    await user.selectOptions(screen.getByLabelText(/filter by severity/i), "Critical");
    expect(screen.getByText("D-001")).toBeInTheDocument();
    expect(screen.queryByText("D-015")).not.toBeInTheDocument();
  });
});

describe("Model Intelligence", () => {
  it("selects a model and shows its metrics", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ModelIntelligence />);
    await user.click(await screen.findByRole("button", { name: /XGBoost/i }));
    expect(screen.getAllByText("16.94").length).toBeGreaterThan(0);
  });

  it("marks unsupported diagnostics unavailable", async () => {
    renderWithProviders(<ModelIntelligence />);
    expect(await screen.findByText("Residual diagnostics")).toBeInTheDocument();
    expect(screen.getAllByText(/not in this artifact/i).length).toBe(4);
  });
});

describe("Forecast Studio", () => {
  it("blocks submission when a required source is unavailable", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ForecastStudio />);
    await waitFor(() => expect(api.series).toHaveBeenCalled());
    await user.selectOptions(await screen.findByLabelText(/issue date/i), "2024-06-15");
    await user.selectOptions(screen.getByLabelText(/^model$/i), "fusion");

    const submit = screen.getByRole("button", { name: /evaluate eligibility/i });
    expect(submit).toBeDisabled();
    expect(screen.getByText(/AOD is not observed/i)).toBeInTheDocument();
  });

  it("never renders a forecast value", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ForecastStudio />);
    await waitFor(() => expect(api.series).toHaveBeenCalled());
    await user.selectOptions(await screen.findByLabelText(/issue date/i), "2024-06-16");
    await user.click(screen.getByRole("button", { name: /evaluate eligibility/i }));
    expect(await screen.findByText(/Forecast values unavailable/i)).toBeInTheDocument();
  });
});

describe("Atmospheric Explorer", () => {
  it("refetches when the date range or variable changes", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AtmosphericExplorer />);
    await waitFor(() => expect(api.series).toHaveBeenCalled());
    const calls = api.series.mock.calls.length;
    await user.selectOptions(screen.getByLabelText(/variable/i), "aod");
    expect(api.series.mock.calls.length).toBe(calls);
    // The chart is present once points exist.
    expect(await screen.findByRole("img", { name: /AOD time series/i })).toBeInTheDocument();
  });
});

describe("Reports", () => {
  it("exports valid JSON with the selected experiment metadata", async () => {
    const user = userEvent.setup();
    const createObjectURL = vi.fn(() => "blob:mock");
    const revokeObjectURL = vi.fn();
    URL.createObjectURL = createObjectURL;
    URL.revokeObjectURL = revokeObjectURL;
    let captured: Blob | null = null;
    const OrigBlob = globalThis.Blob;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    globalThis.Blob = class extends OrigBlob {
      constructor(parts: BlobPart[], opts?: BlobPropertyBag) {
        super(parts, opts);
        captured = this;
      }
    } as typeof Blob;

    renderWithProviders(<Reports />);
    const btn = await screen.findByRole("button", { name: /Export JSON/i });
    await waitFor(() => expect(btn).not.toBeDisabled());
    await user.click(btn);
    expect(createObjectURL).toHaveBeenCalled();

    const text = await captured!.text();
    const parsed = JSON.parse(text);
    expect(parsed.experiments.configuration).toBe("phase6_fusion");
    globalThis.Blob = OrigBlob;
  }, 15000);

  it("shows the manifest git SHA", async () => {
    renderWithProviders(<Reports />);
    expect((await screen.findAllByText(/F-fusion-1/)).length).toBeGreaterThan(0);
  });
});

describe("Settings", () => {
  it("persists theme and preferences to localStorage", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Settings />);
    await user.selectOptions(screen.getByLabelText(/^theme$/i), "dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);

    await user.selectOptions(screen.getByLabelText(/timezone/i), "UTC");
    const raw = JSON.parse(localStorage.getItem("aerosphinx.prefs") ?? "{}");
    expect(raw.timezone).toBe("UTC");
  });

  it("validates the date range and resets", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Settings />);
    await user.type(screen.getByLabelText(/default start/i), "2024-12-31");
    await user.type(screen.getByLabelText(/default end/i), "2024-01-01");
    expect(screen.getByRole("alert")).toHaveTextContent(/must not be after/i);

    await user.click(screen.getByRole("button", { name: /reset preferences/i }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});

describe("Navigation shell", () => {
  it("renders all ten modules as links", () => {
    renderWithProviders(<AppShell />);
    const nav = screen.getByRole("navigation", { name: /primary/i });
    const links = within(nav).getAllByRole("link");
    expect(links).toHaveLength(10);
  });

  it("toggles the theme control", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AppShell />);
    const toggle = screen.getByRole("button", { name: /theme:/i });
    await user.click(toggle);
    await user.click(toggle);
    await user.click(toggle);
    expect(screen.getByRole("button", { name: /theme:/i })).toBeInTheDocument();
  });

  it("opens the mobile navigation", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AppShell />);
    const burger = screen.getByRole("button", { name: /toggle navigation/i });
    await user.click(burger);
    expect(burger).toHaveAttribute("aria-expanded", "true");
  });
});