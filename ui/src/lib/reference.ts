/**
 * Reference content transcribed from repository documents.
 *
 * These are *documents*, not measurements: they describe defects and open
 * questions recorded in docs/reference/DEFECT_REGISTER.md and experiment definitions
 * from docs/reference/EXPERIMENT_PROTOCOL.md. Measurements and current dataset values
 * always come from the API, never from this file.
 */

export type Severity = "Critical" | "High" | "Medium" | "Low";

export type DefectStatus =
  | "FIXED"
  | "CLOSED"
  | "RESOLVED"
  | "QUANTIFIED"
  | "ADDRESSED"
  | "PARTIALLY_ADDRESSED"
  | "DOCUMENTED";

export interface Defect {
  id: string;
  title: string;
  severity: Severity;
  status: DefectStatus;
  location: string;
  evidence: string;
  impact: string;
  resolution: string;
  resolvedIn: string;
}

export const DEFECTS: Defect[] = [
  {
    id: "D-001",
    title: "Row-based PM2.5 lags are not calendar-aware",
    severity: "Critical",
    status: "FIXED",
    location: "legacy/aerosphinx_pipeline.py:427-430",
    evidence:
      "123 of 2,979 rows have PM25_Lag_1 not from the previous calendar day; worst case 25 days old (2017-10-09 → 2017-11-03).",
    impact:
      "Inflates apparent temporal skill wherever the series is gappy.",
    resolution:
      "Calendar-aware lags in src/aerosphinx/features/lags.py; a missing date becomes explicit NaN.",
    resolvedIn: "Phase 1",
  },
  {
    id: "D-002",
    title: "Forecast consumes future observed satellite/reanalysis data",
    severity: "Critical",
    status: "CLOSED",
    location: "legacy/aerosphinx_pipeline.py:774",
    evidence:
      "recursive_forecast read the target day's row (observed AOD and MERRA-2) directly.",
    impact:
      "Presented retrospective estimation as operational forecasting.",
    resolution:
      "Availability contract enforced by construction in multi_horizon.py; the five-day exercise is documented as retrospective.",
    resolvedIn: "Phase 8",
  },
  {
    id: "D-003",
    title: "Forward-fill manufactures environmental observations",
    severity: "High",
    status: "FIXED",
    location: "legacy/aerosphinx_pipeline.py:407-408",
    evidence:
      "7 of 166 modelling rows carried a forward-filled AOD value; 1 forward-filled MERRA-2.",
    impact: "A stale value presented as a contemporaneous observation.",
    resolution:
      "Explicit availability mask in src/aerosphinx/features/availability.py; missing values stay NaN.",
    resolvedIn: "Phase 6",
  },
  {
    id: "D-004",
    title: "27 INSAT granules contribute zero valid pixels",
    severity: "High",
    status: "QUANTIFIED",
    location: "src/aerosphinx/ingestion/aod.py",
    evidence:
      "27 of 169 granules yield no valid pixel inside the station window; median granule contributes 12 valid pixels.",
    impact:
      "A single-pixel AOD is not a station-representative measurement.",
    resolution:
      "AOD group carries aod_valid_pixels and aod_overpasses; coverage reported per date.",
    resolvedIn: "Phase 4",
  },
  {
    id: "D-005",
    title: "Climatology baseline is mis-specified",
    severity: "High",
    status: "FIXED",
    location: "legacy/aerosphinx_pipeline.py:632-638",
    evidence:
      "Baseline predicted mean 64.78 against observed 100.63 (bias -35.85, MAE 35.87, R² -2.68).",
    impact: "Flattered every other model by comparison.",
    resolution:
      "ClimatologyBuilder with per-fold lookback selected on an inner validation block.",
    resolvedIn: "Phase 2",
  },
  {
    id: "D-006",
    title: "Improvement over persistence is not statistically significant",
    severity: "High",
    status: "ADDRESSED",
    location: "results/legacy_chapter/metrics.json",
    evidence:
      "Paired Wilcoxon p = 0.1495; paired bootstrap 95% CI [-6.47, +1.39] spans zero.",
    impact: "The headline result is not statistically supported at 0.05.",
    resolution:
      "Paired tests and bootstrap intervals published alongside every metric.",
    resolvedIn: "Phase 2",
  },
  {
    id: "D-007",
    title: "Fusion does not beat its own ground-only ablation",
    severity: "High",
    status: "RESOLVED",
    location: "results/legacy_chapter/metrics.json",
    evidence:
      "Ground_history_only MAE 16.617 vs full ensemble 17.298; full ensemble R² 0.017.",
    impact:
      "No evidence that satellite or meteorological information adds predictive value.",
    resolution:
      "Full ablation table reported as the primary Track B result; negative result stated as the finding.",
    resolvedIn: "Phase 7",
  },
  {
    id: "D-008",
    title: "PBLH unavailable; PBLTOP fallback is a latent trap",
    severity: "Medium",
    status: "RESOLVED",
    location: "legacy/aerosphinx_pipeline.py:124-125",
    evidence:
      "Neither PBLH nor PBLTOP present in any of the 172 granules.",
    impact:
      "A dead fallback that would silently produce a physically invalid feature.",
    resolution:
      "No PBLTOP fallback; PBLH reported under features_unavailable.",
    resolvedIn: "Phase 5",
  },
  {
    id: "D-009",
    title: "Humidity_Ratio is mis-named and mis-defined",
    severity: "Medium",
    status: "RESOLVED",
    location: "legacy/aerosphinx_pipeline.py:425",
    evidence:
      "QV2M / (T2M + EPS) is not a humidity ratio; QV2M is specific humidity (kg/kg).",
    impact: "The label invites misinterpretation.",
    resolution: "Documented in SCIENTIFIC_ASSUMPTIONS.md.",
    resolvedIn: "Phase 5",
  },
  {
    id: "D-010",
    title: "MERRA-2 features are midday single-cell values",
    severity: "Medium",
    status: "FIXED",
    location: "data/merra2/*.SUB.nc",
    evidence:
      "Granules subsetted to timesteps 12, 13, 14 (11:30–13:30 UTC) on a 1×1 grid at 28.5N, 77.5E.",
    impact: "Features are not diurnally representative.",
    resolution:
      "Every variable carries a _midday suffix; provenance recorded in the run manifest.",
    resolvedIn: "Phase 5",
  },
  {
    id: "D-011",
    title: "Feature importance is dominated by the month term",
    severity: "Medium",
    status: "PARTIALLY_ADDRESSED",
    location: "results/legacy_chapter/feature_importance.csv",
    evidence: "month_sin carries 37.93% of ensemble importance.",
    impact:
      "The model is substantially a seasonal climatology with a persistence anchor.",
    resolution:
      "Reported in Phase 2; a month-only comparison remains an open question (B-017).",
    resolvedIn: "Phase 2",
  },
  {
    id: "D-012",
    title: "Duplicate CPCB source files across two repository locations",
    severity: "Medium",
    status: "FIXED",
    location: "data/ vs ML model/CPCB Data/",
    evidence: "7 pairs are byte-identical (identical SHA-256).",
    impact: "A latent double-counting hazard in the aggregation contract.",
    resolution:
      "Deduplicated to a single canonical location; duplicates preserved under legacy/.",
    resolvedIn: "Phase 0",
  },
  {
    id: "D-013",
    title: "Mean-of-file-means aggregation is structurally unsafe",
    severity: "Medium",
    status: "FIXED",
    location: "src/aerosphinx/ingestion/cpcb.py",
    evidence:
      "0 mismatching days today, but correctness depended on the input layout.",
    impact:
      "Adding one overlapping file would silently change every affected daily mean.",
    resolution:
      "Hour-weighted pooling across all files; identical to the legacy output on this record.",
    resolvedIn: "Phase 1",
  },
  {
    id: "D-014",
    title: "Legacy pipelines absent from the worktree",
    severity: "Medium",
    status: "FIXED",
    location: "66 tracked paths deleted",
    evidence: "407 tracked paths deleted; 66 exist nowhere on disk.",
    impact: "The experimental history would be lost.",
    resolution:
      "50 unique artefacts recovered under legacy/ and verified byte-identical.",
    resolvedIn: "Phase 0",
  },
  {
    id: "D-015",
    title: "CRLF line-ending conversion in relocated CPCB files",
    severity: "Low",
    status: "DOCUMENTED",
    location: "data/cpcb/*.csv",
    evidence:
      "9 files use CRLF; byte-identical to the git originals after normalisation.",
    impact: "Raw digests differ; no data alteration.",
    resolution: "Documented in DATA_DICTIONARY.md; no action required.",
    resolvedIn: "Phase 0",
  },
  {
    id: "D-016",
    title: "Operational forward forecast extrapolates an upward drift",
    severity: "Low",
    status: "CLOSED",
    location: "results/legacy_chapter/operational_forward_forecast.csv",
    evidence: "96.6 → 112.3 → 122.2 → 129.6 → 138.5 µg/m³ across five days.",
    impact:
      "A one-step persistence anchor applied unchanged at every horizon of a recursion.",
    resolution:
      "Horizon-appropriate baselines and explicit issue timestamps; output labelled a projection, not a forecast.",
    resolvedIn: "Phase 8",
  },
  {
    id: "D-017",
    title: "A baseline defect silently shrank the headline evaluation set",
    severity: "High",
    status: "FIXED",
    location: "src/aerosphinx/models/baselines.py",
    evidence: "409 rows undefined for the baseline; 15 of 96 folds lost it.",
    impact: "Understated the cold-start problem.",
    resolution:
      "Headline set defined by models under test; undefined predictions scored NaN, not dropped.",
    resolvedIn: "Phase 3 audit",
  },
  {
    id: "D-018",
    title: "The bootstrap overstated the effective sample size",
    severity: "High",
    status: "FIXED",
    location: "src/aerosphinx/evaluation/block_inference.py",
    evidence: "Error lag-1 autocorrelation +0.31, lag-30 +0.16.",
    impact:
      "Intervals were too narrow; this changed a conclusion for Random Forest vs persistence.",
    resolution:
      "Moving-block and year-blocked bootstrap, Holm and Benjamini-Hochberg corrections.",
    resolvedIn: "Phase 3 audit",
  },
  {
    id: "D-019",
    title: "The forecast baseline was not horizon-appropriate",
    severity: "Critical",
    status: "FIXED",
    location: "src/aerosphinx/tracks/multi_horizon.py",
    evidence:
      "A day-h forecast was compared against the day-1 persistence value.",
    impact: "This inverted the Phase 3 conclusion.",
    resolution:
      "persistence_forecast gained a periods argument; the correct baseline is used.",
    resolvedIn: "Phase 3 Workstream A",
  },
  {
    id: "D-020",
    title: "The CPCB barometric pressure column is corrupt",
    severity: "High",
    status: "FIXED",
    location: "config/data_quality.yaml",
    evidence:
      "107 days above 800 mmHg (max 1062 mmHg = 1416 hPa); r = -0.214 with MERRA-2 SLP; diurnal phase inverted.",
    impact:
      "Supplied as a feature; removing it moves the station group from significantly harmful to neutral.",
    resolution:
      "BP screened and excluded from the station_met feature group.",
    resolvedIn: "Phase 5",
  },
];

export type QuestionStatus =
  | "OPEN"
  | "ANSWERED"
  | "PARTIALLY_RESOLVED"
  | "DECIDED";

export interface ResearchQuestion {
  id: string;
  question: string;
  note: string;
  status: QuestionStatus;
}

export const RESEARCH_QUESTIONS: ResearchQuestion[] = [
  { id: "B-001", question: "Is a boundary-layer-height product obtainable for 2024?", note: "Requires a MERRA-2 re-subset with PBLH.", status: "OPEN" },
  { id: "B-002", question: "Can INSAT-3D granules for 2016–2023 be obtained?", note: "Without them the fused dataset stays confined to 2024.", status: "OPEN" },
  { id: "B-003", question: "Are the 27 zero-pixel AOD granules cloud-flagged or retrievable elsewhere?", note: "No QA/cloud-flag dataset is present locally.", status: "OPEN" },
  { id: "B-004", question: "Should chapter draft and .pbix/.docx artefacts be retained?", note: "A project decision, not a research input.", status: "OPEN" },
  { id: "B-005", question: "Is the PowerBI work part of the research record?", note: "Affects legacy/ placement.", status: "OPEN" },
  { id: "B-006", question: "What is the intended availability-aware fallback policy?", note: "Described conceptually but not specified.", status: "OPEN" },
  { id: "B-007", question: "What is the provenance of the root-level legacy CSVs?", note: "Needs upstream context not in the repository.", status: "OPEN" },
  { id: "B-008", question: "Should the 124 rows without a complete 1-day lag be dropped?", note: "Changes the usable dataset by ~4%.", status: "OPEN" },
  { id: "B-009", question: "How should a day with a duplicated hour be resolved?", note: "No effect on the current record.", status: "OPEN" },
  { id: "B-010", question: "Can CPCB export provenance and QA conventions be established?", note: "Partially resolved by the Phase 3 Workstream D audit, but the availability timestamp remains unknown.", status: "PARTIALLY_RESOLVED" },
  { id: "B-011", question: "Should the cold-start folds (2017) be headline or excluded?", note: "DECIDED: keep them; all 96 folds are retained.", status: "DECIDED" },
  { id: "B-012", question: "Is a regime-conditional model in scope?", note: "DECIDED: approved as an experiment, not the primary model.", status: "DECIDED" },
  { id: "B-013", question: "Should the regime split be re-derived on validation folds?", note: "ANSWERED in Workstream C: the calendar split is a fixed definition.", status: "ANSWERED" },
  { id: "B-014", question: "Is continuous regime weighting better than a hard split?", note: "Open.", status: "OPEN" },
  { id: "B-015", question: "Can the cold-start coverage failure be addressed directly?", note: "Open; weighting toward rare regimes is a candidate.", status: "OPEN" },
  { id: "B-016", question: "Does a regime-aware fallback help at multi-day horizons?", note: "ANSWERED in Workstream A: no.", status: "ANSWERED" },
  { id: "B-017", question: "Is the multi-horizon advantage driven by calendar or PM2.5 history?", note: "Open; an ablation would separate them.", status: "OPEN" },
  { id: "B-018", question: "Should day-1 and multi-day use separate models?", note: "PARTIALLY ANSWERED in Workstream C.", status: "PARTIALLY_RESOLVED" },
  { id: "B-019", question: "Does the multi-horizon advantage survive nine-year heterogeneity?", note: "Open; confounded with the Phase 0 measurement change.", status: "OPEN" },
  { id: "B-020", question: "Can recursive forecasting be salvaged with per-horizon damping?", note: "ANSWERED in Workstream C: no.", status: "ANSWERED" },
  { id: "B-021", question: "Does CPCB publish hourly data with a timestamp and what is the latency?", note: "Blocks making station columns forecast-usable.", status: "OPEN" },
  { id: "B-022", question: "Is a declared one-hour publication lag acceptable?", note: "DECIDED: retrospective-only; no lag may be assumed.", status: "DECIDED" },
  { id: "B-023", question: "Which station-local variables carry genuine PM2.5 signal?", note: "ANSWERED in Workstream C: none, as a group.", status: "ANSWERED" },
  { id: "B-024", question: "Should BP and VWS be excluded on completeness grounds?", note: "ANSWERED in Phase 5: BP excluded for a physical reason (D-020).", status: "ANSWERED" },
  { id: "B-038", question: "Can MERRA-2 granules be obtained for 2016–2023?", note: "Without them only 5 matched test rows exist.", status: "OPEN" },
  { id: "B-039", question: "Is the station BP corruption a fault or a transformation?", note: "The diurnal phase inversion points to a transformation.", status: "OPEN" },
  { id: "B-040", question: "Should every source be cross-validated against every other?", note: "The BP defect was found by cross-source validation.", status: "OPEN" },
  { id: "B-041", question: "Should MERRA-2 be re-requested with full 24-hour coverage?", note: "The current window samples 17:00–19:00 IST.", status: "OPEN" },
  { id: "B-042", question: "Does station meteorology help under a tree model with BP excluded?", note: "Open; Workstream C used ridge throughout.", status: "OPEN" },
];

/* ----------------------------- Validation records ---------------------------- */

export type ValidationStatus =
  | "PASS"
  | "WARNING"
  | "FAIL"
  | "NOT_COMPARABLE"
  | "INSUFFICIENT_EVIDENCE";

export interface ValidationRecord {
  id: string;
  source1: string;
  source2: string;
  variablePair: string;
  methodology: string;
  tolerance: string;
  sampleSize: number | null;
  status: ValidationStatus;
  evidence: string;
  notes: string;
}

export const VALIDATION_RECORDS: ValidationRecord[] = [
  {
    id: "VAL-001",
    source1: "MERRA-2",
    source2: "CPCB station",
    variablePair: "T2M_midday ↔ AT",
    methodology: "Daily Pearson correlation",
    tolerance: "r > 0.90",
    sampleSize: 169,
    status: "PASS",
    evidence: "r = +0.9728, mean difference +0.96 °C.",
    notes: "Values confirmed. The sample is an evening window (17:00–19:00 IST).",
  },
  {
    id: "VAL-002",
    source1: "MERRA-2",
    source2: "CPCB station",
    variablePair: "QV2M_midday ↔ implied specific humidity",
    methodology: "Daily Pearson correlation",
    tolerance: "r > 0.50",
    sampleSize: 169,
    status: "PASS",
    evidence: "r = +0.6038.",
    notes: "QV2M is specific humidity (kg/kg), not a humidity ratio.",
  },
  {
    id: "VAL-003",
    source1: "MERRA-2",
    source2: "CPCB station",
    variablePair: "SLP_midday ↔ surface pressure",
    methodology: "Daily mean difference",
    tolerance: "|Δp| < 30 hPa",
    sampleSize: 169,
    status: "PASS",
    evidence: "SLP minus PS = +23.32 hPa, implied elevation 185 m.",
    notes: "Consistent with Faridabad's ~200 m station elevation.",
  },
  {
    id: "VAL-004",
    source1: "CPCB station",
    source2: "MERRA-2",
    variablePair: "BP ↔ SLP_midday",
    methodology: "Correlation + diurnal phase analysis",
    tolerance: "r > 0.80 (expected positive)",
    sampleSize: 169,
    status: "FAIL",
    evidence:
      "r = -0.214; 107 days above 800 mmHg (max 1062 mmHg = 1416 hPa); diurnal max at 10h (expected near dawn).",
    notes:
      "The BP column is corrupt. This is D-020; B-039 investigation is open.",
  },
  {
    id: "VAL-005",
    source1: "INSAT-3D AOD",
    source2: "CPCB station",
    variablePair: "AOD availability ↔ PM2.5",
    methodology: "Informative-missingness test (Mann-Whitney U)",
    tolerance: "p > 0.05",
    sampleSize: 169,
    status: "WARNING",
    evidence:
      "p = 0.032; days without a valid AOD pixel mean 119.38 µg/m³ vs 105.34 µg/m³ with one.",
    notes:
      "Missingness is informative: the satellite record is least available when pollution is worst.",
  },
  {
    id: "VAL-006",
    source1: "CPCB station",
    source2: "CPCB station",
    variablePair: "BP (22.4% missing) vs VWS (43.3% missing)",
    methodology: "Completeness screen",
    tolerance: "missing < 30%",
    sampleSize: 2979,
    status: "NOT_COMPARABLE",
    evidence: "BP 22.4% missing, VWS 43.3% missing.",
    notes:
      "BP was excluded for a physical reason (D-020), not for completeness. VWS remains included.",
  },
  {
    id: "VAL-007",
    source1: "MERRA-2",
    source2: "MERRA-2",
    variablePair: "PBLH ↔ PBLTOP",
    methodology: "Availability check",
    tolerance: "—",
    sampleSize: null,
    status: "NOT_COMPARABLE",
    evidence: "Neither PBLH nor PBLTOP is present in any of the 172 granules.",
    notes: "PBLH is unavailable; no safe proxy exists (D-008).",
  },
  {
    id: "VAL-008",
    source1: "INSAT-3D AOD",
    source2: "INSAT-3D AOD",
    variablePair: "Granule validity (169 granules)",
    methodology: "Pixel-support audit",
    tolerance: "≥ 1 valid pixel per granule",
    sampleSize: 169,
    status: "WARNING",
    evidence:
      "27 of 169 granules contribute zero valid pixels; the median granule contributes 12.",
    notes:
      "No QA or cloud-flag dataset is present, so a pixel means a retrieval was attempted, not that it is trustworthy.",
  },
  {
    id: "VAL-009",
    source1: "CPCB station",
    source2: "CPCB station",
    variablePair: "Duplicate files across two locations",
    methodology: "SHA-256 byte comparison",
    tolerance: "0 mismatch",
    sampleSize: 9,
    status: "PASS",
    evidence: "7 pairs byte-identical after CRLF/LF normalisation.",
    notes: "D-012; duplicates preserved under legacy/.",
  },
  {
    id: "VAL-010",
    source1: "CPCB station",
    source2: "CPCB station",
    variablePair: "Hourly → daily aggregation",
    methodology: "Mean-of-file-means vs hour-weighted mean",
    tolerance: "0 mismatch days",
    sampleSize: 2979,
    status: "PASS",
    evidence: "0 mismatching days (max absolute difference 1.1e-13).",
    notes: "D-013; correct by construction after the Phase 1 fix.",
  },
];

/* ----------------------------- Experiment catalog ---------------------------- */

export interface ExperimentDef {
  id: string;
  track: "A" | "B";
  name: string;
  inputs: string[];
  question: string;
  mode: "retrospective" | "forecast" | "conditional";
}

export const EXPERIMENT_CATALOG: ExperimentDef[] = [
  { id: "A0", track: "A", name: "Persistence baseline", inputs: ["Previous-day PM2.5"], question: "How well does yesterday's PM2.5 predict today?", mode: "forecast" },
  { id: "A1", track: "A", name: "Seasonal baseline", inputs: ["Historical PM2.5", "calendar"], question: "Does a day-of-year climatology help?", mode: "forecast" },
  { id: "A2", track: "A", name: "Temporal ML", inputs: ["Historical PM2.5", "calendar", "engineered temporal features"], question: "Does temporal ML beat persistence?", mode: "forecast" },
  { id: "A3", track: "A", name: "Local meteorology", inputs: ["A2", "station-local meteorology"], question: "Does local meteorology add value?", mode: "forecast" },
  { id: "A4", track: "A", name: "Environmental covariates", inputs: ["A2", "AOD / MERRA-2 if available"], question: "Can environmental covariates be used operationally?", mode: "conditional" },
  { id: "A5", track: "A", name: "Direct horizon models", inputs: ["Historical PM2.5", "calendar"], question: "Independent day 1–5 forecasts?", mode: "forecast" },
  { id: "A6", track: "A", name: "Recursive forecasting", inputs: ["Historical PM2.5", "calendar"], question: "How does error compound with horizon?", mode: "forecast" },
  { id: "B0", track: "B", name: "CPCB historical only", inputs: ["CPCB historical features"], question: "Ground-only benchmark", mode: "retrospective" },
  { id: "B1", track: "B", name: "AOD only", inputs: ["INSAT-3D AOD"], question: "How informative is satellite aerosol alone?", mode: "retrospective" },
  { id: "B2", track: "B", name: "MERRA-2 only", inputs: ["MERRA-2 reanalysis"], question: "How informative is meteorology alone?", mode: "retrospective" },
  { id: "B3", track: "B", name: "CPCB + AOD", inputs: ["CPCB", "INSAT AOD"], question: "Does satellite add value over ground-only?", mode: "retrospective" },
  { id: "B4", track: "B", name: "CPCB + MERRA-2", inputs: ["CPCB", "MERRA-2"], question: "Does meteorology add value over ground-only?", mode: "retrospective" },
  { id: "B5", track: "B", name: "Full fusion", inputs: ["CPCB", "AOD", "MERRA-2"], question: "Does full fusion beat its components?", mode: "retrospective" },
  { id: "B6", track: "B", name: "Availability-aware combination", inputs: ["CPCB", "availability-aware source selection"], question: "How robust is the model when a source is missing?", mode: "retrospective" },
];

export const STATION = {
  name: "Faridabad",
  lat: 28.4089,
  lon: 77.3178,
} as const;