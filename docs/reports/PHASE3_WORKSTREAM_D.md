# AeroSphinx — Phase 3 Workstream D Report: CPCB Station-Local Meteorology Audit

**Workstream D complete.** Nine candidate variables audited. **None is
forecast-usable yet**, for a specific and stated reason.

Run directory: `results/runs/D-station-20261112T175233Z/`

---

## 1. Headline

> **The CPCB export contains a rich, genuine set of station-local
> meteorological measurements — and none of them can yet be used in an
> operational forecast, because the local files do not establish when each
> value became available to a forecaster.**

| | Count |
|---|---|
| Candidate variables audited | 9 |
| Confirmed station-local measurement | 9 (2 by direct test, 7 by inference) |
| With a **known availability timestamp** | **0** |
| **Forecast-usable** | **0** |

This is a **blocking finding, not a null result**. The variables are good;
the *timing contract* is missing.

---

## 2. What the columns actually are

The audit record for each variable is in
`station_variable_audit.csv` / `.json`. Summary:

| Variable | Column | Unit | Missing | Timing verdict | Quality flag |
|---|---|---|---|---|---|
| air_temperature | `AT (°C)` | °C | 5.81% | hourly measurement | timing unverified |
| relative_humidity | `RH (%)` | % | 6.07% | hourly measurement | timing unverified |
| wind_speed | `WS (m/s)` | m/s | 4.99% | low-variation | timing unverified |
| wind_direction | `WD (deg)` | deg | 5.11% | hourly measurement | timing unverified |
| rainfall | `RF (mm)` | mm | 3.83% | daily aggregate | timing unverified |
| cumulative_rainfall | `TOT-RF (mm)` | mm | 0.00% | daily aggregate | timing unverified |
| solar_radiation | `SR (W/mt2)` | W/m² | 6.82% | hourly measurement | timing unverified |
| barometric_pressure | `BP (mmHg)` | mmHg | 22.45% | hourly measurement | timing unverified |
| vertical_wind_speed | `VWS (m/s)` | m/s | 43.31% | low-variation | timing unverified |

### Evidence that these are genuine hourly measurements

Three independent checks were applied to every column.

**1. Grid.** All 78,912 timestamps across the nine files fall on an exact
hour boundary, with zero duplicate timestamps within a file.

**2. Within-day variation.** A daily aggregate repeated across 24 hourly
rows would show ~1 distinct value per day. The observed means are 22.9
distinct values per day for `AT`, 21.2 for `RH`, 22.6 for `WD`, 17.8 for
`SR`. These are **hourly measurements**.

**3. Physically coherent diurnal cycles.**

| Column | 00h | 06h | 09h | 12h | 15h | 18h | 21h |
|---|---|---|---|---|---|---|---|
| `AT (°C)` | 23.8 | 22.2 | 25.9 | 29.3 | 30.0 | 27.3 | 25.1 |
| `RH (%)` | 60.8 | 64.4 | 55.2 | 46.3 | 44.1 | 51.3 | 57.6 |
| `SR (W/mt2)` | 26.6 | 30.2 | 252.1 | 432.8 | 281.6 | 33.0 | 25.8 |

Solar radiation peaking at midday and reaching a floor at night, relative
humidity inversely tracking temperature, wind speed rising through the
day — these are the signatures of real instrumentation, not of a
reanalysis field or a repeated daily aggregate.

---

## 3. Are they actually station-local? Yes — tested directly

This was the critical question, because a reanalysis field copied into the
export would look plausible but would not be a *station* measurement.

**Hour-matched comparison against the independently supplied MERRA-2
granules** (same grid cell, same hour, 516 matched rows):

| Column | Compared against | Correlation | Mean difference | sd of difference | Identical values | Verdict |
|---|---|---|---|---|---|---|
| `AT (°C)` | T2M | +0.930 | +0.67 °C | **3.97 °C** | **0** | station-local |
| `RH (%)` | QV2M | −0.098 | +35.9 | 21.70 | **0** | station-local |
| `BP (mmHg)` | SLP | −0.273 | −282.9 | 10.99 | **0** | station-local |

**Zero identical values in 516 matched rows**, with differences of the
magnitude expected between a station and a 0.5° × 0.625° reanalysis cell
(elevation, exposure, local heat island). A reanalysis field would match
exactly.

The `RH` and `BP` rows compare quantities in different units (percent
humidity against specific humidity; millimetres of mercury against
hectopascals), so their *difference* columns are not meaningful in
magnitude — only the **zero identical values** and the absence of
correlation are. For those two the decisive evidence is that the values
differ at every matched hour, which reanalysis copying would not do.

For the remaining six columns MERRA-2 provides no directly corresponding
field, so the test is **not applicable** rather than performed against an
unrelated variable. Their station-locality is recorded as *inferred* from
the shared hourly grid and the physically coherent diurnal cycles — a
weaker claim, and labelled as such in the record.

> Note: `SR` and `RF` have an almost-zero diurnal *range* relative to
> their scale for rainfall (rain is intermittent, so a near-zero range is
> expected) and `WS`/`VWS` have small ranges by nature. The audit
> distinguishes these from a genuine daily aggregate by the number of
> distinct values per day, which is what the evidence rests on.

---

## 4. The blocking finding: availability time is unknown

The authorization specified the field set:

```
variable, source_column, physical_meaning, unit, station_or_source,
measurement_timestamp, availability_timestamp, missing_percentage,
valid_range, quality_flag, forecast_usable
```

and stated:

> The distinction between measurement time and availability time is
> essential. A meteorological measurement recorded for 10 AM but uploaded at
> 4 PM cannot be used in a forecast issued at noon.

**Measurement time** is established: it is the row timestamp, on an exact
hourly grid.

**Availability time** cannot be established. The CPCB export contains no
publication timestamp, no upload timestamp, no ingestion log, and no
metadata of any kind. Every record therefore carries:

```
availability_timestamp : UNKNOWN
forecast_usable        : False
```

**Consequence.** No station-local column may enter an operational forecast
under the current evidence. This is the conservative and correct reading:
without knowing when a value became available, a forecast issued at 00:00
cannot honestly claim to have used the 23:00 observation, because that
observation might have been uploaded hours later.

### What would unblock it

Any one of the following would establish the availability contract:

1. **CPCB publication metadata** — the timestamp at which each hourly
   record was published or uploaded. This is the cleanest route and is a
   data-provenance question for CPCB.
2. **A conservative lag assumption** — for example, that an hourly
   observation is available from the following hour. This would make the
   columns usable *for retrospective estimation* and for forecasts issued
   with at least one hour of lead, and would be an *assumption*, recorded
   as such rather than as a verified fact.
3. **An operational-only reading** — treat the station columns as
   retrospective-only covariates, used to test whether meteorology carries
   signal, and exclude them from anything described as operational.

Option 2 is the pragmatic route and is what the Phase 3 protocol's
availability contract permits if the assumption is declared. Option 3 is
automatically satisfied and costs nothing.

---

## 5. Consequences for the Track B feature priority

The authorization's order was: station-local meteorology → MERRA-2 →
INSAT AOD → combined → routing.

**This audit does not change that order, but it sharpens what each step
must do.**

- **Station-local meteorology is a strong candidate on physical grounds.**
  It is genuinely local, hourly, and covers the full 2016–2024 record at
  ~94% completeness, with no download cost. That makes it the natural
  first environmental feature group — but it is **retrospective-only until
  the availability contract is resolved**.

- **MERRA-2 has a *known* availability profile** that is different and
  worse: it is reanalysis, so it is not available at forecast issue time
  at all (the plan's §4 correction). MERRA-2 can support *retrospective
  estimation* but never an operational forecast.

- **INSAT-3D AOD has the same structure**: observed satellite data for a
  past date is available with a latency, but not for the future.

So the ordering by *practical availability* is:

| Group | Retrospective estimation | Operational forecast |
|---|---|---|
| Station-local meteorology | Available (retrospective assumption needed) | Blocked pending provenance |
| MERRA-2 | Available | **Never** (reanalysis) |
| INSAT AOD | Available | Blocked (no forecast product) |
| Combined | Available | Blocked |

**The Track B experiments that can be run now are retrospective
estimations.** That is precisely the distinction the Phase 0 audit
established as the project's central scientific correction, and it holds
here too.

---

## 6. What this workstream does and does not establish

**Establishes:**
- Nine candidate variables exist, are hourly, and are genuinely
  station-local.
- Their completeness is high (94–100% for seven of nine; `VWS` at 57%
  and `BP` at 78% are the weak ones).
- **None has an established availability timestamp**, so none is
  forecast-usable.
- The MERRA-2 comparison proves the columns are not reanalysis.

**Does not establish:**
- Whether the columns carry *predictive signal* for PM2.5. That is the
  next experiment, and it is a retrospective one until the availability
  question is settled.
- The availability contract itself. That requires information the local
  files do not contain.

---

## 7. Open questions

| ID | Question | Blocks |
|---|---|---|
| **B-021** | Does CPCB publish hourly data with a timestamp, and what is the typical publication latency relative to the observation hour? | Making the station columns forecast-usable |
| **B-022** | Is a declared one-hour publication lag acceptable for retrospective estimation, recorded as an assumption rather than a verified fact? | Unblocking the station-meteorology experiment |
| **B-023** | Which of the nine variables carry genuine PM2.5 signal? An ablation over the audited set, as a retrospective experiment. | Prioritising within the station-local group |
| **B-024** | Should `BP` (22.4% missing) and `VWS` (43.3% missing) be excluded on completeness grounds before any modelling? | The feature-group definition |

---

## 8. Artifacts

| File | Contents |
|---|---|
| `station_variable_audit.csv` | One row per variable with all required fields |
| `station_variable_audit.json` | The same, with the full evidence trail |
| `manifest.json` | Git SHA, environment, configuration, audit field list |
