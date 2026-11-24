# AeroSphinx — Legacy Artefacts

This directory preserves the pre-Phase-0 experimental record of AeroSphinx.
It is **archival**: nothing here is imported by the current pipeline, and
nothing here is a target for further development.

## Why these files exist

During the Phase 0 repository autopsy, the working tree contained a large
in-progress reorganisation that had **deleted 407 tracked paths** without
committing the deletions. A content-hash comparison of every deleted path
against every file remaining on disk classified them as:

| Class | Files | Disposition |
|---|---|---|
| Byte-identical to a file relocated under `data/` | 341 | Committed as **renames** — no content change |
| Identical to a `data/cpcb/` file after CRLF→LF normalisation | 16 | Dropped — the `data/cpcb/` copy is the preserved original |
| **Unique content, absent from disk** | **50** | **Restored here** |

The 50 files in this directory are the unique experimental record that would
otherwise have been lost from the working tree. They were recovered from
commit `859fd0842f48bfb8aba2bff93318005a45548665` and are verified
**byte-identical** to that commit's blobs.

## Contents

### Legacy research pipelines

| Path | Lines | Role |
|---|---|---|
| `scripts/run_research_pipeline.py` | 662 | Previous canonical pipeline. Uses the same CPCB aggregation, AOD window and MERRA-2 extraction logic as the current pipeline, so its historical results are directly comparable. |
| `scripts/merge_data.py`, `train_model.py`, `visualize_map.py`, `fetch_ground_pm25.py`, `load_insat_aod.py` | — | Per-stage helpers of the previous pipeline. |
| `ML model/ML_Model.py`, `ML model/AQI_Model.py`, `ML model/Confusion Matrix.py` | 159 / — / — | Earlier single-file models. |
| `ML_Model.py` | 159 | Root-level copy of `ML model/ML_Model.py`. Same length but **not byte-identical** — the two differ in content and are retained separately. |

### Exploratory notebooks

| Path | Notes |
|---|---|
| `Data_Reading.ipynb` | Byte-identical to `ML model/Data_Reading.ipynb`. |
| `ML model/Data_Reading.ipynb` | Duplicate of the above. |
| `ML model/merra_downloads/Data Reading.ipynb` | Zero bytes — an empty notebook placeholder. |

### Legacy derived datasets and results

`aod_data_1.csv`, `aod_data_2.csv`, `aod_data_3.csv` (**three files that
differ from each other** — same size, different content),
`combined_pm2.5_para.csv`, `combined_pm2.5_para_2.csv`,
`pm25_cpcb_05AM_01janto31dec_2019.csv`,
`predicted_pm25_jan_to_dec_2019.csv`, `ML model/aod_data.csv`,
`ML model/combined_pm2.5_para.csv`, `ML model/combined_pm2.5_para_2.csv`,
`ML model/mean_aod_faridabad.csv`, `ML model/CPCB Data/mean_aod_faridabad.csv`
(duplicate), `ML model/pm25_cpcb_*.csv`, `ML model/predicted_pm25*.csv`,
`results/audit_report.md`, `results/modeling_dataset_complete_cases.csv`.

**Provenance note.** The plan requires determining the provenance of the root
CSV files before moving or deleting them. That determination is **not yet
done** — see B-007 in `docs/DEFECT_REGISTER.md`. They are preserved here
unmodified so the question can be answered later from the data itself.

### Legacy figures

`figures/actual_vs_predicted_pm25.png`, `feature_importance.png`,
`five_day_forecast_example.png`, `monthly_pm25_seasonality.png`,
`residual_plot.png`, `ML model/pm25_confusion_matrix.png`,
`Flow chart.png`, `AeroSphinx.png`.

These are the **previous** figure set. The current pipeline produces
`figures/fig3`–`fig7`. Both sets are retained.

### Chapter and project deliverables

`AeroSphinx Indraneel.docx`, `AeroSphinx Indraneel2.docx`,
`Bharatiya Antariksh Hackathon 2025 Idea Submission PPT.pptx`,
`AeroSphinx.pbix`, `jhj.pdf`, `README.txt`.

These are **deliverables, not research inputs**. Whether they belong in the
repository long-term is an open project decision (B-004, B-005).

### Application / demo project

`pawanai-project/` — `pawanai.py` (1,106 lines), `API time.py`,
`API trial.py`, `requirements.txt`.

Per the plan, this is an application/demo codebase distinct from the research
experiments. It is isolated here rather than mixed into the research package.

### Byte-code

`scripts/__pycache__/load_insat_aod.cpython-313.pyc` — a compiled artefact of
a legacy script, retained only for completeness. Not source.

## Rules

1. Nothing in this directory may be imported by the research package.
2. Nothing here may be modified. These files are the historical record.
3. Any future migration of legacy logic into the research package must be a
   **new** file under `src/aerosphinx/`, with the legacy file left untouched
   and a note in `docs/DEFECT_REGISTER.md` recording what was migrated.
4. `git mv` may be used to relocate these files within `legacy/` if the
   layout needs tidying, but their content must not change.

## Verification

```bash
git diff --stat 859fd08 -- legacy/
```

A clean diff confirms the restored files match the audited baseline commit.
