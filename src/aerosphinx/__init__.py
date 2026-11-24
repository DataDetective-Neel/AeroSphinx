"""AeroSphinx - availability-aware multisource PM2.5 research platform.

Two independently evaluated research tracks share infrastructure but not
experimental assumptions:

* **Track A** - historical CPCB-based PM2.5 forecasting.
* **Track B** - multisource PM2.5 estimation using CPCB, INSAT-3D AOD and
  NASA MERRA-2, with forecasting only when future covariates are genuinely
  available at forecast issue time.

See ``AGENTS.md`` for the engineering contract and
``docs/reference/EXPERIMENT_PROTOCOL.md`` for the experimental protocol.
"""

__version__ = "0.1.0"

from aerosphinx.features.lags import (
    LagSpec,
    calendar_lag,
    calendar_lags,
    calendar_rolling,
    calendar_rolling_stats,
    observation_age_days,
)
from aerosphinx.ingestion.cpcb import (
    CPCBAudit,
    CPCBDaily,
    CPCBRules,
    aggregate_daily,
    load_cpcb_daily,
    read_cpcb_hourly,
)

__all__ = [
    "__version__",
    "CPCBAudit",
    "CPCBDaily",
    "CPCBRules",
    "aggregate_daily",
    "load_cpcb_daily",
    "read_cpcb_hourly",
    "LagSpec",
    "calendar_lag",
    "calendar_lags",
    "calendar_rolling",
    "calendar_rolling_stats",
    "observation_age_days",
]
