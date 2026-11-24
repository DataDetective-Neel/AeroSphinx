"""Research tracks.

Track A and Track B share infrastructure but never share experimental
assumptions. See ``docs/reference/EXPERIMENT_PROTOCOL.md`` section 2.
"""

from aerosphinx.tracks.track_a import (
    DEFAULT_FEATURES,
    TrackAResult,
    run_track_a,
)

__all__ = ["DEFAULT_FEATURES", "TrackAResult", "run_track_a"]
