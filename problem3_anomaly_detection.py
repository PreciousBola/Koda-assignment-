"""
Problem 3: anomaly and spike detection for occupancy.parquet.

Three anomaly classes, each with a distinct root cause and a distinct
audience:

1. Hard invalid values: occupancy or traffic below zero, or occupancy
   above the space's stated capacity. These are almost always a sensor
   or pipeline bug (miscounted exits, double-counted entries, a stale
   capacity value in the space config), not a real crowding event.

2. Sudden jumps (spikes): occupancy that moves far more in one hour
   than the space's own history says is normal. Flagged with a
   per-space z-score on the hour-over-hour change, so a busy retail
   entrance and a quiet back office get different thresholds instead
   of one global cutoff that's wrong for both.

3. Flatline / stuck sensor: occupancy that reports the exact same
   nonzero value for many consecutive hours. Real foot traffic moves;
   a perfectly flat signal for half a day usually means the sensor
   stopped updating, not that the room genuinely held the same
   headcount all day.
"""

import pandas as pd
import numpy as np


def detect_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows with negative counts or occupancy above capacity."""
    flags = pd.DataFrame(index=df.index)
    flags["negative_occupancy"] = df["occupancy"] < 0
    flags["negative_traffic"] = df["traffic"] < 0
    flags["negative_people_in"] = df["people_in"] < 0
    flags["negative_people_out"] = df["people_out"] < 0
    flags["over_capacity"] = df["occupancy"] > df["capacity"]

    out = df.copy()
    out["anomaly_invalid"] = flags.any(axis=1)
    out["invalid_reason"] = flags.apply(
        lambda r: ", ".join([c for c in flags.columns if r[c]]), axis=1
    )
    return out


def detect_spikes(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """
    Flag hour-over-hour occupancy changes that are extreme relative to
    that specific space's own volatility. Uses a per-space z-score on
    the first difference, not a fixed headcount cutoff, since a jump of
    50 people means something very different in a 4000-capacity lobby
    versus a 20-person meeting room.

    The z-score is computed within each (space, hour-of-day) group
    rather than across the whole day. Every building has a real,
    expected surge at open and a real, expected drop at close; scoring
    those against the full day's volatility flags normal business
    hours as anomalies. Conditioning on hour-of-day means a spike only
    gets flagged when it's unusual for that specific hour, not just
    unusual for the day overall.
    """
    out = df.sort_values(["space_name", "date_time"]).copy()
    out["occupancy_delta"] = out.groupby("space_name")["occupancy"].diff()
    out["hour_of_day"] = out["date_time"].dt.hour

    # Expected delta for this space at this hour of day (removes the
    # predictable open/close swing). With about a week of history this
    # is only 5-7 samples per hour, too few to also estimate a stable
    # std from, so the std comes from the residuals pooled across the
    # whole space instead, that has enough data to be reliable.
    expected = out.groupby(["space_name", "hour_of_day"])["occupancy_delta"].transform("mean")
    out["delta_residual"] = out["occupancy_delta"] - expected

    residual_std = out.groupby("space_name")["delta_residual"].transform(
        lambda s: s.std(ddof=0) if s.std(ddof=0) > 0 else np.nan
    )
    out["delta_zscore"] = out["delta_residual"] / residual_std
    out["anomaly_spike"] = out["delta_zscore"].abs() >= z_threshold
    return out


def detect_flatline(df: pd.DataFrame, min_consecutive_hours: int = 6) -> pd.DataFrame:
    """
    Flag stretches where a space reports the identical nonzero
    occupancy value for min_consecutive_hours or more in a row.
    """
    out = df.sort_values(["space_name", "date_time"]).copy()

    def _flag_group(g: pd.DataFrame) -> pd.Series:
        same_as_prev = g["occupancy"].eq(g["occupancy"].shift())
        run_id = (~same_as_prev).cumsum()
        run_len = g.groupby(run_id)["occupancy"].transform("size")
        return (run_len >= min_consecutive_hours) & (g["occupancy"] != 0)

    out["anomaly_flatline"] = out.groupby("space_name", group_keys=False).apply(_flag_group)
    return out


def run_all_anomaly_checks(
    df: pd.DataFrame, z_threshold: float = 3.0, min_consecutive_hours: int = 6
) -> pd.DataFrame:
    """Runs all three checks and returns the dataframe with flag columns added."""
    out = detect_invalid_values(df)
    out = detect_spikes(out, z_threshold=z_threshold)
    out = detect_flatline(out, min_consecutive_hours=min_consecutive_hours)
    out["anomaly_any"] = out[["anomaly_invalid", "anomaly_spike", "anomaly_flatline"]].any(axis=1)
    return out


def anomaly_summary(flagged_df: pd.DataFrame) -> pd.DataFrame:
    """Per-building rollup of anomaly counts, for the executive dashboard."""
    return (
        flagged_df.groupby("building_name")
        .agg(
            total_rows=("occupancy", "size"),
            invalid_count=("anomaly_invalid", "sum"),
            spike_count=("anomaly_spike", "sum"),
            flatline_count=("anomaly_flatline", "sum"),
            any_anomaly_count=("anomaly_any", "sum"),
        )
        .assign(anomaly_rate_pct=lambda d: (d["any_anomaly_count"] / d["total_rows"] * 100).round(2))
        .sort_values("any_anomaly_count", ascending=False)
        .reset_index()
    )


if __name__ == "__main__":
    df = pd.read_parquet("../data/occupancy.parquet")
    flagged = run_all_anomaly_checks(df)

    print("Total rows:", len(flagged))
    print("Invalid value rows:", flagged["anomaly_invalid"].sum())
    print("Spike rows:", flagged["anomaly_spike"].sum())
    print("Flatline rows:", flagged["anomaly_flatline"].sum())
    print("Any anomaly:", flagged["anomaly_any"].sum())
    print()
    print(anomaly_summary(flagged).to_string(index=False))
    print()
    print("Sample flagged rows:")
    cols = [
        "building_name",
        "space_name",
        "date_time",
        "occupancy",
        "capacity",
        "anomaly_invalid",
        "invalid_reason",
        "anomaly_spike",
        "delta_zscore",
        "anomaly_flatline",
    ]
    print(flagged[flagged["anomaly_any"]][cols].head(15).to_string(index=False))
