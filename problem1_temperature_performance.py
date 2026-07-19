"""
Problem 1: Hourly temperature performance by occupied mode.

Goal: measure how well a device holds its target temperature, split by
occupied vs unoccupied mode, on an hourly basis.

Input: timeseries.parquet with at least these fields per device/point:
    device_id, timestamp, point_field, value
where point_field includes 'zone_air_temperature_sensor',
'zone_air_temperature_setpoint', and 'occupied_mode'.

If your real timeseries.parquet has a wide format instead (one column per
point), skip the pivot step in load_and_pivot() and pass your dataframe
straight into compute_hourly_performance().
"""

import pandas as pd
import numpy as np


def load_and_pivot(path: str) -> pd.DataFrame:
    """
    Load a long-format timeseries file and pivot points into columns.
    Expects columns: device_id, timestamp, point_field, value.
    Returns a wide dataframe indexed by device_id, timestamp with one
    column per point.
    """
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    wide = df.pivot_table(
        index=["device_id", "timestamp"],
        columns="point_field",
        values="value",
        aggfunc="last",
    ).reset_index()
    wide.columns.name = None
    return wide


def compute_hourly_performance(
    df: pd.DataFrame,
    device_col: str = "device_id",
    time_col: str = "timestamp",
    temp_col: str = "zone_air_temperature_sensor",
    setpoint_col: str = "zone_air_temperature_setpoint",
    mode_col: str = "occupied_mode",
    max_gap_minutes: int = 30,
) -> pd.DataFrame:
    """
    Resample raw device timeseries to hourly windows and compute
    temperature performance, split by occupied mode.

    Handling of missing values:
    - Raw readings are forward-filled up to max_gap_minutes before
      resampling. Gaps longer than that are left null rather than
      filled, so a sensor outage doesn't silently get treated as a
      real reading.
    - occupied_mode is forward-filled with no time limit, since mode
      is a state, not a continuous measurement. It only changes on
      an actual transition event.

    Returns one row per device per hour with:
    - avg / min / max temperature
    - avg setpoint
    - avg absolute deviation from setpoint
    - percent of the hour spent within +/-1 degree of setpoint
    - dominant occupied_mode for that hour
    - pct_data_available, a coverage flag for how much of the hour
      has real (non-imputed) readings
    """
    df = df.sort_values([device_col, time_col]).copy()

    out_rows = []
    for device_id, g in df.groupby(device_col):
        g = g.set_index(time_col)

        # Cap forward-fill of continuous sensor values to max_gap_minutes,
        # so long outages are not disguised as flat readings.
        limit = max(1, max_gap_minutes)
        temp_filled = g[temp_col].resample("1min").ffill(limit=limit)
        setpoint_filled = g[setpoint_col].resample("1min").ffill(limit=limit)

        # Mode is a state variable: fill forward with no limit.
        mode_filled = g[mode_col].resample("1min").ffill()

        minute_df = pd.DataFrame(
            {
                "temp": temp_filled,
                "setpoint": setpoint_filled,
                "mode": mode_filled,
            }
        )
        minute_df["deviation"] = (minute_df["temp"] - minute_df["setpoint"]).abs()
        minute_df["within_1deg"] = minute_df["deviation"] <= 1.0

        hourly = minute_df.resample("1h").agg(
            avg_temp=("temp", "mean"),
            min_temp=("temp", "min"),
            max_temp=("temp", "max"),
            avg_setpoint=("setpoint", "mean"),
            avg_abs_deviation=("deviation", "mean"),
            pct_within_1deg=("within_1deg", "mean"),
            minutes_with_data=("temp", "count"),
        )
        hourly["pct_within_1deg"] = (hourly["pct_within_1deg"] * 100).round(1)
        hourly["pct_data_available"] = (hourly["minutes_with_data"] / 60 * 100).round(1)

        # Dominant mode per hour: the mode value with the most minutes.
        mode_per_hour = (
            minute_df["mode"].resample("1h").agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        )
        hourly["occupied_mode"] = mode_per_hour
        hourly[device_col] = device_id
        out_rows.append(hourly.reset_index())

    result = pd.concat(out_rows, ignore_index=True)
    result = result.rename(columns={time_col: "hour"})
    cols = [
        device_col,
        "hour",
        "occupied_mode",
        "avg_temp",
        "min_temp",
        "max_temp",
        "avg_setpoint",
        "avg_abs_deviation",
        "pct_within_1deg",
        "pct_data_available",
    ]
    return result[cols]


def summarize_by_mode(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll hourly performance up to a per-device, per-mode summary.
    This is the number you'd put in front of a facilities manager:
    "how well does this device hold setpoint when occupied vs not."
    """
    return (
        hourly_df.groupby(["device_id", "occupied_mode"])
        .agg(
            avg_abs_deviation=("avg_abs_deviation", "mean"),
            pct_within_1deg=("pct_within_1deg", "mean"),
            hours_observed=("hour", "count"),
            avg_data_coverage=("pct_data_available", "mean"),
        )
        .reset_index()
        .round(2)
    )


if __name__ == "__main__":
    from load_real_data import load_real_timeseries

    wide = load_real_timeseries("../data/timeseries.parquet")
    hourly = compute_hourly_performance(wide, device_col="device_id")
    summary = summarize_by_mode(hourly)

    print(hourly.to_string(index=False))
    print()
    print(summary.to_string(index=False))
