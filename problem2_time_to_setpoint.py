"""
Problem 2: Time to reach setpoint after turning on.

For each Unoccupied -> Occupied transition, measure how long the zone
takes to reach its setpoint. For devices that never reach it, estimate
the observed heating/cooling rate and project when they would.

Input: wide dataframe with device_id, timestamp, zone_air_temperature_sensor,
zone_air_temperature_setpoint, occupied_mode (same shape as Problem 1's
load_and_pivot output).
"""

import pandas as pd
import numpy as np


def find_occupancy_transitions(
    df: pd.DataFrame,
    device_col: str = "device_id",
    time_col: str = "timestamp",
    mode_col: str = "occupied_mode",
) -> pd.DataFrame:
    """
    Return one row per Unoccupied -> Occupied transition:
    device_id, transition_time.
    """
    df = df.sort_values([device_col, time_col]).copy()
    df["prev_mode"] = df.groupby(device_col)[mode_col].shift(1)
    transitions = df[(df["prev_mode"] == 0) & (df[mode_col] == 1)]
    return transitions[[device_col, time_col]].rename(columns={time_col: "transition_time"})


def time_to_reach_setpoint(
    df: pd.DataFrame,
    transitions: pd.DataFrame,
    device_col: str = "device_id",
    time_col: str = "timestamp",
    temp_col: str = "zone_air_temperature_sensor",
    setpoint_col: str = "zone_air_temperature_setpoint",
    tolerance_deg: float = 1.0,
    sustain_minutes: int = 10,
    lookahead_hours: int = 6,
) -> pd.DataFrame:
    """
    For each transition, find the first time the zone temperature comes
    within tolerance_deg of setpoint and stays there for at least
    sustain_minutes. That "sustained" requirement matters: a temperature
    that clips the setpoint for one noisy reading then drifts back out
    isn't actually "reached" in any practical sense.

    If setpoint is never reached within lookahead_hours, the row is
    flagged reached=False and handed to project_time_to_reach() for a
    rate-based estimate.

    Returns one row per event with:
    device_id, transition_time, reached (bool), minutes_to_reach,
    starting_temp, target_setpoint
    """
    results = []
    df = df.sort_values([device_col, time_col])

    for _, row in transitions.iterrows():
        device_id = row[device_col]
        t0 = row["transition_time"]

        # Source data here is event-driven (change-of-value logging),
        # not a fixed clock. A point only logs a new row when its value
        # changes, so at the exact transition timestamp the temperature
        # reading is usually null, it hasn't changed since the last log.
        # Forward-fill temp and setpoint across the device's full history
        # first, so we can find the last known reading at or before t0,
        # then take the window from there.
        device_df = df[df[device_col] == device_id].set_index(time_col)
        device_df[temp_col] = device_df[temp_col].ffill()
        device_df[setpoint_col] = device_df[setpoint_col].ffill()

        window = device_df[
            (device_df.index >= t0 - pd.Timedelta(minutes=1))
            & (device_df.index <= t0 + pd.Timedelta(hours=lookahead_hours))
        ].copy()

        if window.empty or window[temp_col].isna().all():
            continue

        # Interpolate short sensor gaps so a single missing reading
        # doesn't break the "sustained" check.
        window[temp_col] = window[temp_col].interpolate(method="time", limit=5)

        # Real controllers don't update the setpoint field in the same
        # instant occupied_mode flips; there's a short control-loop lag
        # (a couple minutes in this dataset) before the new occupied
        # target is written. Reading target at exactly t0 grabs the
        # stale unoccupied setback value. Use the setpoint a few minutes
        # after the transition instead, once it has caught up.
        setpoint_grace = pd.Timedelta(minutes=5)
        post_grace = window[window.index >= t0 + setpoint_grace]
        target = (
            post_grace[setpoint_col].iloc[0]
            if not post_grace.empty and post_grace[setpoint_col].notna().any()
            else window[setpoint_col].dropna().iloc[0]
            if window[setpoint_col].notna().any()
            else np.nan
        )
        starting_temp = window[temp_col].iloc[0]
        heating = target > starting_temp if pd.notna(target) else np.nan

        # Compare against the live, time-varying setpoint (already
        # forward-filled), not the frozen target above, so a mid-window
        # setpoint change is still handled correctly.
        window["within_tol"] = (window[temp_col] - window[setpoint_col]).abs() <= tolerance_deg

        # Rolling check: within tolerance for the full sustain window.
        sustained = (
            window["within_tol"]
            .rolling(f"{sustain_minutes}min")
            .apply(lambda x: x.all(), raw=True)
        )
        reached_idx = sustained[sustained == 1].index.min()

        if pd.notna(reached_idx):
            minutes_to_reach = (reached_idx - t0).total_seconds() / 60
            reached = True
        else:
            minutes_to_reach = np.nan
            reached = False

        results.append(
            {
                device_col: device_id,
                "transition_time": t0,
                "starting_temp": round(starting_temp, 2),
                "target_setpoint": round(target, 2),
                "heating": heating,
                "reached": reached,
                "minutes_to_reach": round(minutes_to_reach, 1) if reached else np.nan,
            }
        )

    return pd.DataFrame(results)


def project_time_to_reach(
    df: pd.DataFrame,
    events: pd.DataFrame,
    device_col: str = "device_id",
    time_col: str = "timestamp",
    temp_col: str = "zone_air_temperature_sensor",
    lookahead_hours: int = 6,
    min_realistic_rate: float = 0.02,
) -> pd.DataFrame:
    """
    For events where reached is False, fit a simple linear rate over the
    observed window and project how many additional minutes it would
    take to close the remaining gap, if that rate held.

    Also flags whether setpoint looks realistically reachable at all:
    - If the observed rate is flat or moving away from setpoint
      (rate below min_realistic_rate degrees/minute in the right
      direction), it's flagged not_realistic. That usually points to
      undersized equipment, a stuck valve/damper, or a setpoint outside
      what the system can physically deliver.
    """
    not_reached = events[~events["reached"]].copy()
    if not_reached.empty:
        not_reached["observed_rate_deg_per_min"] = []
        not_reached["projected_additional_minutes"] = []
        not_reached["realistically_reachable"] = []
        return not_reached

    df = df.sort_values([device_col, time_col])
    projections = []

    for _, row in not_reached.iterrows():
        device_id = row[device_col]
        t0 = row["transition_time"]
        target = row["target_setpoint"]
        heating = row["heating"]

        device_df = df[df[device_col] == device_id].copy()
        device_df[temp_col] = device_df[temp_col].ffill()
        window = device_df[
            (device_df[time_col] >= t0)
            & (device_df[time_col] <= t0 + pd.Timedelta(hours=lookahead_hours))
        ].dropna(subset=[temp_col])

        if len(window) < 2:
            rate = np.nan
        else:
            elapsed_min = (window[time_col] - t0).dt.total_seconds() / 60
            # Linear fit: degrees per minute.
            coeffs = np.polyfit(elapsed_min, window[temp_col], 1)
            rate = coeffs[0]

        last_temp = window[temp_col].iloc[-1] if not window.empty else np.nan
        remaining_gap = target - last_temp if heating else last_temp - target

        signed_rate = rate if heating else -rate
        realistic = signed_rate >= min_realistic_rate

        if realistic and remaining_gap > 0:
            projected_minutes = remaining_gap / signed_rate
        else:
            projected_minutes = np.nan

        projections.append(
            {
                device_col: device_id,
                "transition_time": t0,
                "observed_rate_deg_per_min": round(rate, 4) if pd.notna(rate) else np.nan,
                "remaining_gap_deg": round(remaining_gap, 2) if pd.notna(remaining_gap) else np.nan,
                "projected_additional_minutes": round(projected_minutes, 1)
                if pd.notna(projected_minutes)
                else np.nan,
                "realistically_reachable": bool(realistic),
            }
        )

    proj_df = pd.DataFrame(projections)
    return not_reached.merge(proj_df, on=[device_col, "transition_time"])


if __name__ == "__main__":
    from load_real_data import load_real_timeseries

    wide = load_real_timeseries("../data/timeseries.parquet")

    transitions = find_occupancy_transitions(wide, device_col="device_id")
    print(f"Found {len(transitions)} occupancy transitions")
    print(transitions.to_string(index=False))
    print()

    events = time_to_reach_setpoint(wide, transitions, device_col="device_id")
    print(events.to_string(index=False))
    print()

    projections = project_time_to_reach(wide, events, device_col="device_id")
    if not projections.empty:
        print("Devices that never reached setpoint, with projections:")
        print(projections.to_string(index=False))
    else:
        print("All devices reached setpoint within the lookahead window.")
