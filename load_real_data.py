"""
Loads the real timeseries.parquet (long format, one row per point
reading, COV/event-based logging) and reshapes it into the wide
per-device, per-timestamp schema that problem1 and problem2 expect.

Real schema quirks handled here:
- value is stored as string for every kind, so Number fields need a
  cast to float and Bool fields need a mapping to 0/1.
- occupied_mode arrives as text ("Occupied" / "Unoccupied"), not 0/1.
- the setpoint field is named effective_cooling_zone_air_temperature_setpoint,
  not zone_air_temperature_setpoint as in the task doc's example. This
  loader renames it so it drops straight into the existing functions.
- readings are event-driven (change of value), not on a fixed clock.
  Problem 1 and 2 both handle that already via resample + ffill.
"""

import pandas as pd


def load_real_timeseries(path: str = "data/timeseries.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date_time_local"] = pd.to_datetime(df["date_time_local"])

    df = df.copy()
    is_bool = df["kind"] == "Bool"
    df.loc[is_bool, "value_clean"] = (df.loc[is_bool, "value"] == "Occupied").astype(float)
    df.loc[~is_bool, "value_clean"] = pd.to_numeric(df.loc[~is_bool, "value"], errors="coerce")

    wide = df.pivot_table(
        index=["device_id", "device_name", "date_time_local"],
        columns="field",
        values="value_clean",
        aggfunc="last",
    ).reset_index()
    wide.columns.name = None

    wide = wide.rename(
        columns={
            "date_time_local": "timestamp",
            "effective_cooling_zone_air_temperature_setpoint": "zone_air_temperature_setpoint",
        }
    )
    return wide


if __name__ == "__main__":
    wide = load_real_timeseries("../data/timeseries.parquet")
    print(wide.shape)
    print(wide.head(10).to_string(index=False))
    print()
    print(wide.groupby("device_name").size())
