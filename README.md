# Building Analytics: Temperature Performance, Setpoint Response, and Occupancy Anomalies

Analysis of HVAC device telemetry and building occupancy data. Three problems: hourly temperature performance by occupied mode, time to reach setpoint after occupancy starts, and anomaly detection with dashboard design for occupancy trends.

## Data

- `data/timeseries.parquet`: device point readings (temperature sensor, cooling setpoint, occupied mode) for 4 HVAC devices in one building, March 14-15 2023. Event-driven logging: a row is written only when a value changes, not on a fixed clock.
- `data/occupancy.parquet`: hourly occupancy aggregates across 63 spaces in 41 buildings, March 5-11 2023.

## Repo structure

```
src/
  load_real_data.py                  reshapes timeseries.parquet into a wide, per-device schema
  problem1_temperature_performance.py hourly temp performance by occupied mode
  problem2_time_to_setpoint.py        time to reach setpoint after occupancy starts, plus projection for devices that never reach it
  problem3_anomaly_detection.py       invalid value, spike, and flatline detection for occupancy data
  generate_dashboard_charts.py        builds the mockup charts referenced in DASHBOARD_DESIGN.md
outputs/charts/                       PNG mockups for the operator and executive dashboards
DASHBOARD_DESIGN.md                   full writeup for Problem 3's dashboard design
```

## Problem 1: hourly temperature performance by occupied mode

`compute_hourly_performance()` resamples each device's readings to hourly windows and reports average deviation from setpoint, percent of the hour within 1 degree of setpoint, and a data coverage percentage per hour.

Two handling decisions worth calling out:

- Temperature and setpoint are forward-filled with a 30-minute cap before resampling. A sensor that goes quiet for 6 hours shouldn't have its last reading treated as true for those 6 hours; the row is left null instead, and `pct_data_available` surfaces how much of each hour is real versus imputed.
- `occupied_mode` is forward-filled with no cap, since mode is a state, not a continuous measurement, and only changes on a real transition event.

Why unoccupied-period drift matters: when a space is unoccupied, the AHU typically drops to minimal or setback operation, so temperature drift during that window is mostly a read on the building envelope and outdoor conditions, not the equipment. A stable unoccupied temperature suggests decent insulation and low load. Fast drift toward outdoor conditions during unoccupied hours points to envelope loss (poor insulation, air leaks) or a damper/valve that isn't fully closing, both of which show up as extra energy cost at the next occupied ramp-up.

## Problem 2: time to reach setpoint after occupancy starts

`find_occupancy_transitions()` finds every Unoccupied to Occupied flip. `time_to_reach_setpoint()` measures how long it takes the zone to get within 1 degree C of setpoint and stay there for 10 minutes, not just touch it once.

One real-data issue surfaced and fixed here: the setpoint field updates about 2 minutes after the occupied_mode flip, a normal control-loop lag, not a data error. Reading the setpoint at the exact transition timestamp grabs the stale unoccupied setback value instead of the real occupied target. The fix reads the setpoint a few minutes after the transition, once it has caught up, and compares against the live time-varying setpoint throughout the window rather than a frozen value.

For devices that don't reach setpoint in the lookahead window, `project_time_to_reach()` fits a linear rate on the observed temperature trajectory and projects the remaining time, and flags whether the setpoint looks realistically reachable at all given that rate. A setpoint that the observed rate would take days to reach, or is moving away from, usually points to one of: undersized equipment for the zone, a stuck or leaking valve/damper, or a setpoint outside what the system can physically deliver (worth checking against the equipment's rated capacity before assuming it's a control problem).

## Problem 3: dashboard design and anomaly detection

Full writeup: `DASHBOARD_DESIGN.md`. Anomaly detection covers three distinct failure modes:

- **Invalid values**: negative occupancy/traffic, or occupancy above stated capacity. Found 48 rows across 2 buildings; almost certainly a sensor or capacity-config issue, not real crowding.
- **Spikes**: hour-over-hour occupancy change that's extreme for that space at that specific hour of day. Comparing against the whole day's volatility flags every building's normal open/close swing as an anomaly; the fix conditions the expected delta on hour-of-day first, then pools the residual variance across the full week so the std estimate isn't built on 5-7 samples.
- **Flatline**: identical nonzero occupancy for 6+ consecutive hours, a sign the sensor stopped updating rather than the room genuinely holding a constant headcount.

## Running it

```bash
pip install pandas pyarrow numpy matplotlib
cd src
python3 problem1_temperature_performance.py    # runs against real timeseries.parquet
python3 problem2_time_to_setpoint.py           # runs against real timeseries.parquet
python3 problem3_anomaly_detection.py          # runs against real occupancy.parquet
python3 generate_dashboard_charts.py           # regenerates the chart mockups
```

## Known limitations

- Timeseries data covers roughly one day for 4 devices; Problem 1's per-mode averages and Problem 2's rate projections would firm up considerably with a week or more of history.
- One device (FC_L_3_1) logs a cooling setpoint of 0.0 for its entire history. That's read as-is here and flagged as not realistically reachable rather than silently dropped, since a 0-degree setpoint is itself a finding worth surfacing (likely a disabled point or config error, not a real target).
- Spike detection's z-score threshold (3.0) and flatline's minimum run length (6 hours) are reasonable starting points, not tuned against labeled ground truth. In production I'd want a sample of confirmed real anomalies to validate against.
