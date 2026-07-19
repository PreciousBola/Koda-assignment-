# Dashboard Design: Operator vs Executive

Two dashboards on the same occupancy data, built for different jobs. The operator dashboard answers "what needs my attention right now." The executive dashboard answers "how is the portfolio trending and where should I invest."

## Operator dashboard

**Audience**: facilities or building ops staff monitoring day-to-day conditions.

**Time grain**: hourly, current day plus rolling 7-day comparison.

**Metrics**:
- Live occupancy vs capacity per space, as a percentage
- Anomaly count by building, today
- Flagged anomaly detail: which check triggered (invalid value, spike, flatline), timestamp, magnitude
- Data coverage: percent of expected hourly readings actually received, so a quiet sensor doesn't get mistaken for a quiet room

**Charts**:
- Bar chart, anomaly count by building, today (`outputs/charts/operator_anomalies_by_building_today.png`), sorted so the worst building is immediately visible
- Time series with flagged points overlaid, for a drilled-into space (`outputs/charts/operator_space_detail_with_flags.png`), so an operator can see exactly when and how far a reading departed from normal

**Level of detail**: row-level. An operator needs to click into Building_13, Space_X, 2pm, and see the raw occupancy number next to capacity and the specific rule that fired. Aggregates hide the thing they're trying to fix.

**Interaction**: filterable by building and space, sortable by anomaly count, with a way to acknowledge or dismiss a flagged anomaly once it's investigated (a real config issue vs a one-off event shouldn't keep re-alerting the same way).

## Executive dashboard

**Audience**: leadership deciding where to invest in space, staffing, or equipment.

**Time grain**: weekly and monthly, trended over quarters.

**Metrics**:
- Portfolio-wide average utilization, trended over time
- Utilization by building, ranked, to surface consistent over- and under-utilization rather than single-day noise
- Anomaly rate as a percent of total readings, by building, as a data-quality signal, not an operational alert
- State/region rollups, since the portfolio spans multiple provinces and states

**Charts**:
- Line chart, portfolio-wide daily average utilization (`outputs/charts/executive_portfolio_utilization_trend.png`), to spot a broad trend at a glance
- Ranked bar chart, top and bottom 5 buildings by average utilization (`outputs/charts/executive_top_bottom_buildings.png`), the kind of view that drives a real estate or staffing decision
- Ranked bar chart, anomaly rate by building, top 10 (`outputs/charts/executive_anomaly_rate_by_building.png`), the data-quality signal, flagged here as a rate to review periodically, not an alert to act on today

**Level of detail**: aggregated, no single-hour drill-down. An executive doesn't need to know that Space1_Building_10 spiked at 2am on March 7th; they need to know that Building_10 has run at 12% average utilization for a month and might be a candidate for consolidation.

**Interaction**: filter by date range, state, and space type (Building/Tower/Retail); export to a slide-ready image or PDF for board reporting.

## Why they differ

Same underlying data, different failure mode if you get it wrong. Show an operator a monthly trend line and they can't act on it in the moment; the anomaly that needs fixing today gets buried in an aggregate. Show an executive row-level anomaly detail and the signal that matters, a building running at 15% utilization for two months, gets buried under one-off sensor blips that don't warrant executive attention. The operator dashboard is built to surface individual outliers fast. The executive dashboard is built to suppress individual outliers so the trend underneath is visible.

## Anomaly logic (shared by both, surfaced differently)

Implemented in `src/problem3_anomaly_detection.py`, three checks:

1. **Invalid values**: `occupancy < 0`, `traffic < 0`, `people_in/out < 0`, or `occupancy > capacity`. Deterministic, no threshold to tune. Found 48 rows, concentrated in 2 buildings, consistent with a sensor or config bug rather than scattered noise.

2. **Spikes**: hour-over-hour change in occupancy, z-scored against that same space's typical change for that hour of day, then pooled across the week for a stable std estimate. Catches jumps that are unusual for a specific hour (a 2am spike, say) without flagging every building's normal 8am/5pm swing, which a naive whole-day z-score does.

3. **Flatline**: 6 or more consecutive hours of an identical nonzero occupancy value. Real foot traffic moves hour to hour; a sensor reporting the exact same number for half a day is more likely stuck than a room that genuinely holds a constant headcount overnight.

Operator view surfaces every flagged row, since that's what needs a look. Executive view surfaces only the aggregate anomaly rate per building, as a data-quality health signal, not day-to-day noise.
