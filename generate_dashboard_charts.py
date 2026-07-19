"""
Generates static chart mockups for the two Problem 3 dashboards.
These are illustrations of what each dashboard would show, not the
dashboards themselves; a real build would be Tableau, Power BI, or a
Plotly/Dash app with filters and drill-downs.

Operator dashboard mockup: today's anomalies, per-building capacity
pressure, hourly trend for a single flagged space.

Executive dashboard mockup: portfolio-wide utilization trend, top and
bottom buildings by average utilization, weekly anomaly rate.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from problem3_anomaly_detection import run_all_anomaly_checks, anomaly_summary

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

df = pd.read_parquet("../data/occupancy.parquet")
flagged = run_all_anomaly_checks(df)
flagged["utilization_pct"] = (flagged["occupancy"] / flagged["capacity"] * 100).clip(upper=150)

OUT = "../outputs/charts"

# ---- Operator chart 1: anomaly count by building, most recent day ----
last_day = flagged["date_daily"].max()
today = flagged[flagged["date_daily"] == last_day]
by_building_today = (
    today.groupby("building_name")["anomaly_any"].sum().sort_values(ascending=False).head(10)
)

fig, ax = plt.subplots(figsize=(8, 5))
by_building_today.plot(kind="barh", ax=ax, color="#c0392b")
ax.invert_yaxis()
ax.set_title(f"Operator view: flagged anomalies by building, {pd.Timestamp(last_day).date()}")
ax.set_xlabel("Anomaly count")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT}/operator_anomalies_by_building_today.png", dpi=150)
plt.close()

# ---- Operator chart 2: hourly trend for the most anomalous space, with flags overlaid ----
worst_space = today.groupby("space_name")["anomaly_any"].sum().idxmax()
space_hist = flagged[flagged["space_name"] == worst_space].sort_values("date_time")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(space_hist["date_time"], space_hist["occupancy"], color="#2c3e50", linewidth=1.5, label="Occupancy")
ax.axhline(space_hist["capacity"].iloc[0], color="#7f8c8d", linestyle="--", linewidth=1, label="Capacity")
flagged_points = space_hist[space_hist["anomaly_any"]]
ax.scatter(flagged_points["date_time"], flagged_points["occupancy"], color="#c0392b", zorder=5, label="Flagged anomaly")
ax.set_title(f"Operator view: hourly occupancy, {worst_space} (flagged points highlighted)")
ax.set_ylabel("Occupancy")
ax.legend(loc="upper right", frameon=False)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(f"{OUT}/operator_space_detail_with_flags.png", dpi=150)
plt.close()

# ---- Executive chart 1: portfolio daily utilization trend ----
daily_util = flagged.groupby("date_daily")["utilization_pct"].mean()

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(daily_util.index, daily_util.values, marker="o", color="#2980b9", linewidth=2)
ax.set_title("Executive view: portfolio-wide average utilization by day")
ax.set_ylabel("Avg utilization (%)")
ax.set_xlabel("")
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(f"{OUT}/executive_portfolio_utilization_trend.png", dpi=150)
plt.close()

# ---- Executive chart 2: top and bottom 5 buildings by avg utilization ----
building_util = flagged.groupby("building_name")["utilization_pct"].mean().sort_values(ascending=False)
top5 = building_util.head(5)
bottom5 = building_util.tail(5)
combined = pd.concat([top5, bottom5])

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#27ae60"] * len(top5) + ["#c0392b"] * len(bottom5)
combined.plot(kind="barh", ax=ax, color=colors)
ax.invert_yaxis()
ax.set_title("Executive view: highest and lowest utilization buildings")
ax.set_xlabel("Avg utilization (%)")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT}/executive_top_bottom_buildings.png", dpi=150)
plt.close()

# ---- Executive chart 3: anomaly rate by building, top 10 ----
summary = anomaly_summary(flagged).sort_values("anomaly_rate_pct", ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(summary["building_name"], summary["anomaly_rate_pct"], color="#8e44ad")
ax.invert_yaxis()
ax.set_title("Executive view: anomaly rate by building (top 10)")
ax.set_xlabel("Anomaly rate (% of readings flagged)")
plt.tight_layout()
plt.savefig(f"{OUT}/executive_anomaly_rate_by_building.png", dpi=150)
plt.close()

print("Charts written to", OUT)
summary = anomaly_summary(flagged)
print(summary.head(5).to_string(index=False))
