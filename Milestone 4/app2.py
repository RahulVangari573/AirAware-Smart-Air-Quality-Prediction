import os
import streamlit as st
import pandas as pd
from utils_m3m4 import (
	load_dataset,
	gauge_figure,
	pollutant_trend_figure,
	compute_current_aqi_value,
	aqi_category_from_pm25,
	forecast_figure,
	forecast_7day_aqi_colored,
	forecast_7day_list,
	build_alerts,
)

st.set_page_config(page_title="AirAware - M3 & M4", layout="wide")

DATA_PATH = "cleaned_air_quality.csv"
if not os.path.exists(DATA_PATH):
	st.error(f"Cleaned data not found at {DATA_PATH}.")
	st.stop()

df = load_dataset(DATA_PATH)

st.title("AirAware — Alert System & Web Dashboard")
st.markdown("This app contains Milestone 3 (Alert System) and Milestone 4 (Streamlit Dashboard).")

st.sidebar.header("Controls")
city_options = sorted(df["City"].dropna().unique().tolist())
city = st.sidebar.selectbox("Monitoring Station", city_options, index=0)
time_range = st.sidebar.selectbox("Time Range", ["Last 24 Hours", "Last 7 Days", "Last 30 Days"], index=1)
pollutant = st.sidebar.selectbox("Pollutant", ["PM2.5", "PM10", "NO2", "O3"], index=0)
forecast_horizon = st.sidebar.selectbox("Forecast Horizon", ["24 Hours", "48 Hours", "7 Days"], index=0)

# Filter by time window
df_city = df[df["City"] == city].copy()
df_city = df_city.sort_values("Date")
if time_range == "Last 24 Hours":
	start_time = df_city["Date"].max() - pd.Timedelta(hours=24)
	df_city_window = df_city[df_city["Date"] >= start_time]
elif time_range == "Last 7 Days":
	start_time = df_city["Date"].max() - pd.Timedelta(days=7)
	df_city_window = df_city[df_city["Date"] >= start_time]
else:
	start_time = df_city["Date"].max() - pd.Timedelta(days=30)
	df_city_window = df_city[df_city["Date"] >= start_time]

tab1, tab2 = st.tabs(["Alert System (M3)", "Web Dashboard (M4)"])

with tab1:
	col1, col2 = st.columns([1, 1])
	with col1:
		st.subheader("Current Air Quality")
		aqi_val = compute_current_aqi_value(df_city_window)
		st.plotly_chart(gauge_figure(aqi_val), use_container_width=True, key="m3_gauge")
		st.caption(f"{aqi_category_from_pm25(aqi_val)} — {city}")
	with col2:
		st.subheader("7-Day AQI Forecast")
		items = forecast_7day_list(df_city_window, city)
		cols = st.columns(7)
		for i, it in enumerate(items[:7]):
			with cols[i]:
				st.markdown(f"""
<div style="background:{it['color']}; padding:12px; border-radius:10px; text-align:center; color:white">
	<div style="font-weight:600">{it['date'].strftime('%a')}</div>
	<div style="font-size:20px; font-weight:700; line-height:1.2">{int(round(it['value']))}</div>
	<div style="font-size:12px">{it['category']}</div>
</div>
""", unsafe_allow_html=True)

	st.subheader("Pollutant Concentrations")
	trend_fig = pollutant_trend_figure(df_city_window, ["PM2.5", "PM10", "NO2", "O3"])
	if trend_fig:
		st.plotly_chart(trend_fig, use_container_width=True, key="m3_trends")

	st.subheader("Active Alerts")
	for alert in build_alerts(df_city_window):
		if alert["level"] == "warning":
			st.warning(f"{alert['title']} — {alert['when']}")
		elif alert["level"] == "error":
			st.error(f"{alert['title']} — {alert['when']}")
		else:
			st.info(f"{alert['title']} — {alert['when']}")

with tab2:
	left, right = st.columns([1, 1])
	with left:
		st.subheader("Current Air Quality")
		aqi_val = compute_current_aqi_value(df_city_window)
		st.plotly_chart(gauge_figure(aqi_val), use_container_width=True, key="m4_gauge")
	with right:
		st.subheader("PM2.5 Forecast")
		h_days = 1 if forecast_horizon == "24 Hours" else (2 if forecast_horizon == "48 Hours" else 7)
		st.plotly_chart(forecast_figure(df_city_window, city, h_days), use_container_width=True, key="m4_pm25_fc")

	st.subheader("Pollutant Trends")
	trend_fig2 = pollutant_trend_figure(df_city_window, ["PM2.5", "PM10", "NO2", "O3"])
	if trend_fig2:
		st.plotly_chart(trend_fig2, use_container_width=True, key="m4_trends")

	st.subheader("Alert Notifications")
	for alert in build_alerts(df_city_window):
		if alert["level"] == "warning":
			st.warning(f"{alert['title']} — {alert['when']}")
		elif alert["level"] == "error":
			st.error(f"{alert['title']} — {alert['when']}")
		else:
			st.success(f"{alert['title']} — {alert['when']}")


