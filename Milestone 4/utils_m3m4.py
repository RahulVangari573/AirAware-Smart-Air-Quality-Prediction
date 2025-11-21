import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import timedelta
from utils_m1m2 import load_saved_model_by_label


@st.cache_data
def load_dataset(path: str = "cleaned_air_quality.csv") -> pd.DataFrame:
	if not os.path.exists(path):
		raise FileNotFoundError(path)
	df = pd.read_csv(path)
	df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
	df = df.dropna(subset=["Date"])
	return df


def compute_current_aqi_value(df_city: pd.DataFrame) -> float:
	if df_city.empty or "PM2.5" not in df_city.columns:
		return float("nan")
	row = df_city.sort_values("Date").iloc[-1]
	return float(row.get("PM2.5", np.nan))


def aqi_category_from_pm25(value: float) -> str:
	try:
		v = float(value)
	except Exception:
		return "Unknown"
	if v <= 50:
		return "Good"
	if v <= 100:
		return "Moderate"
	if v <= 200:
		return "Unhealthy for Sensitive"
	if v <= 300:
		return "Unhealthy"
	if v <= 400:
		return "Very Unhealthy"
	return "Hazardous"

def category_color(category: str) -> str:
	c = category or "Unknown"
	if c == "Good":
		return "#2ecc71"
	if c in ["Moderate", "Unhealthy for Sensitive"]:
		return "#f1c40f"
	if c in ["Unhealthy", "Very Unhealthy", "Hazardous"]:
		return "#e74c3c"
	return "#95a5a6"


def gauge_figure(aqi_value: float) -> go.Figure:
	if np.isnan(aqi_value):
		aqi_value = 0.0
	fig = go.Figure(go.Indicator(
		mode="gauge+number",
		value=aqi_value,
		title={"text": "AQI"},
		gauge={
			"axis": {"range": [0, 500]},
			"bar": {"color": "#ed7d31"},
			"steps": [
				{"range": [0, 50], "color": "#3cb371"},
				{"range": [50, 100], "color": "#ffd966"},
				{"range": [100, 200], "color": "#ffa07a"},
				{"range": [200, 300], "color": "#ff7f7f"},
				{"range": [300, 400], "color": "#c00000"},
				{"range": [400, 500], "color": "#7f0000"},
			],
		}
	))
	fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
	return fig


def pollutant_trend_figure(df_city: pd.DataFrame, pollutants: list[str]) -> go.Figure | None:
	if df_city.empty:
		return None
	cols = [c for c in pollutants if c in df_city.columns]
	if not cols:
		return None
	dfp = df_city.sort_values("Date")[["Date"] + cols].copy()
	dfm = dfp.melt(id_vars="Date", value_vars=cols, var_name="Pollutant", value_name="Value")
	fig = px.line(dfm, x="Date", y="Value", color="Pollutant", template="plotly_white",
	              title="Pollutant Trends")
	return fig


def naive_forecast(series: pd.Series, horizon_days: int) -> pd.Series:
	series = series.dropna().sort_index()
	if series.empty:
		return pd.Series(dtype=float)
	last_val = float(series.iloc[-1])
	future_idx = pd.date_range(series.index[-1] + timedelta(days=1), periods=horizon_days, freq="D")
	return pd.Series([last_val] * horizon_days, index=future_idx)


def produce_forecast(df_city: pd.DataFrame, label: str, horizon_days: int) -> pd.Series:
	series = df_city.set_index("Date")["PM2.5"].resample("D").mean().interpolate()
	model, mtype = load_saved_model_by_label(label)
	# If we have ARIMA/XGB/Prophet from PKL, try to forecast simply; otherwise naive.
	try:
		if model is not None and mtype not in ("LSTM_FILE", "LSTM"):
			if hasattr(model, "forecast"):
				values = model.forecast(steps=horizon_days)
				future_idx = pd.date_range(series.index[-1] + timedelta(days=1), periods=horizon_days, freq="D")
				return pd.Series(values, index=future_idx)
			if hasattr(model, "predict"):
				recent = series.values[-3:].tolist()
				preds = []
				for _ in range(horizon_days):
					x_input = np.array(recent[-3:]).reshape(1, -1)
					p = float(model.predict(x_input)[0])
					preds.append(p)
					recent.append(p)
				future_idx = pd.date_range(series.index[-1] + timedelta(days=1), periods=horizon_days, freq="D")
				return pd.Series(preds, index=future_idx)
	except Exception:
		pass
	return naive_forecast(series, horizon_days)


def forecast_figure(df_city: pd.DataFrame, label: str, horizon_days: int) -> go.Figure:
	series = df_city.set_index("Date")["PM2.5"].resample("D").mean().interpolate()
	fc = produce_forecast(df_city, label, horizon_days)
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Historical", line=dict(color="#4c6ef5")))
	if not fc.empty:
		fig.add_trace(go.Scatter(x=fc.index, y=fc.values, name="Forecast", line=dict(color="#ff7f0e", dash="dash")))
	fig.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="PM2.5", title=f"PM2.5 Forecast — {label}")
	return fig

def forecast_7day_aqi_colored(df_city: pd.DataFrame, label: str) -> go.Figure:
	"""
	Builds a 7-day forecast and colors each day by AQI category.
	Good: green, Moderate/middle: orange/yellow, Bad: red.
	"""
	horizon_days = 7
	fc = produce_forecast(df_city, label, horizon_days)
	if fc.empty:
		return go.Figure()
	# Map PM2.5 to colors
	colors = []
	for v in fc.values:
		cat = aqi_category_from_pm25(v)
		if cat == "Good":
			colors.append("#2ecc71")
		elif cat in ["Moderate", "Unhealthy for Sensitive"]:
			colors.append("#f1c40f")
		else:
			colors.append("#e74c3c")
	fig = go.Figure()
	fig.add_trace(go.Bar(x=fc.index, y=fc.values, marker_color=colors, name="AQI Forecast"))
	fig.update_layout(
		template="plotly_white",
		title=f"7-Day AQI Forecast — {label}",
		xaxis_title="Date",
		yaxis_title="PM2.5 (as AQI input)"
	)
	return fig

def forecast_7day_list(df_city: pd.DataFrame, label: str) -> list[dict]:
	"""
	Returns a list of 7 dicts with fields: date, value, category, color.
	Useful to render compact tiles in Streamlit.
	"""
	fc = produce_forecast(df_city, label, 7)
	items: list[dict] = []
	for dt, val in fc.items():
		cat = aqi_category_from_pm25(val)
		items.append({"date": dt.date(), "value": float(val), "category": cat, "color": category_color(cat)})
	return items


def build_alerts(df_city: pd.DataFrame) -> list[dict]:
	alerts: list[dict] = []
	current = compute_current_aqi_value(df_city)
	category = aqi_category_from_pm25(current)
	alerts.append({"level": "info", "title": f"{category} air quality", "when": "Today"})
	# Simple rule-based forward-looking alert on rising trend
	dfc = df_city.sort_values("Date").tail(24)
	if "PM2.5" in dfc.columns and len(dfc) >= 6:
		diff = dfc["PM2.5"].iloc[-6:].mean() - dfc["PM2.5"].iloc[:6].mean()
		if diff > 10:
			alerts.append({"level": "warning", "title": "Rising PM2.5 expected", "when": "Tomorrow"})
	# Forecast-based alerts for next 7 days
	try:
		fc = produce_forecast(df_city, str(df_city["City"].iloc[-1]) if "City" in df_city.columns and not df_city.empty else "City", 7)
		for dt, val in fc.items():
			cat = aqi_category_from_pm25(val)
			if cat in ["Unhealthy for Sensitive", "Unhealthy", "Very Unhealthy", "Hazardous"]:
				alerts.append({"level": "warning", "title": f"High AQI expected ({cat})", "when": dt.strftime("%a, %I:%M %p")})
	except Exception:
		pass
	return alerts


