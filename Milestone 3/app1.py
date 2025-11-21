import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import date
from utils_m1m2 import (
    load_data, filter_df, data_quality_report, summary_stats, corr_matrix,
    plot_time_series, plot_corr_heatmap, plot_distribution, current_aqi_info,
    load_saved_model_by_label, build_model_performance_table, best_model_by_pollutant,
    forecast_accuracy_curve
)
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="AirAware - M1 & M2", layout="wide")

st.markdown("""<style>
.header {font-size:22px; color:#0b5394; font-weight:600;}
</style>""", unsafe_allow_html=True)

st.title("AirAware — Data Explorer & Forecast Engine")
st.markdown("Use sidebar filters. This includes Milestone 1 (Explorer) and Milestone 2 (Forecast).")

# -----------------------------
# Load dataset
# -----------------------------
DATA_PATH = "cleaned_air_quality.csv"
if not os.path.exists(DATA_PATH):
    st.error(f"Cleaned data not found at {DATA_PATH}. Place cleaned CSV in project folder.")
    st.stop()

df = load_data(DATA_PATH)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Filters")
page = st.sidebar.selectbox("Choose page", ["Data Explorer (M1)", "Forecast Engine (M2)"])
cities = ["All"] + sorted(df["City"].dropna().unique().tolist())
city = st.sidebar.selectbox("City", cities, index=0)
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
start_date, end_date = st.sidebar.date_input("Date range", value=(min_date, max_date),
                                            min_value=min_date, max_value=max_date)
all_pollutants = [c for c in df.columns if c not in ["City", "Date", "AQI_Bucket", "AQI"]]
polls = st.sidebar.multiselect("Pollutants", options=all_pollutants, default=["PM2.5"])

filtered = filter_df(df, city=None if city == "All" else city,
                     start_date=start_date, end_date=end_date, pollutants=polls)

# ============================
# PAGE 1 — DATA EXPLORER (M1)
# ============================
if page == "Data Explorer (M1)":
    st.header("Data Explorer (M1)")
    st.subheader("Data Quality & Filters")
    st.write(f"Rows selected: {len(filtered)}")

    dq = data_quality_report(filtered, polls)
    st.table(dq)

    st.subheader("Time Series")
    if len(polls) > 0:
        fig = plot_time_series(filtered.sort_values("Date"), polls)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Statistical Summary")
    if len(polls) > 0:
        st.dataframe(summary_stats(filtered, polls).round(3))

    st.subheader("Pollutant Correlations")
    if len(polls) > 1:
        fig = plot_corr_heatmap(filtered, polls)
        if fig is not None:
            st.pyplot(fig)
    else:
        st.info("Select at least two pollutants to view correlation heatmap.")

    st.subheader("Distribution Analysis")
    for p in polls:
        fig = plot_distribution(filtered, p)
        if fig is not None:
            st.pyplot(fig)

# ============================
# PAGE 2 — FORECAST ENGINE (M2)
# ============================
elif page == "Forecast Engine (M2)":
    st.header("Forecast Engine (M2)")
    st.markdown("Model Performance, PM2.5 Forecast, Best Model by Pollutant, and Forecast Accuracy.")
    sel_city = st.selectbox("Forecast city", sorted(df["City"].unique()))
    st.write(f"Using data points: {len(df[df['City'] == sel_city])}")

    # Prepare city series
    city_df = df[df["City"] == sel_city].copy()

    # Top row: Model Performance (bar) and PM2.5 Forecast (line)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Model Performance")
        pollutants_perf = ["PM2.5","PM10","NO2","O3"]
        # Build grouped data: for each pollutant, collect RMSE for ARIMA/Prophet; LSTM shown as N/A if not evaluable
        categories = []
        arima_vals, prophet_vals, lstm_vals, lstm_hover = [], [], [], []
        # Detect if an LSTM file exists for this city
        _, lstm_type = load_saved_model_by_label(sel_city)
        lstm_available = lstm_type in ["LSTM","LSTM_FILE"]
        for pol in pollutants_perf:
            tbl = build_model_performance_table(city_df, pol)
            rmse_map = {r["Model"]: r["RMSE"] for _, r in tbl.iterrows()} if not tbl.empty else {}
            categories.append(pol)
            arima_vals.append(rmse_map.get("ARIMA", None))
            prophet_vals.append(rmse_map.get("Prophet", None))
            if lstm_available:
                # LSTM exists, but we don't evaluate due to missing preprocessing; mark as N/A
                lstm_vals.append(0)
                lstm_hover.append("LSTM: available (RMSE N/A)")
            else:
                lstm_vals.append(0)
                lstm_hover.append("LSTM: not available")
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Bar(name="ARIMA", x=categories, y=arima_vals, marker_color="#1f77b4"))
        fig_perf.add_trace(go.Bar(name="Prophet", x=categories, y=prophet_vals, marker_color="#ff7f0e"))
        fig_perf.add_trace(go.Bar(name="LSTM", x=categories, y=lstm_vals,
                                  marker_color="#7f7f7f", marker_pattern_shape="/",
                                  hovertext=lstm_hover, hoverinfo="text"))
        fig_perf.update_layout(barmode="group", template="plotly_white", yaxis_title="RMSE",
                               title="Model Performance by Pollutant")
        st.plotly_chart(fig_perf, use_container_width=True)

    with c2:
        st.subheader("PM2.5 Forecast")
        # Load saved model (for display note and potential use later)
        best_label = sel_city
        model, typ = load_saved_model_by_label(best_label)

        # Clean and resample series
        city_series = city_df.set_index("Date")["PM2.5"].dropna().sort_index()
        city_series = city_series.resample("D").mean().interpolate()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=city_series.index, y=city_series.values,
            name="Historical", line=dict(color="#0b5394")
        ))

        fc_days = 7

        # ---------- Prophet ----------
        if model is not None and "prophet" in str(type(model)).lower():
            future = model.make_future_dataframe(periods=fc_days, freq="D")
            forecast = model.predict(future)
            fc = forecast.set_index("ds").iloc[-fc_days:]
            fig.add_trace(go.Scatter(x=fc.index, y=fc["yhat"], name="Forecast", line=dict(color="#ff7f0e")))
            fig.add_trace(go.Scatter(x=fc.index, y=fc["yhat_upper"], name="Upper CI",
                                     line=dict(dash="dot", color="#ffb380")))
            fig.add_trace(go.Scatter(x=fc.index, y=fc["yhat_lower"], name="Lower CI",
                                     line=dict(dash="dot", color="#ffc8a6")))

        # ---------- ARIMA ----------
        elif model is not None and "arima" in str(type(model)).lower():
            fc = model.forecast(steps=fc_days)
            future_dates = pd.date_range(city_series.index[-1] + pd.Timedelta(days=1), periods=fc_days)
            fig.add_trace(go.Scatter(
                x=future_dates, y=fc, name=f"{fc_days}-Day Forecast (ARIMA)",
                line=dict(color="#ff7f0e")
            ))

        # ---------- XGBoost ----------
        elif model is not None and "xgb" in str(type(model)).lower():
            recent = city_series.values[-3:].tolist()
            preds = []
            for _ in range(fc_days):
                x_input = np.array(recent[-3:]).reshape(1, -1)
                next_pred = model.predict(x_input)[0]
                preds.append(next_pred)
                recent.append(next_pred)
            future_dates = pd.date_range(city_series.index[-1] + pd.Timedelta(days=1), periods=fc_days)
            fig.add_trace(go.Scatter(
                x=future_dates, y=preds, name=f"{fc_days}-Day Forecast (XGBoost)",
                line=dict(color="#ff7f0e")
            ))

        # ---------- LSTM ----------
        elif typ in ["LSTM", "LSTM_FILE"]:
            if typ == "LSTM_FILE":
                st.info("Found LSTM .h5 model file. TensorFlow not available, showing historical data only.")
            else:
                st.info("LSTM model loaded. Visualization placeholder — scaler pipeline required for forecast display.")

        fig.update_layout(
            title=f"PM2.5 Forecast — {sel_city}",
            xaxis_title="Date",
            yaxis_title="PM2.5",
            template="plotly_white",
            legend_title_text="Series"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Bottom row: Best Model by Pollutant (table) and Forecast Accuracy (line)
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Best Model by Pollutant")
        pollutants = ["PM2.5","PM10","NO2","O3"]
        table = best_model_by_pollutant(city_df, pollutants)
        if table.empty:
            st.info("Not enough data to determine best models.")
        else:
            st.dataframe(table, use_container_width=True)
    with c4:
        st.subheader("Forecast Accuracy")
        acc = forecast_accuracy_curve(city_df, "PM2.5")
        if acc.empty:
            st.info("Accuracy curve unavailable (insufficient data or dependencies missing).")
        else:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=acc["Horizon"], y=acc["ARIMA"], name="ARIMA"))
            if "Prophet" in acc.columns:
                fig_acc.add_trace(go.Scatter(x=acc["Horizon"], y=acc["Prophet"], name="Prophet"))
            fig_acc.update_layout(template="plotly_white", yaxis_title="Accuracy (%)", xaxis_title="Forecast Horizon")
            st.plotly_chart(fig_acc, use_container_width=True)
