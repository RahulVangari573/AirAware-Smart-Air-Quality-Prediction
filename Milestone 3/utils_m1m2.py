# utils_m1m2.py
import os, pickle, joblib
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
import streamlit as st, plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

try:
    from tensorflow.keras.models import load_model as tf_load_model
except Exception:
    tf_load_model = None

try:
    from prophet import Prophet
except Exception:
    Prophet = None

@st.cache_data
def load_data(path="cleaned_air_quality.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
    return df

def filter_df(df, city=None, start_date=None, end_date=None, pollutants=None):
    temp = df.copy()
    if city and city != "All":
        temp = temp[temp["City"] == city]
    if start_date:
        temp = temp[temp["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        temp = temp[temp["Date"] <= pd.to_datetime(end_date)]
    if pollutants:
        keep = ["Date"]
        if "City" in temp.columns: keep.append("City")
        keep += [p for p in pollutants if p in temp.columns]
        temp = temp[[c for c in keep if c in temp.columns]]
    return temp

def data_quality_report(df, columns=None):
    if columns is None: columns = df.columns.tolist()
    total = len(df)
    return pd.DataFrame([{
        "column": c, "missing": int(df[c].isna().sum()),
        "pct_missing": round(df[c].isna().mean()*100,2),
        "dtype": str(df[c].dtype), "unique": int(df[c].nunique())
    } for c in columns if c in df.columns])

def summary_stats(df, cols):
    cols = [c for c in cols if c in df.columns]
    return df[cols].describe().T if cols else pd.DataFrame()

def corr_matrix(df, cols):
    cols = [c for c in cols if c in df.columns]
    return df[cols].corr() if len(cols) > 1 else pd.DataFrame()

# ---------- Fixed visualizations ----------
def plot_corr_heatmap(df, numeric_columns, figsize=(8, 6)):
    corr = corr_matrix(df, numeric_columns)
    if corr.empty:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    sns.set_theme(style="darkgrid")

    heat = sns.heatmap(
        corr,
        annot=True,
        cmap="RdYlBu_r",     # ✅ vivid, bright contrasting colors
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        square=True
    )

    ax.set_title("Pollutant Correlation Heatmap", color="white", fontsize=12)
    ax.tick_params(colors="white", labelsize=10)
    fig.patch.set_facecolor("#111")  # blend background
    ax.set_facecolor("#111")

    # ✅ Make annotations readable (white text over dark cells)
    for text in heat.texts:
        text.set_color("white")

    return fig



def plot_time_series(df, pollutants, city=None):
    if df.empty or "Date" not in df.columns: return None
    valid = [p for p in pollutants if p in df.columns]
    if not valid: return None
    melted = df.melt(id_vars=["Date","City"] if "City" in df.columns else ["Date"],
                     value_vars=valid, var_name="Pollutant", value_name="Value")
    fig = px.line(melted, x="Date", y="Value", color="Pollutant",
                  title=f"Time Series - {city if city else 'All Cities'}",
                  template="plotly_white")
    fig.update_layout(legend_title_text="Pollutants")
    return fig

def plot_distribution(df, column, figsize=(8,5)):
    if column not in df.columns or df[column].dropna().empty: return None
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")
    sns.histplot(df[column].dropna(), kde=True, ax=ax, color="#0b5394", bins=30)
    ax.set_title(f"Distribution: {column}", color="black", fontsize=12)
    ax.set_xlabel(column, color="black")
    ax.set_ylabel("Frequency", color="black")
    return fig
# ------------------------------------------

def pm25_to_aqi_category(pm):
    try: pm = float(pm)
    except: return "Unknown"
    if pm<=50: return "Good"
    if pm<=100: return "Moderate"
    if pm<=200: return "Unhealthy for Sensitive"
    if pm<=300: return "Unhealthy"
    if pm<=400: return "Very Unhealthy"
    return "Hazardous"

def current_aqi_info(df_city):
    if df_city.empty: return {"aqi": None, "category": "No Data", "datetime": None}
    last = df_city.sort_values("Date").iloc[-1]
    pm = last.get("PM2.5", np.nan)
    return {"aqi": pm, "category": pm25_to_aqi_category(pm), "datetime": last.get("Date")}

def load_saved_model_by_label(label):
    """
    Case-insensitive + folder-safe model loader.
    Works for .pkl, .joblib, .h5 files inside /models.
    """
    base_dir = "models"
    if not os.path.exists(base_dir):
        return None, None

    label_clean = label.strip().lower().replace(" ", "_")

    for root, _, files in os.walk(base_dir):
        for file in files:
            f_lower = file.lower().replace(" ", "_")
            if f_lower.startswith(label_clean) and "_best_model" in f_lower:
                path = os.path.join(root, file)
                try:
                    # Handle TensorFlow models even when TF isn't installed.
                    if path.endswith(".h5"):
                        if tf_load_model:
                            return tf_load_model(path), "LSTM"
                        # TensorFlow not available: return the path as a sentinel
                        # so the caller knows a model file exists for this label.
                        return path, "LSTM_FILE"
                    if path.endswith(".pkl") or path.endswith(".joblib"):
                        try:
                            return joblib.load(path), "other"
                        except Exception:
                            with open(path, "rb") as handle:
                                return pickle.load(handle), "other"
                except Exception as e:
                    print("⚠️ Failed to load model:", path, "error:", e)
                    continue
    return None, None


def evaluate_model(actual, predicted):
    try:
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        return float(mae), float(rmse)
    except: return (np.nan, np.nan)


# ---------- Forecast Engine analytics (lightweight) ----------
def _prepare_series(df_city, column="PM2.5"):
    s = df_city.set_index("Date")[column].dropna().sort_index()
    if s.empty: return s
    return s.resample("D").mean().interpolate()

def _train_test_split_series(series, test_size=30):
    n = len(series)
    test_size = min(test_size, max(7, n//10))
    return series.iloc[:-test_size], series.iloc[-test_size:]

def evaluate_simple_models_for_series(series):
    """
    Trains quick ARIMA(1,1,1) and Prophet (if available) on the training split,
    forecasts len(test) steps, and returns MAE/RMSE per model.
    """
    if series.empty: return {}
    train, test = _train_test_split_series(series)
    results = {}

    # ARIMA
    try:
        model = ARIMA(train, order=(1,1,1)).fit(method_kwargs={"warn_convergence": False})
        fc = model.forecast(steps=len(test))
        mae, rmse = evaluate_model(test.values, fc.values)
        results["ARIMA"] = {"MAE": mae, "RMSE": rmse}
    except Exception:
        pass

    # Prophet
    if Prophet is not None:
        try:
            dfp = train.reset_index().rename(columns={"Date":"ds", train.name if train.name else 0:"y"})
            dfp["y"] = train.values
            m = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=True)
            m.fit(dfp)
            future = m.make_future_dataframe(periods=len(test), freq="D")
            forecast = m.predict(future).set_index("ds")["yhat"].iloc[-len(test):]
            mae, rmse = evaluate_model(test.values, forecast.values)
            results["Prophet"] = {"MAE": mae, "RMSE": rmse}
        except Exception:
            pass

    return results

def build_model_performance_table(df_city, pollutant="PM2.5"):
    s = _prepare_series(df_city, pollutant)
    perf = evaluate_simple_models_for_series(s)
    if not perf: return pd.DataFrame(columns=["Model","RMSE","MAE"])
    rows = [{"Model": k, "RMSE": v["RMSE"], "MAE": v["MAE"]} for k,v in perf.items()]
    return pd.DataFrame(rows).sort_values("RMSE")

def best_model_by_pollutant(df_city, pollutants):
    rows = []
    for p in pollutants:
        tbl = build_model_performance_table(df_city, p)
        if not tbl.empty:
            best = tbl.iloc[0]
            rows.append({"Pollutant": p, "Best Model": best["Model"], "RMSE": round(best["RMSE"], 3), "Status": "Active"})
    return pd.DataFrame(rows)

def forecast_accuracy_curve(df_city, pollutant="PM2.5"):
    """
    Very light backtest to approximate accuracy vs horizon.
    Uses ARIMA on historical series and evaluates at several horizons.
    """
    s = _prepare_series(df_city, pollutant)
    if s.empty or len(s) < 60:
        return pd.DataFrame(columns=["Horizon","ARIMA","Prophet"])
    horizons = [1,3,6,12,24,48]  # days, for visualization
    horizon_labels = ["1d","3d","6d","12d","24d","48d"]
    arima_scores, prophet_scores = [], []

    # Fit once on all but max horizon
    train, test_full = _train_test_split_series(s, test_size=max(horizons)+7)
    try:
        arima_model = ARIMA(train, order=(1,1,1)).fit(method_kwargs={"warn_convergence": False})
    except Exception:
        arima_model = None

    if Prophet is not None:
        try:
            dfp = train.reset_index().rename(columns={"Date":"ds", train.name if train.name else 0:"y"})
            dfp["y"] = train.values
            prophet_model = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=True).fit(dfp)
        except Exception:
            prophet_model = None
    else:
        prophet_model = None

    for h in horizons:
        true = s.iloc[len(train): len(train)+h]
        # ARIMA
        if arima_model is not None:
            try:
                fc = arima_model.forecast(steps=h)
                mae, rmse = evaluate_model(true.values, fc.values[:len(true)])
                acc = max(0.0, 100.0 - rmse)  # simple readability metric
            except Exception:
                acc = np.nan
        else:
            acc = np.nan
        arima_scores.append(acc)

        # Prophet
        if prophet_model is not None:
            try:
                future = prophet_model.make_future_dataframe(periods=h, freq="D")
                forecast = prophet_model.predict(future).set_index("ds")["yhat"].iloc[-h:]
                mae, rmse = evaluate_model(true.values, forecast.values[:len(true)])
                acc = max(0.0, 100.0 - rmse)
            except Exception:
                acc = np.nan
        else:
            acc = np.nan
        prophet_scores.append(acc)

    return pd.DataFrame({"Horizon": horizon_labels, "ARIMA": arima_scores, "Prophet": prophet_scores})
