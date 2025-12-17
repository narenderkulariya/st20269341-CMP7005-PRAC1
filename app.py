
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# ===================== DATA LOADING ===================== #
@st.cache_data
def load_data():
    df = pd.read_csv("all_cities_combined_cleaned.csv")

    # Fix Date column
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Date"] = df["Date"].dt.date
    df["Date"] = pd.to_datetime(df["Date"])

    # Ensure non-date object columns are strings
    for c in df.columns:
        if df[c].dtype == "object" and c != "Date":
            df[c] = df[c].astype(str)

    return df

df = load_data()

FEATURE_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
]
TARGET_COL = "AQI"

# ===================== MODEL LOADING/TRAINING ===================== #
@st.cache_resource
def load_or_train_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    joblib.dump(model, "model.pkl")
    return model

model = load_or_train_model()

# ===================== STREAMLIT LAYOUT ===================== #
st.set_page_config(page_title="Air Quality Dashboard",
                   layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Data Overview", "EDA", "Modelling & Prediction"]
)

# --------- PAGE 1: DATA OVERVIEW --------- #
if page == "Data Overview":
    st.title("🏭 Air Quality Data Overview")

    total_cities = df["City"].nunique()
    total_rows = len(df)
    min_year = int(df["Year"].min())
    max_year = int(df["Year"].max())
    avg_aqi = int(df["AQI"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cities", total_cities)
    c2.metric("Records", f"{total_rows:,}")
    c3.metric("Years", f"{min_year} - {max_year}")
    c4.metric("Average AQI", avg_aqi)

    st.subheader("Raw Data Sample")
    st.dataframe(df.head(50))

    st.subheader("Basic Statistics")
    st.write(df.describe())

    st.subheader("Missing Values per Column")
    miss = df.isna().sum()
    miss_df = pd.DataFrame({
        "Column": miss.index,
        "Missing": miss.values,
        "Missing %": (miss.values / len(df)) * 100
    })
    st.dataframe(miss_df)
    st.bar_chart(miss)

# --------- PAGE 2: EDA --------- #
elif page == "EDA":
    st.title("🔍 Exploratory Data Analysis")

    # City filter
    cities = sorted(df["City"].unique())
    city = st.selectbox("Select city", cities)

    city_df = df[df["City"] == city].copy()

    # AQI over time for selected city
    st.subheader(f"AQI over Time – {city}")
    ts = city_df[["Date", "AQI"]].set_index("Date").sort_index()
    st.line_chart(ts)

    # Yearly average AQI for whole dataset
    st.subheader("Yearly Average AQI (All Cities)")
    yearly = df.groupby("Year")["AQI"].mean().sort_index()
    st.bar_chart(yearly)

    # Monthly pattern (all cities)
    st.subheader("Monthly Average AQI (All Cities)")
    monthly = df.groupby("month")["AQI"].mean().sort_index()
    st.bar_chart(monthly)

    # Top 10 polluted and cleanest cities
    st.subheader("Top 10 Most Polluted Cities (Avg AQI)")
    top10 = df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10)
    st.bar_chart(top10)

    st.subheader("10 Cleanest Cities (Avg AQI)")
    clean10 = df.groupby("City")["AQI"].mean().sort_values(ascending=True).head(10)
    st.bar_chart(clean10)

    # Pollutant distribution for selected city
    st.subheader(f"Pollutant Distribution in {city}")
    pollutant = st.selectbox("Select pollutant", FEATURE_COLS)

    fig, ax = plt.subplots()
    city_df[pollutant].dropna().hist(bins=30, edgecolor="white", ax=ax)
    ax.set_xlabel(pollutant)
    ax.set_ylabel("Count")
    st.pyplot(fig)

# --------- PAGE 3: MODELLING & PREDICTION --------- #
else:
    st.title("🤖 Modelling & Prediction")

    st.markdown(
        "A **Random Forest Regressor** is trained on pollutant features "
        "to predict the Air Quality Index (AQI)."
    )

    # Evaluate button
    if st.button("Evaluate model on 20% test split"):
        X = df[FEATURE_COLS]
        y = df[TARGET_COL]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        eval_model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        eval_model.fit(X_train, y_train)
        y_pred = eval_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Evaluation Metrics")
        st.write(f"MAE:  {mae:.2f}")
        st.write(f"MSE:  {mse:.2f}")
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"R²:   {r2:.3f}")

    st.markdown("---")
    st.subheader("Manual AQI Prediction")

    defaults = df[FEATURE_COLS].median()

    col1, col2 = st.columns(2)
    with col1:
        pm25 = st.number_input("PM2.5", value=float(defaults["PM2.5"]))
        pm10 = st.number_input("PM10", value=float(defaults["PM10"]))
        no = st.number_input("NO", value=float(defaults["NO"]))
        no2 = st.number_input("NO2", value=float(defaults["NO2"]))
        nox = st.number_input("NOx", value=float(defaults["NOx"]))
        nh3 = st.number_input("NH3", value=float(defaults["NH3"]))
    with col2:
        co = st.number_input("CO", value=float(defaults["CO"]))
        so2 = st.number_input("SO2", value=float(defaults["SO2"]))
        o3 = st.number_input("O3", value=float(defaults["O3"]))
        benzene = st.number_input("Benzene", value=float(defaults["Benzene"]))
        toluene = st.number_input("Toluene", value=float(defaults["Toluene"]))
        xylene = st.number_input("Xylene", value=float(defaults["Xylene"]))

    if st.button("Predict AQI"):
        arr = np.array([[pm25, pm10, no, no2, nox, nh3,
                         co, so2, o3, benzene, toluene, xylene]])
        pred = model.predict(arr)[0]
        st.success(f"Predicted AQI: {pred:.2f}")

        # Simple AQI bucket
        if pred <= 50:
            bucket = "Good"
        elif pred <= 100:
            bucket = "Satisfactory"
        elif pred <= 200:
            bucket = "Moderate"
        elif pred <= 300:
            bucket = "Poor"
        elif pred <= 400:
            bucket = "Very Poor"
        else:
            bucket = "Severe"
        st.write(f"Estimated AQI Category: **{bucket}**")
