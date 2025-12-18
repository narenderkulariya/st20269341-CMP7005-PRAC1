
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

# ---------- NEW: REGION MAP SUPPORT ---------- #

region_map = {
    "Delhi": "North",
    "Jaipur": "North",
    "Lucknow": "North",
    "Amritsar": "North",
    "Chandigarh": "North",
    "Gurugram": "North",
    "Patna": "East",
    "Guwahati": "East",
    "Kolkata": "East",
    "Jorapokhar": "East",
    "Talcher": "East",
    "Brajrajnagar": "East",
    "Aizawl": "North‑East",
    "Mumbai": "West",
    "Ahmedabad": "West",
    "Bhopal": "Central",
    "Chennai": "South",
    "Bengaluru": "South",
    "Hyderabad": "South",
    "Coimbatore": "South",
    "Kochi": "South",
    "Ernakulam": "South",
    "Thiruvananthapuram": "South",
    "Visakhapatnam": "South",
    "Amaravati": "South",
    "Shillong": "North‑East"
}
df["Region"] = df["City"].map(region_map).fillna("Other")

# coordinates for your 26 cities (lat, lon)
CITY_COORDS = {
    "Ahmedabad": (23.0333, 72.6167),
    "Amaravati": (16.5062, 80.6480),
    "Amritsar": (31.6340, 74.8723),
    "Aizawl": (23.7271, 92.7176),
    "Bengaluru": (12.9716, 77.5946),
    "Bhopal": (23.2599, 77.4126),
    "Brajrajnagar": (21.8167, 83.9167),
    "Chandigarh": (30.7333, 76.7794),
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Delhi": (28.7041, 77.1025),
    "Ernakulam": (9.9816, 76.2999),
    "Guwahati": (26.1445, 91.7362),
    "Gurugram": (28.4595, 77.0266),
    "Hyderabad": (17.3850, 78.4867),
    "Jaipur": (26.9124, 75.7873),
    "Jorapokhar": (23.7167, 86.4000),
    "Kochi": (9.9312, 76.2673),
    "Kolkata": (22.5726, 88.3639),
    "Lucknow": (26.8467, 80.9462),
    "Mumbai": (19.0760, 72.8777),
    "Patna": (25.5941, 85.1376),
    "Shillong": (25.5788, 91.8933),
    "Talcher": (20.9500, 85.2000),
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Visakhapatnam": (17.6868, 83.2185)
}
                      
# ============== NEW PAGE 3: REGIONS & MAP ============== #
elif mode == "🌍 Regions & Map":
     st.title("🌍 Regional Comparison & City Map")

      filtered.empty:
        st.warning("No data for the current filters. Please select more cities or years.")
    else:
        st.markdown(
            "This page compares **North vs South (and other regions)** and "
            "shows where the monitoring cities are located on the world map."
        )

        # ----- Part 1: Regional AQI comparison ----- #
        st.subheader("Regional AQI comparison")

        region_aqi = (
            filtered
            .groupby("Region")["AQI"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(region_aqi)

        st.caption(
            "Bars show the average AQI by region for the current city and year filters. "
            "Use the sidebar to switch which cities and years are included."
        )

        if "AQIBucket" in filtered.columns:
            st.subheader("Share of 'Severe' days by region")
            severe_share = (
                filtered.assign(is_severe=filtered["AQIBucket"] == "Severe")
                .groupby("Region")["is_severe"]
                .mean()
                .sort_values(ascending=False)
            )
            st.bar_chart(severe_share)
            st.caption("Higher values mean a larger fraction of days classified as 'Severe'.")
        else:
            st.info("AQIBucket column not available to compute Severe‑day share.")

        st.markdown("---")

        # ----- Part 2: World map with city locations ----- #
        st.subheader("City locations on the map")

        # Building here a DataFrame with lat/lon and mean AQI for each city in the filtered set
        city_mean_aqi = filtered.groupby("City")["AQI"].mean()

        map_rows = []
        for city, mean_aqi in city_mean_aqi.items():
            if city in CITY_COORDS:
                lat, lon = CITY_COORDS[city]
                map_rows.append(
                    {"City": city, "lat": lat, "lon": lon, "AQI": mean_aqi}
                )

        if not map_rows:
            st.info("No coordinates found for the selected cities.")
        else:
            map_df = pd.DataFrame(map_rows)
            st.map(map_df, latitude="lat", longitude="lon")
            st.caption(
                "Each dot marks a monitoring city; position is approximate. "
                "You can pan/zoom to explore the geography."
            )

# ============== PAGE 3: CITY REPORT CARD ============== #
elif mode == "📘 City Report Card":
    st.title("📘 City Report Card")

    if filtered.empty:
        st.warning("No data for the current filters. Please select more cities or years.")
    else:
        city_choice = st.selectbox("Choose a city", sorted(filtered["City"].unique()))
        city_df = filtered[filtered["City"] == city_choice].copy().sort_values("Date")

        st.markdown(f"### Overview for **{city_choice}**")

        # 1) High-level metrics
        avg_aqi = city_df["AQI"].mean()
        best_year = city_df.groupby("Year")["AQI"].mean().idxmin()
        worst_year = city_df.groupby("Year")["AQI"].mean().idxmax()

        c1, c2, c3 = st.columns(3)
        c1.metric("Average AQI", f"{avg_aqi:.1f}")
        c2.metric("Best year (lowest AQI)", int(best_year))
        c3.metric("Worst year (highest AQI)", int(worst_year))

        st.markdown("---")

        # 2) AQI bucket distribution
        st.subheader("AQI bucket distribution")
        if "AQIBucket" in city_df.columns:
            bucket_counts = city_df["AQIBucket"].value_counts()
            st.bar_chart(bucket_counts)
            st.caption("Number of days in each AQI category for this city.")
        else:
            st.info("AQIBucket column not available.")

        st.markdown("---")

        # 3) Dominant pollutants
        st.subheader("Dominant pollutants (average levels)")
        pollutant_means = city_df[FEATURE_COLS].mean().sort_values(ascending=False)
        st.bar_chart(pollutant_means)
        st.caption("Pollutants with higher average values dominate the city’s air quality.")

        st.markdown("---")

        # 4) Cleanest vs dirtiest day
        st.subheader("Cleanest vs dirtiest recorded day")

        best_row = city_df.loc[city_df["AQI"].idxmin()]
        worst_row = city_df.loc[city_df["AQI"].idxmax()]

        colA, colB = st.columns(2)

        def day_card(row, label, icon):
            st.markdown(f"#### {icon} {label}")
            st.write(f"**Date:** {row['Date'].date()}")
            st.write(f"**AQI:** {row['AQI']:.0f} ({row.get('AQIBucket', 'N/A')})")
            st.caption(
                "Key pollutants: " +
                ", ".join(f"{p} = {row[p]:.1f}" for p in ["PM2.5", "PM10", "NO2", "SO2"])
            )

        with colA:
            day_card(best_row, "Cleanest day", "🌿")
        with colB:
            day_card(worst_row, "Dirtiest day", "🏭")


# --------- PAGE 4: MODELLING & PREDICTION --------- #
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
