import streamlit as st
import pandas as pd
import numpy as np

# ----------------------------------------
# Utility: fuzzy column finder
# ----------------------------------------
def find_col(possible_names, df):
    cols = df.columns.str.lower().str.replace(" ", "").str.replace("_", "")
    for name in possible_names:
        for col in df.columns:
            if name in col.lower().replace(" ", "").replace("_", ""):
                return col
    return None

# ----------------------------------------
# Page setup
# ----------------------------------------
st.set_page_config(page_title="Strava Activity Dashboard", layout="wide")
st.title("🚴 Strava Activity Dashboard")

uploaded = st.file_uploader("Upload your Strava activities CSV", type=["csv"])

if uploaded is None:
    st.info("Upload activities.csv to begin.")
    st.stop()

df = pd.read_csv(uploaded)

# ----------------------------------------
# Detect core columns automatically
# ----------------------------------------
distance_col = find_col(["distance"], df)
speed_col = find_col(["averagespeed", "avgspeed", "speed"], df)
date_col = find_col(["startdatelocal", "startdate", "date"], df)
sport_col = find_col(["sporttype", "type", "activitytype"], df)

# ----------------------------------------
# Normalize / convert values
# ----------------------------------------
if distance_col:
    # Convert meters → km
    df["distance_km"] = df[distance_col] / 1000

if speed_col:
    # Strava average_speed = m/s, convert to km/h
    df["speed_kmh"] = df[speed_col] * 3.6

if date_col:
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["date_pretty"] = df["date"].dt.strftime("%a, %d %b %Y")

if sport_col:
    df["sport"] = df[sport_col].astype(str)
else:
    df["sport"] = "Activity"

# ----------------------------------------
# Sidebar filtering
# ----------------------------------------
unique_sports = sorted(df["sport"].dropna().unique())
selected_sport = st.sidebar.selectbox("🏅 Select Activity Type", unique_sports)

df_filtered = df[df["sport"] == selected_sport]

st.subheader(f"📊 Dashboard for {selected_sport}")

# ----------------------------------------
# Summary Metrics
# ----------------------------------------
col1, col2, col3 = st.columns(3)

if distance_col:
    total_km = df_filtered["distance_km"].sum()
    col1.metric("Total Distance (km)", f"{total_km:,.1f}")

if speed_col:
    avg_speed = df_filtered["speed_kmh"].mean()
    col2.metric("Avg Speed (km/h)", f"{avg_speed:,.1f}")

col3.metric("Activities Count", len(df_filtered))

# ----------------------------------------
# Charts
# ----------------------------------------
st.divider()
st.subheader("📈 Trends Over Time")

if date_col and distance_col:
    st.line_chart(df_filtered.sort_values("date"), x="date", y="distance_km", height=300)

if date_col and speed_col:
    st.line_chart(df_filtered.sort_values("date"), x="date", y="speed_kmh", height=300)

# ----------------------------------------
# Data Table
# ----------------------------------------
st.divider()
st.subheader("📄 Activity Records")

show_cols = ["date_pretty", "sport"]
if distance_col:
    show_cols.append("distance_km")
if speed_col:
    show_cols.append("speed_kmh")

st.dataframe(df_filtered[show_cols])

# ----------------------------------------
# Export filtered data
# ----------------------------------------
csv_export = df_filtered.to_csv(index=False)
st.download_button(
    "⬇️ Download Filtered CSV", 
    data=csv_export, 
    file_name=f"{selected_sport}_history.csv"
)
