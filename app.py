import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Strava Cycling Dashboard", layout="wide")

st.title("🚴 Strava Cycling Dashboard")
st.write("Upload your Strava export CSV and get an automatically generated cycling dashboard.")

# ----------------------------
# 1. Upload File
# ----------------------------
file = st.file_uploader("Upload Strava CSV", type=["csv"])

if file is None:
    st.stop()

# ----------------------------
# 2. Load CSV
# ----------------------------
try:
    df = pd.read_csv(file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# Normalize column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

st.subheader("Raw Data Preview")
st.dataframe(df.head())

# ----------------------------
# 3. Safety Checks
# ----------------------------
if "sport_type" not in df.columns:
    st.error("Your Strava CSV does not contain a 'sport_type' column. "
             "Export the full Strava 'All Activities' CSV.")
    st.stop()

# ----------------------------
# 4. Extract Cycling Only
# ----------------------------
cycling_keywords = [
    "ride", "virtualride", "ebikeride",
    "gravelride", "mountainbikeride"
]

df["sport_type_clean"] = df["sport_type"].astype(str).str.lower()

df_cyc = df[df["sport_type_clean"].isin(cycling_keywords)].copy()

if df_cyc.empty:
    st.error("No cycling activities found in your file.")
    st.stop()

st.success(f"Found **{len(df_cyc)}** cycling activities.")

# ----------------------------
# 5. Convert Dates (if possible)
# ----------------------------
date_col = None
for col in ["start_date", "start_date_local", "activity_date", "date"]:
    if col in df_cyc.columns:
        date_col = col
        break

if date_col:
    df_cyc[date_col] = pd.to_datetime(df_cyc[date_col], errors="ignore")
else:
    st.warning("No date column found — skipping date-based charts.")

# ----------------------------
# 6. Show Summary Stats
# ----------------------------
st.subheader("📊 Key Stats")

cols = st.columns(3)

# Distance
if "distance" in df_cyc.columns:
    total_km = df_cyc["distance"].sum() / 1000 if df_cyc["distance"].max() > 100 else df_cyc["distance"].sum()
    cols[0].metric("Total Distance (km)", f"{total_km:,.1f}")
else:
    cols[0].metric("Total Distance", "N/A")

# Time
if "moving_time" in df_cyc.columns:
    total_hours = df_cyc["moving_time"].sum() / 3600
    cols[1].metric("Total Moving Time (hrs)", f"{total_hours:,.1f}")
else:
    cols[1].metric("Total Moving Time", "N/A")

# Average Speed
if "average_speed" in df_cyc.columns:
    avg_speed = df_cyc["average_speed"].mean()
    cols[2].metric("Avg Speed (m/s)", f"{avg_speed:,.2f}")
else:
    cols[2].metric("Avg Speed", "N/A")

# ----------------------------
# 7. Charts
# ----------------------------

st.subheader("📈 Cycling Activity Charts")

# Distance over time
if date_col and "distance" in df_cyc.columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    df_cyc_sorted = df_cyc.sort_values(date_col)
    ax.plot(df_cyc_sorted[date_col], df_cyc_sorted["distance"])
    ax.set_ylabel("Distance")
    ax.set_title("Distance Over Time")
    st.pyplot(fig)

# Moving time histogram
if "moving_time" in df_cyc.columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df_cyc["moving_time"], bins=30)
    ax.set_title("Distribution of Moving Time (seconds)")
    st.pyplot(fig)

# Speed histogram
if "average_speed" in df_cyc.columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df_cyc["average_speed"], bins=30)
    ax.set_title("Distribution of Average Speed")
    st.pyplot(fig)

st.markdown("---")
st.write("Dashboard complete 🚀")
