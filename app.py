
import streamlit as st
import pandas as pd

st.title("🚴 Cycling Performance Dashboard (Strava Upload)")

st.write("""
Upload your Strava exported **activities.csv** file and optionally a
**sleep.csv** file from Apple Health, Garmin, Fitbit, etc.
""")

# File uploads
strava_file = st.file_uploader("Upload Strava activities.csv", type=["csv"])
sleep_file = st.file_uploader("Upload Sleep data (optional)", type=["csv"])

if not strava_file:
    st.info("👆 Upload a Strava CSV export to begin")
    st.stop()

# Load data
df = pd.read_csv(strava_file)

# Clean column names
df.columns = [c.lower().replace(" ", "_") for c in df.columns]

st.subheader("Preview of uploaded Strava data")
st.dataframe(df.head())

# Detect power meter
has_power = "average_watts" in df.columns
estimated_power_col = "estimated_watts" if "estimated_watts" in df.columns else None

if has_power:
    st.success("Detected **real power meter** data (average_watts present).")
elif estimated_power_col:
    st.warning("No real power meter detected. ⚠ Using estimated watts.")
else:
    st.error("No power data found. Some charts will be unavailable.")

# Basic chart example (only if power exists)
if has_power or estimated_power_col:
    watt_col = "average_watts" if has_power else estimated_power_col

    st.subheader("Power Over Time")
    df["start_date"] = pd.to_datetime(df["start_date_local"])
    st.line_chart(df.set_index("start_date")[watt_col])
