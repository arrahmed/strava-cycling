import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

st.set_page_config(page_title="Cycling Stats Dashboard", layout="wide")
st.title("🚴 Cycling Dashboard — YoY & MTD Comparisons")

# -------------------------
# File upload
# -------------------------
file = st.file_uploader("Upload cycling CSV", type=["csv"])
if not file:
    st.info("Upload a CSV to start.")
    st.stop()

# Load CSV
df = pd.read_csv(file)
df.columns = [c.strip() for c in df.columns]  # clean whitespace

# -------------------------
# Filter cycling only
# -------------------------
cycling_df = df[df['Activity Type'].str.lower() == 'ride'].copy()

# Parse dates
cycling_df['Activity Date'] = pd.to_datetime(cycling_df['Activity Date'], errors='coerce')
cycling_df['Year'] = cycling_df['Activity Date'].dt.year
cycling_df['Month'] = cycling_df['Activity Date'].dt.month
cycling_df['Month Name'] = cycling_df['Activity Date'].dt.strftime('%b')
cycling_df['Elapsed Hours'] = pd.to_numeric(cycling_df['Elapsed Time'], errors='coerce') / 3600.0
cycling_df['Distance KM'] = pd.to_numeric(cycling_df['Distance'], errors='coerce')

# -------------------------
# YoY: number of rides
# -------------------------
rides_yoy = cycling_df.groupby('Year').agg({'Activity ID':'count'}).reset_index().rename(columns={'Activity ID':'Rides'})
st.subheader("Year-over-Year: Number of Rides")
chart_rides = alt.Chart(rides_yoy).mark_bar().encode(
    x='Year:O',
    y='Rides:Q',
    tooltip=['Year','Rides']
)
st.altair_chart(chart_rides, use_container_width=True)

# -------------------------
# YoY: total distance
# -------------------------
dist_yoy = cycling_df.groupby('Year').agg({'Distance KM':'sum'}).reset_index()
st.subheader("Year-over-Year: Total Distance (KM)")
chart_dist = alt.Chart(dist_yoy).mark_bar(color='green').encode(
    x='Year:O',
    y='Distance KM:Q',
    tooltip=['Year','Distance KM']
)
st.altair_chart(chart_dist, use_container_width=True)

# -------------------------
# MTD: number of rides & distance (for current month)
# -------------------------
today = datetime.today()
cycling_df['YearMonth'] = cycling_df['Activity Date'].dt.to_period('M')

current_month = today.strftime('%Y-%m')
mtd_df = cycling_df[cycling_df['YearMonth'] == today.strftime('%Y-%m')]

st.subheader(f"Month-to-Date ({today.strftime('%b %Y')}): Summary")
col1, col2 = st.columns(2)
col1.metric("Rides (MTD)", len(mtd_df))
col2.metric("Distance (KM, MTD)", round(mtd_df['Distance KM'].sum(),1))

# -------------------------
# Average rides & distance per month & year
# -------------------------
avg_month = cycling_df.groupby('Month').agg({'Activity ID':'count','Distance KM':'mean'}).reset_index().rename(columns={'Activity ID':'Avg Rides','Distance KM':'Avg Distance'})
st.subheader("Average Rides / Month & Distance")
chart_avg_month = alt.Chart(avg_month).mark_line(point=True).encode(
    x='Month:O',
    y='Avg Rides:Q',
    tooltip=['Month','Avg Rides','Avg Distance']
)
st.altair_chart(chart_avg_month, use_container_width=True)

# -------------------------
# Time spent on rides comparison
# -------------------------
time_yoy = cycling_df.groupby('Year').agg({'Elapsed Hours':'sum'}).reset_index()
st.subheader("Total Time Spent Riding (Hours) YoY")
chart_time = alt.Chart(time_yoy).mark_line(point=True,color='orange').encode(
    x='Year:O',
    y='Elapsed Hours:Q',
    tooltip=['Year','Elapsed Hours']
)
st.altair_chart(chart_time, use_container_width=True)

# -------------------------
# Heart rate overview
# -------------------------
cycling_df['Average Heart Rate'] = pd.to_numeric(cycling_df['Average Heart Rate'], errors='coerce')
cycling_df['Max Heart Rate'] = pd.to_numeric(cycling_df['Max Heart Rate'], errors='coerce')

st.subheader("Heart Rate Trends")
chart_hr_avg = alt.Chart(cycling_df).mark_line(color='red').encode(
    x='Activity Date:T',
    y='Average Heart Rate:Q',
    tooltip=['Activity Date','Average Heart Rate','Max Heart Rate']
)
st.altair_chart(chart_hr_avg, use_container_width=True)
