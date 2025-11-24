import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

st.set_page_config(page_title="Cycling Dashboard", layout="wide")
st.title("🚴 Cycling Dashboard — YoY & MTD Insights")

# -------------------------
# File upload
# -------------------------
file = st.file_uploader("Upload cycling CSV", type=["csv"])
if not file:
    st.info("Upload a CSV to start.")
    st.stop()

# Load CSV
df = pd.read_csv(file)
df.columns = [c.strip() for c in df.columns]

# Filter cycling only
cycling_df = df[df['Activity Type'].str.lower() == 'ride'].copy()

# Parse dates and normalize
cycling_df['Activity Date'] = pd.to_datetime(cycling_df['Activity Date'], errors='coerce')
cycling_df['Year'] = cycling_df['Activity Date'].dt.year
cycling_df['Month'] = cycling_df['Activity Date'].dt.month
cycling_df['Month Name'] = cycling_df['Activity Date'].dt.strftime('%b')
cycling_df['Elapsed Hours'] = pd.to_numeric(cycling_df['Elapsed Time'], errors='coerce') / 3600.0
cycling_df['Distance KM'] = pd.to_numeric(cycling_df['Distance'], errors='coerce')
cycling_df['Average Heart Rate'] = pd.to_numeric(cycling_df['Average Heart Rate'], errors='coerce')
cycling_df['Max Heart Rate'] = pd.to_numeric(cycling_df['Max Heart Rate'], errors='coerce')

# -------------------------
# Sidebar selectors
# -------------------------
st.sidebar.header("Filters")
years = sorted(cycling_df['Year'].dropna().unique())
selected_year = st.sidebar.selectbox("Select Year for YoY comparison", years, index=len(years)-1)

months = sorted(cycling_df['Month'].dropna().unique())
selected_month = st.sidebar.selectbox("Select Month for MTD", months, index=datetime.today().month-1)

# -------------------------
# YoY Number of Rides
# -------------------------
rides_yoy = cycling_df.groupby('Year').agg({'Activity ID':'count'}).reset_index().rename(columns={'Activity ID':'Rides'})
st.subheader("Year-over-Year: Number of Rides")

bar_rides = alt.Chart(rides_yoy).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
    x=alt.X('Year:O', title='Year'),
    y=alt.Y('Rides:Q', title='Rides'),
    color=alt.condition(
        alt.datum.Year == selected_year, 
        alt.value('orange'),  # highlight selected year
        alt.value('steelblue')
    ),
    tooltip=['Year','Rides']
).properties(height=350)
st.altair_chart(bar_rides, use_container_width=True)

# -------------------------
# YoY Total Distance
# -------------------------
dist_yoy = cycling_df.groupby('Year').agg({'Distance KM':'sum'}).reset_index()
st.subheader("Year-over-Year: Total Distance (KM)")

bar_dist = alt.Chart(dist_yoy).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
    x=alt.X('Year:O', title='Year'),
    y=alt.Y('Distance KM:Q', title='Total Distance (KM)'),
    color=alt.condition(
        alt.datum.Year == selected_year, 
        alt.value('green'), 
        alt.value('lightgreen')
    ),
    tooltip=['Year','Distance KM']
).properties(height=350)
st.altair_chart(bar_dist, use_container_width=True)

# -------------------------
# MTD Summary
# -------------------------
mtd_df = cycling_df[(cycling_df['Year'] == selected_year) & (cycling_df['Month'] == selected_month)]
st.subheader(f"Month-to-Date ({datetime(1900, selected_month, 1).strftime('%b')} {selected_year}) Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Rides (MTD)", len(mtd_df))
col2.metric("Distance (KM)", round(mtd_df['Distance KM'].sum(),1))
col3.metric("Elapsed Hours", round(mtd_df['Elapsed Hours'].sum(),1))

# -------------------------
# Average rides & distance per month
# -------------------------
avg_month = cycling_df.groupby('Month').agg({'Activity ID':'count','Distance KM':'mean'}).reset_index().rename(columns={'Activity ID':'Avg Rides','Distance KM':'Avg Distance (KM)'})
st.subheader("Average Rides & Distance per Month")

line_avg = alt.Chart(avg_month).mark_line(point=True).encode(
    x=alt.X('Month:O', title='Month'),
    y=alt.Y('Avg Rides:Q', title='Avg Rides'),
    tooltip=['Month','Avg Rides','Avg Distance (KM)']
).properties(height=300)
st.altair_chart(line_avg, use_container_width=True)

# -------------------------
# Time spent YoY
# -------------------------
time_yoy = cycling_df.groupby('Year').agg({'Elapsed Hours':'sum'}).reset_index()
st.subheader("Total Time Spent Riding (Hours) YoY")

line_time = alt.Chart(time_yoy).mark_line(point=True, color='orange').encode(
    x='Year:O',
    y='Elapsed Hours:Q',
    tooltip=['Year','Elapsed Hours']
).properties(height=300)
st.altair_chart(line_time, use_container_width=True)

# -------------------------
# Heart Rate Trends
# -------------------------
st.subheader("Heart Rate Trends")
chart_hr = alt.Chart(cycling_df).mark_line(color='red').encode(
    x='Activity Date:T',
    y='Average Heart Rate:Q',
    tooltip=['Activity Date','Average Heart Rate','Max Heart Rate']
).properties(height=300)
st.altair_chart(chart_hr, use_container_width=True)
