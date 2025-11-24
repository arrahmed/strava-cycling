import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Cycling Dashboard", layout="wide")
st.title("🚴 Cycling Metrics Dashboard")

# -------------------------
# Upload CSV
# -------------------------
csv_file = st.file_uploader("Upload cycling activities CSV", type=["csv"])
if not csv_file:
    st.stop()

# -------------------------
# Load CSV
# -------------------------
df = pd.read_csv(csv_file)
df.columns = [c.strip() for c in df.columns]

# Filter for rides
df['Activity Type'] = df['Activity Type'].str.strip().str.lower()
cycling_df = df[df['Activity Type'] == 'ride'].copy()

# -------------------------
# Preprocess
# -------------------------
cycling_df['Activity Date'] = pd.to_datetime(cycling_df['Activity Date'])
cycling_df['Elapsed Hours'] = cycling_df['Elapsed Time'] / 3600  # if seconds
cycling_df['Distance KM'] = cycling_df['Distance']  # assuming km

cycling_df['Year'] = cycling_df['Activity Date'].dt.year
cycling_df['Month'] = cycling_df['Activity Date'].dt.month
cycling_df['Month_Year'] = cycling_df['Activity Date'].dt.to_period('M').dt.to_timestamp()

# Sidebar filters
st.sidebar.header("Filters")
years = sorted(cycling_df['Year'].unique())
selected_year = st.sidebar.selectbox("Select Year", years, index=len(years)-1)
months = sorted(cycling_df['Month'].unique())
selected_month = st.sidebar.selectbox("Select Month", months, index=0)

df_filtered = cycling_df[cycling_df['Year'] == selected_year]

# -------------------------
# Aggregations
# -------------------------
monthly_stats = df_filtered.groupby('Month_Year').agg(
    rides=('Activity ID', 'count'),
    distance_km=('Distance KM', 'sum'),
    elapsed_hours=('Elapsed Hours', 'sum'),
    avg_hr=('Average Heart Rate', 'mean'),
    max_hr=('Max Heart Rate', 'mean')
).reset_index()

yearly_stats = df_filtered.groupby('Year').agg(
    rides=('Activity ID', 'count'),
    distance_km=('Distance KM', 'sum'),
    elapsed_hours=('Elapsed Hours', 'sum'),
    avg_hr=('Average Heart Rate', 'mean'),
    max_hr=('Max Heart Rate', 'mean')
).reset_index()

# MTD stats
df_mtd = df_filtered[df_filtered['Month'] == selected_month]
mtd_stats = df_mtd.agg(
    rides=('Activity ID', 'count'),
    distance_km=('Distance KM', 'sum'),
    elapsed_hours=('Elapsed Hours', 'sum')
)

# -------------------------
# Overview metrics
# -------------------------
st.subheader(f"Overview Metrics ({selected_year})")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Rides", f"{yearly_stats['rides'].sum()}")
col2.metric("Total Distance (km)", f"{yearly_stats['distance_km'].sum():.1f}")
col3.metric("Total Riding Hours", f"{yearly_stats['elapsed_hours'].sum():.1f}")
col4.metric("Average HR (bpm)", f"{yearly_stats['avg_hr'].mean():.1f}")

# -------------------------
# YoY comparison charts
# -------------------------
st.subheader("Year-over-Year Comparison")
# Rides YoY
rides_chart = alt.Chart(df.groupby('Year').agg(rides=('Activity ID','count')).reset_index()
                       ).mark_bar().encode(
    x='Year:O', y='rides:Q', tooltip=['Year','rides']
).properties(height=300)
st.altair_chart(rides_chart, use_container_width=True)

# Distance YoY
distance_chart = alt.Chart(df.groupby('Year').agg(distance_km=('Distance KM','sum')).reset_index()
                          ).mark_bar(color='orange').encode(
    x='Year:O', y='distance_km:Q', tooltip=['Year','distance_km']
).properties(height=300)
st.altair_chart(distance_chart, use_container_width=True)

# -------------------------
# MTD comparison
# -------------------------
st.subheader(f"Month-to-Date Comparison ({selected_month}/{selected_year})")
st.write(f"Rides: {mtd_stats['rides']}, Distance (km): {mtd_stats['distance_km']:.1f}, Riding Hours: {mtd_stats['elapsed_hours']:.1f}")

# -------------------------
# Monthly trends
# -------------------------
st.subheader("Monthly Trends")
rides_month_chart = alt.Chart(monthly_stats).mark_line(point=True).encode(
    x='Month_Year:T', y='rides:Q', tooltip=['Month_Year','rides']
)
st.altair_chart(rides_month_chart, use_container_width=True)

distance_month_chart = alt.Chart(monthly_stats).mark_line(point=True, color='green').encode(
    x='Month_Year:T', y='distance_km:Q', tooltip=['Month_Year','distance_km']
)
st.altair_chart(distance_month_chart, use_container_width=True)

time_month_chart = alt.Chart(monthly_stats).mark_line(point=True, color='purple').encode(
    x='Month_Year:T', y='elapsed_hours:Q', tooltip=['Month_Year','elapsed_hours']
)
st.altair_chart(time_month_chart, use_container_width=True)

# -------------------------
# Heart Rate Trends
# -------------------------
st.subheader("Heart Rate Trends")
avg_hr_chart = alt.Chart(monthly_stats).mark_line(point=True, color='red').encode(
    x='Month_Year:T', y='avg_hr:Q', tooltip=['Month_Year','avg_hr']
)
st.altair_chart(avg_hr_chart, use_container_width=True)

max_hr_chart = alt.Chart(monthly_stats).mark_line(point=True, color='blue').encode(
    x='Month_Year:T', y='max_hr:Q', tooltip=['Month_Year','max_hr']
)
st.altair_chart(max_hr_chart, use_container_width=True)

# -------------------------
# Averages per month & year
# -------------------------
st.subheader("Average Metrics")
st.write(f"Avg rides per month ({selected_year}): {monthly_stats['rides'].mean():.1f}")
st.write(f"Avg distance per month ({selected_year}): {monthly_stats['distance_km'].mean():.1f} km")
st.write(f"Avg riding hours per month ({selected_year}): {monthly_stats['elapsed_hours'].mean():.1f} h")
st.write(f"Avg rides per year (all years): {df.groupby('Year')['Activity ID'].count().mean():.1f}")
st.write(f"Avg distance per year (all years): {df.groupby('Year')['Distance KM'].sum().mean():.1f} km")
