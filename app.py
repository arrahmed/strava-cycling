import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Strava Dashboard", layout="wide")

st.title("🚴 Strava Activity Dashboard (Altair Edition)")

uploaded = st.file_uploader("Upload your Strava activity CSV", type=["csv"])

if uploaded is None:
    st.info("Please upload your Strava export file to continue.")
    st.stop()

# ---------------------------------------------------------
# LOAD + CLEAN DATA
# ---------------------------------------------------------
df = pd.read_csv(uploaded)

# Auto-detect possible column names (Strava exports vary a lot)
def best_column(options):
    for col in options:
        if col in df.columns:
            return col
    return None

col_name        = best_column(["name", "activity_name"])
col_type        = best_column(["sport_type", "type"])
col_distance    = best_column(["distance", "Distance"])
col_speed       = best_column(["average_speed", "avg_speed", "Average Speed"])
col_moving      = best_column(["moving_time", "Moving Time"])
col_start       = best_column(["start_date", "start_date_local", "Start Date"])

# Convert and clean
if col_start:
    df["date"] = pd.to_datetime(df[col_start])
else:
    df["date"] = pd.date_range("2023-01-01", periods=len(df))

if col_distance:
    df["distance_km"] = df[col_distance] / 1000  # meters → km

if col_speed:
    df["speed_kmh"] = df[col_speed] * 3.6        # m/s → km/h

if col_moving:
    df["moving_hours"] = df[col_moving] / 3600   # seconds → hours

if col_type:
    df["activity_type"] = df[col_type].astype(str)
else:
    df["activity_type"] = "Activity"

# ---------------------------------------------------------
# SIDEBAR FILTER
# ---------------------------------------------------------
st.sidebar.header("Filters")

activities = df["activity_type"].unique().tolist()
selected_activity = st.sidebar.selectbox("Activity type", activities)

df_filtered = df[df["activity_type"] == selected_activity]

st.subheader(f"📌 Showing: **{selected_activity}** ({len(df_filtered)} activities)")

# ---------------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------------
with st.expander("📄 Raw Data Preview"):
    st.dataframe(df_filtered.head())

# ---------------------------------------------------------
# ALTAR CHART HELPERS
# ---------------------------------------------------------
def line_chart(data, x, y, title, y_label):
    chart = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X(x, title="Date"),
            y=alt.Y(y, title=y_label),
            tooltip=[x, y]
        )
        .properties(title=title, height=300)
        .interactive()
    )
    return chart

# ---------------------------------------------------------
# CHARTS
# ---------------------------------------------------------
charts = []

if "distance_km" in df_filtered:
    charts.append(
        line_chart(
            df_filtered,
            "date:T",
            "distance_km:Q",
            f"📏 Distance Over Time ({selected_activity})",
            "Distance (km)"
        )
    )

if "speed_kmh" in df_filtered:
    charts.append(
        line_chart(
            df_filtered,
            "date:T",
            "speed_kmh:Q",
            f"⚡ Avg Speed Over Time ({selected_activity})",
            "Speed (km/h)"
        )
    )

if "moving_hours" in df_filtered:
    charts.append(
        line_chart(
            df_filtered,
            "date:T",
            "moving_hours:Q",
            f"⏱️ Moving Time Over Time ({selected_activity})",
            "Hours"
        )
    )

# ---------------------------------------------------------
# DISPLAY CHARTS
# ---------------------------------------------------------
for chart in charts:
    st.altair_chart(chart, use_container_width=True)

