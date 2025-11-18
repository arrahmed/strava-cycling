# ----------------------------
# FILTER CYCLING ONLY
# ----------------------------
cycling_keywords = ["ride", "ebikeride", "virtualride", "gravelride", "mountainbikeride"]

df["sport_type_clean"] = df["sport_type"].str.lower()
df_cyc = df[df["sport_type_clean"].isin(cycling_keywords)]

if df_cyc.empty:
    st.error("No cycling activities found in your Strava file.")
    st.stop()

st.subheader("🚴 Cycling Activities Detected")
st.write(f"Found **{len(df_cyc)}** cycling sessions.")

# Convert date
df_cyc["start_date"] = pd.to_datetime(df_cyc["start_date_local"], errors="coerce")

# Determine power column
power_col = None
if "average_watts" in df_cyc.columns and df_cyc["average_watts"].notna().any():
    power_col = "average_watts"
    st.success("Using REAL power meter data (average_watts).")
elif "estimated_watts" in df_cyc.columns:
    power_col = "estimated_watts"
    st.warning("Using ESTIMATED power (no power meter detected).")
else:
    st.error("No power data available for cycling. Power charts disabled.")

# ----------------------------
# CHART 1: POWER OVER TIME
# ----------------------------
if power_col:
    st.subheader("📈 Average Power Over Time")
    chart_df = df_cyc[["start_date", power_col]].set_index("start_date").sort_index()
    st.line_chart(chart_df)

# ----------------------------
# CHART 2: MONTHLY SUMMARY
# ----------------------------
st.subheader("📊 Monthly Cycling Summary")

df_cyc["month"] = df_cyc["start_date"].dt.to_period("M")
monthly = df_cyc.groupby("month").agg({
    "distance": "sum",
    "total_elevation_gain": "sum",
    power_col if power_col else "moving_time": "mean",
})

monthly.index = monthly.index.astype(str)
st.bar_chart(monthly)

# ----------------------------
# CHART 3: POWER VS HEART RATE
# ----------------------------
if power_col and "average_heartrate" in df_cyc.columns:
    st.subheader("❤️ Power vs Heart Rate (Efficiency)")
    scatter = df_cyc[[power_col, "average_heartrate"]].dropna()
    st.scatter_chart(scatter)

# ----------------------------
# CHART 4: POWER ZONES
# ----------------------------
if power_col:
    st.subheader("🔥 Estimated Power Zones")

    ftp = st.number_input("Enter FTP (Functional Threshold Power)", 100, 400, 250)

    zones = {
        "Z1 (Active Recovery)": (0, 0.55 * ftp),
        "Z2 (Endurance)": (0.56 * ftp, 0.75 * ftp),
        "Z3 (Tempo)": (0.76 * ftp, 0.90 * ftp),
        "Z4 (Threshold)": (0.91 * ftp, 1.05 * ftp),
        "Z5 (VO2 Max)": (1.06 * ftp, 1.20 * ftp),
        "Z6 (Anaerobic)": (1.21 * ftp, 1.50 * ftp),
        "Z7 (Neuromuscular)": (1.50 * ftp, 5000),
    }

    zone_counts = {}
    for zone_name, (low, high) in zones.items():
        zone_counts[zone_name] = df_cyc[df_cyc[power_col].between(low, high)].shape[0]

    st.bar_chart(pd.DataFrame.from_dict(zone_counts, orient="index", columns=["rides"]))
