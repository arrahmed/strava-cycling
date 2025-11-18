# app.py
import streamlit as st
import pandas as pd
import numpy as np
import zipfile, io, xml.etree.ElementTree as ET
import altair as alt
from datetime import timedelta

st.set_page_config(page_title="Strava+Sleep Performance", layout="wide")
st.title("🚴 Strava + Sleep Performance Dashboard")
st.write("Upload a Strava activities CSV (any filename). Optional: upload Apple Health (ZIP/XML/CSV) to add sleep data.")

# ---------------------------
# Utility helpers
# ---------------------------
def find_col(df, candidates):
    """Return first matching column name from candidates (fuzzy match)."""
    cols = list(df.columns)
    lowcols = [c.lower().replace(" ", "").replace("_", "") for c in cols]
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        for i, lc in enumerate(lowcols):
            if key in lc or lc in key:
                return cols[i]
    return None

def load_csv_bytes(fileobj):
    try:
        return pd.read_csv(fileobj)
    except Exception as e:
        # fallback: try with latin1
        return pd.read_csv(fileobj, encoding="latin1")

def extract_sleep_from_zip(fileobj):
    """Given a file-like zip from Apple Health export, extract SleepAnalysis records."""
    try:
        with zipfile.ZipFile(fileobj, "r") as z:
            xml_files = [f for f in z.namelist() if f.endswith(".xml")]
            if not xml_files:
                return None
            xml_content = z.read(xml_files[0])
            root = ET.fromstring(xml_content)
            records = []
            for r in root.findall(".//Record"):
                if r.attrib.get("type") == "HKCategoryTypeIdentifierSleepAnalysis":
                    records.append({
                        "start": r.attrib.get("startDate"),
                        "end": r.attrib.get("endDate"),
                        "value": r.attrib.get("value")
                    })
            if records:
                sdf = pd.DataFrame(records)
                sdf["start"] = pd.to_datetime(sdf["start"], errors="coerce")
                sdf["end"] = pd.to_datetime(sdf["end"], errors="coerce")
                sdf["duration_hours"] = (sdf["end"] - sdf["start"]).dt.total_seconds() / 3600.0
                return sdf
    except Exception:
        return None
    return None

def extract_sleep_from_xml(fileobj):
    try:
        xml_content = fileobj.read()
        root = ET.fromstring(xml_content)
        records = []
        for r in root.findall(".//Record"):
            if r.attrib.get("type") == "HKCategoryTypeIdentifierSleepAnalysis":
                records.append({
                    "start": r.attrib.get("startDate"),
                    "end": r.attrib.get("endDate"),
                    "value": r.attrib.get("value")
                })
        if records:
            sdf = pd.DataFrame(records)
            sdf["start"] = pd.to_datetime(sdf["start"], errors="coerce")
            sdf["end"] = pd.to_datetime(sdf["end"], errors="coerce")
            sdf["duration_hours"] = (sdf["end"] - sdf["start"]).dt.total_seconds() / 3600.0
            return sdf
    except Exception:
        return None
    return None

def estimate_power_from_physics(speed_m_s, grade, mass_kg=75, cda=0.32, crr=0.004, rho=1.226):
    """Rough physics-based power estimator:
       P = rolling + gravity + aerodynamic
       v = speed (m/s), grade is fraction (e.g. 0.05)
    """
    # Rolling resistance
    P_rr = mass_kg * 9.80665 * crr * speed_m_s
    # Gravity on grade
    P_g = mass_kg * 9.80665 * grade * speed_m_s
    # Aero
    P_aero = 0.5 * rho * cda * (speed_m_s ** 3)
    P = P_rr + P_g + P_aero
    return P

def compute_time_in_zones(series_watts, ftp):
    zones = {
        "Z1 Recovery": (0.0, 0.55),
        "Z2 Endurance": (0.56, 0.75),
        "Z3 Tempo": (0.76, 0.90),
        "Z4 Threshold": (0.91, 1.05),
        "Z5 VO2": (1.06, 1.20),
        "Z6 Anaerobic": (1.21, 2.00),
    }
    counts = {}
    for name, (low, high) in zones.items():
        lw = low * ftp
        hw = high * ftp
        counts[name] = int(((series_watts >= lw) & (series_watts <= hw)).sum())
    return counts

# ---------------------------
# File uploads
# ---------------------------
col1, col2 = st.columns([2, 1])
with col1:
    strava_file = st.file_uploader("Strava activities CSV (any filename) — required", type=["csv"])
with col2:
    sleep_file = st.file_uploader("Optional: Apple Health ZIP / XML / sleep CSV", type=["zip", "xml", "csv"])

if not strava_file:
    st.info("Upload a Strava CSV to begin. Use the mock CSV if testing.")
    st.stop()

# ---------------------------
# Load and normalize Strava CSV
# ---------------------------
try:
    df_raw = load_csv_bytes(strava_file)
except Exception as e:
    st.error(f"Could not read Strava CSV: {e}")
    st.stop()

# keep original columns
orig_cols = list(df_raw.columns)
# normalize lower, stripped column names mapping
df = df_raw.copy()
df.columns = [str(c).strip() for c in df.columns]

# create a lowercase no-space index for fuzzy finding
simple_cols = {c: c.lower().replace(" ", "").replace("_", "") for c in df.columns}

# ---------------------------
# Detect if CSV is GPS-sample vs activity-summary
# If GPS-sample: compress to per-activity summary
# ---------------------------
has_lat = any(k in simple_cols.values() for k in ("latitude", "lat", "lon", "longitude"))
# summary detection
distance_col = find_col(df, ["distance"])
start_col = find_col(df, ["start_date_local", "start_date", "timestamp", "activity_date", "date"])
moving_col = find_col(df, ["moving_time", "elapsed_time"])
avg_speed_col = find_col(df, ["average_speed", "avg_speed", "averagespeed"])
power_col = find_col(df, ["weighted_average_watts", "average_watts", "avg_watts", "average_power"])
max_power_col = find_col(df, ["max_watts", "maxpower"])
hr_col = find_col(df, ["average_heartrate", "avg_heartrate", "heartrate"])
elev_col = find_col(df, ["total_elevation_gain", "elevation_gain", "elev"])

# if we detect lat/lon columns present, assume GPS-level samples
if has_lat and (distance_col is None or avg_speed_col is None):
    st.info("Detected GPS-level export (per-point rows). Aggregating to per-activity summary...")
    # try to find an activity id to group by
    id_cand = find_col(df, ["id", "activityid", "activity_id", "fitid"])
    if id_cand is None:
        # try grouping by day + name as fallback
        name_col = find_col(df, ["name", "activity_name"])
        if name_col:
            df['_group'] = df[name_col].astype(str) + "_" + df[start_col].astype(str) if start_col else df[name_col].astype(str)
        else:
            df['_group'] = (df.index // 1000).astype(str)  # fallback arbitrary chunking
    else:
        df['_group'] = df[id_cand].astype(str)
    groups = df.groupby('_group')
    summary_rows = []
    for gname, g in groups:
        # date = min timestamp found
        date_val = pd.to_datetime(g[start_col], errors="coerce") if start_col else pd.NaT
        if isinstance(date_val, pd.Series):
            date_val = date_val.min()
        # distance
        dist = g[distance_col].sum() if distance_col in g else (g.get('distance', pd.Series(dtype=float)).sum() if 'distance' in g else np.nan)
        # moving time approximate from timestamps if present
        if 'timestamp' in [c.lower() for c in g.columns] or 'time' in [c.lower() for c in g.columns]:
            # try to compute from min/max timestamp
            ts_col = next((c for c in g.columns if 'timestamp' in c.lower() or c.lower()=='time'), None)
            try:
                tmin = pd.to_datetime(g[ts_col], errors="coerce").min()
                tmax = pd.to_datetime(g[ts_col], errors="coerce").max()
                moving = (tmax - tmin).total_seconds() if pd.notna(tmin) and pd.notna(tmax) else np.nan
            except Exception:
                moving = np.nan
        else:
            moving = np.nan
        # elevation gain
        elev = g[elev_col].sum() if elev_col in g else np.nan
        # average speed approximate
        avgspd = g[avg_speed_col].mean() if avg_speed_col in g else ( (dist/ moving) if pd.notna(dist) and pd.notna(moving) and moving>0 else np.nan)
        # mean heartrate if exists
        mean_hr = g[hr_col].mean() if hr_col in g else np.nan
        summary_rows.append({
            "group": gname,
            "start_date": date_val,
            "distance_m": dist,
            "moving_time_s": moving,
            "elevation_m": elev,
            "avg_speed_mps": avgspd,
            "avg_hr": mean_hr
        })
    df_sum = pd.DataFrame(summary_rows)
    # rename for consistency
    if "distance_m" in df_sum.columns:
        df_sum = df_sum.rename(columns={"distance_m": "distance"})
    if "avg_speed_mps" in df_sum.columns:
        df_sum = df_sum.rename(columns={"avg_speed_mps": "average_speed"})
    if "moving_time_s" in df_sum.columns:
        df_sum = df_sum.rename(columns={"moving_time_s": "moving_time"})
    # reassign df summary
    df = df_sum
    # update detected columns
    distance_col = "distance" if "distance" in df.columns else None
    avg_speed_col = "average_speed" if "average_speed" in df.columns else None
    start_col = "start_date" if "start_date" in df.columns else None
    hr_col = "avg_hr" if "avg_hr" in df.columns else None
    elev_col = "elevation_m" if "elevation_m" in df.columns else None

# ---------------------------
# Make sure we have date column
# ---------------------------
if start_col and start_col in df.columns:
    df['date'] = pd.to_datetime(df[start_col], errors='coerce')
else:
    # try other heuristics
    alt_date = find_col(df, ["date","activity_date","start"])
    if alt_date:
        df['date'] = pd.to_datetime(df[alt_date], errors='coerce')
    else:
        df['date'] = pd.NaT

# ---------------------------
# Normalized columns and units
# ---------------------------
# distance -> km
if distance_col and distance_col in df.columns:
    # if values look like meters (max > 1000) convert to km
    try:
        maxd = df[distance_col].dropna().astype(float).max()
        if pd.notna(maxd) and maxd > 1000:
            df['distance_km'] = df[distance_col].astype(float) / 1000.0
        else:
            df['distance_km'] = df[distance_col].astype(float)
    except Exception:
        df['distance_km'] = pd.to_numeric(df[distance_col], errors='coerce')
else:
    df['distance_km'] = np.nan

# average speed -> km/h (Strava often m/s)
if avg_speed_col and avg_speed_col in df.columns:
    try:
        df['speed_kmh'] = df[avg_speed_col].astype(float) * 3.6
    except Exception:
        df['speed_kmh'] = pd.to_numeric(df[avg_speed_col], errors='coerce')
else:
    df['speed_kmh'] = np.nan

# elevation
if elev_col and elev_col in df.columns:
    df['elevation_m'] = pd.to_numeric(df[elev_col], errors='coerce')
else:
    df['elevation_m'] = np.nan

# heart rate
if hr_col and hr_col in df.columns:
    df['hr'] = pd.to_numeric(df[hr_col], errors='coerce')
else:
    df['hr'] = np.nan

# power detection (prefer weighted/average)
power_candidates = [c for c in df.columns if any(s in c.lower() for s in ("weighted_average_watts","average_watts","avg_watts","average_power","estimated_watts"))]
detected_power_col = power_candidates[0] if power_candidates else None
if detected_power_col:
    df['power_w'] = pd.to_numeric(df[detected_power_col], errors='coerce')
else:
    df['power_w'] = np.nan

# max power
if max_power_col and max_power_col in df.columns:
    df['power_max_w'] = pd.to_numeric(df[max_power_col], errors='coerce')
else:
    df['power_max_w'] = np.nan

# moving time seconds
if moving_col and moving_col in df.columns:
    df['moving_time_s'] = pd.to_numeric(df[moving_col], errors='coerce')
else:
    df['moving_time_s'] = np.nan

# ride name
name_col = find_col(df, ["name","title","activity"])
if name_col:
    df['ride_name'] = df[name_col].astype(str)
else:
    df['ride_name'] = df.index.astype(str)

# sport type column
sport_col = find_col(df, ["sport_type","type","activitytype"])
if sport_col:
    df['sport'] = df[sport_col].astype(str)
else:
    # fallback: if all distances non-null and no sport column, assume Ride
    df['sport'] = "Ride"

# ---------------------------
# If no power data present, try to estimate per-ride power
# using physics heuristic (from speed, elevation, distance)
# ---------------------------
if df['power_w'].isna().all():
    st.info("No measured power detected. Estimating power using speed/elevation heuristics (approximate).")
    # compute grade per ride = elevation / distance (both in meters)
    # ensure distance in meters
    df['distance_m'] = df['distance_km'] * 1000.0
    # grade: elevation_m / distance_m (clamped)
    df['grade'] = np.where((df['distance_m']>0) & pd.notna(df['elevation_m']), df['elevation_m'] / df['distance_m'], 0.0)
    # convert speed to m/s
    df['speed_m_s'] = (df['speed_kmh'] / 3.6).fillna(0.0)
    # estimate power per ride as physics estimate (avg)
    df['power_w'] = df.apply(lambda r: estimate_power_from_physics(r['speed_m_s'] if pd.notna(r['speed_m_s']) else 5.0,
                                                                  r['grade'] if pd.notna(r['grade']) else 0.0), axis=1)

# ---------------------------
# Prepare date buckets: week/month/year
# ---------------------------
df['date_only'] = pd.to_datetime(df['date']).dt.date
df['week'] = pd.to_datetime(df['date']).dt.to_period('W').astype(str)
df['month'] = pd.to_datetime(df['date']).dt.to_period('M').astype(str)
df['year'] = pd.to_datetime(df['date']).dt.to_period('Y').astype(str)

# ---------------------------
# Sleep file parsing
# ---------------------------
sleep_df = None
if sleep_file:
    nf = sleep_file.name.lower()
    try:
        if nf.endswith('.zip'):
            sleep_df = extract_sleep_from_zip(sleep_file)
        elif nf.endswith('.xml'):
            sleep_df = extract_sleep_from_xml(sleep_file)
        elif nf.endswith('.csv'):
            try:
                s = pd.read_csv(sleep_file)
                # try to find start/end
                start = find_col(s, ["start","start_date","starttime","begin"])
                end = find_col(s, ["end","end_date","endtime","finish"])
                dur = find_col(s, ["duration","duration_seconds","seconds"])
                if start:
                    s['start'] = pd.to_datetime(s[start], errors='coerce')
                if end:
                    s['end'] = pd.to_datetime(s[end], errors='coerce')
                if dur and 'duration_hours' not in s.columns:
                    try:
                        s['duration_hours'] = pd.to_numeric(s[dur], errors='coerce') / 3600.0
                    except Exception:
                        pass
                if 'duration_hours' not in s.columns and 'start' in s.columns and 'end' in s.columns:
                    s['duration_hours'] = (s['end'] - s['start']).dt.total_seconds() / 3600.0
                sleep_df = s
            except Exception:
                sleep_df = None
    except Exception:
        sleep_df = None

# normalize sleep into per-night aggregated date (use end date as sleep date)
sleep_per_day = None
if sleep_df is not None and not sleep_df.empty:
    if 'end' in sleep_df.columns:
        sleep_df['end_dt'] = pd.to_datetime(sleep_df['end'], errors='coerce')
    elif 'start' in sleep_df.columns:
        sleep_df['end_dt'] = pd.to_datetime(sleep_df['start'], errors='coerce')
    else:
        sleep_df['end_dt'] = None
    sleep_df['sleep_date'] = pd.to_datetime(sleep_df['end_dt']).dt.date
    if 'duration_hours' not in sleep_df.columns and 'end_dt' in sleep_df.columns and 'start' in sleep_df.columns:
        sleep_df['start_dt'] = pd.to_datetime(sleep_df['start'], errors='coerce')
        sleep_df['duration_hours'] = (sleep_df['end_dt'] - sleep_df['start_dt']).dt.total_seconds() / 3600.0
    sleep_per_day = sleep_df.groupby('sleep_date', as_index=False)['duration_hours'].sum().rename(columns={'sleep_date':'date','duration_hours':'sleep_hours'})
    sleep_per_day['date'] = pd.to_datetime(sleep_per_day['date']).dt.date

# ---------------------------
# Sidebar: FTP and activity selection
# ---------------------------
st.sidebar.header("Analysis settings")
default_ftp = 250
# Try auto-estimate FTP from max power if available
if df['power_max_w'].notna().any():
    try:
        est_ftp = int(df['power_max_w'].dropna().max() * 0.75)
        default_ftp = est_ftp
    except Exception:
        default_ftp = 250

ftp = st.sidebar.number_input("Enter FTP (watts) — used for zones", min_value=80, max_value=1500, value=default_ftp, step=1)

# Activity selector based on detected sport values
activity_types = sorted(df['sport'].dropna().unique().tolist())
sel_activity = st.sidebar.selectbox("Choose activity type to analyze", activity_types)

# Date range filter
min_date = pd.to_datetime(df['date']).min()
max_date = pd.to_datetime(df['date']).max()
d1, d2 = st.sidebar.date_input("Date range", [min_date.date() if pd.notna(min_date) else None, max_date.date() if pd.notna(max_date) else None])

# ---------------------------
# Filter data
# ---------------------------
df_f = df[df['sport'] == sel_activity].copy()
if d1 and d2:
    df_f = df_f[(pd.to_datetime(df_f['date']).dt.date >= d1) & (pd.to_datetime(df_f['date']).dt.date <= d2)]

if df_f.empty:
    st.error("No activity rows after filtering. Try expanding date range or selecting 'All' activities.")
    st.stop()

# ---------------------------
# Aggregations and charts
# ---------------------------
st.header(f"Performance for: {sel_activity}")

# Top KPIs row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Distance (km)", f"{df_f['distance_km'].sum():.1f}")
k2.metric("Total Rides", f"{len(df_f)}")
k3.metric("Avg Power (W)", f"{df_f['power_w'].dropna().mean():.0f}")
k4.metric("Avg Speed (km/h)", f"{df_f['speed_kmh'].dropna().mean():.1f}")

# Time series: distance, speed, power
st.subheader("Trends over time")
base = alt.Chart(df_f).encode(x=alt.X('date:T', title='Date'))

if df_f['distance_km'].notna().any():
    c1 = base.mark_line(point=True).encode(y=alt.Y('distance_km:Q', title='Distance (km)'))
    st.altair_chart(c1.properties(height=200), use_container_width=True)

if df_f['speed_kmh'].notna().any():
    c2 = base.mark_line(color='orange', point=True).encode(y=alt.Y('speed_kmh:Q', title='Speed (km/h)'))
    st.altair_chart(c2.properties(height=200), use_container_width=True)

if df_f['power_w'].notna().any():
    c3 = base.mark_line(color='red', point=True).encode(y=alt.Y('power_w:Q', title='Power (W)'))
    st.altair_chart(c3.properties(height=200), use_container_width=True)

# Monthly / weekly aggregations
st.subheader("Aggregates")
cols = st.columns(2)
with cols[0]:
    st.write("Monthly summary")
    monthly = df_f.groupby('month').agg({
        'distance_km':'sum',
        'power_w':'mean',
        'speed_kmh':'mean'
    }).fillna(0).reset_index()
    if not monthly.empty:
        st.dataframe(monthly)
        st.bar_chart(monthly.set_index('month')['distance_km'])
with cols[1]:
    st.write("Weekly summary")
    weekly = df_f.groupby('week').agg({
        'distance_km':'sum',
        'power_w':'mean',
        'speed_kmh':'mean'
    }).fillna(0).reset_index()
    if not weekly.empty:
        st.dataframe(weekly)
        st.bar_chart(weekly.set_index('week')['distance_km'])

# Rolling averages
st.subheader("Rolling averages")
if df_f['power_w'].notna().any():
    df_f = df_f.sort_values('date')
    df_f['power_7d'] = df_f['power_w'].rolling(window=7, min_periods=1).mean()
    st.line_chart(df_f.set_index('date')['power_7d'])

# ---------------------------
# Power zones & time-in-zone
# ---------------------------
st.subheader("Power Zones (rides counted in zones)")
zone_counts = compute_time_in_zones(df_f['power_w'].fillna(0), ftp)
zone_df = pd.DataFrame.from_dict(zone_counts, orient='index', columns=['ride_count']).reset_index().rename(columns={'index':'zone'})
st.bar_chart(zone_df.set_index('zone')['ride_count'])

# Per-ride zone classification (just highest zone per ride)
def classify_zone(w, ftp_val):
    # return zone label based on midpoints
    if pd.isna(w):
        return "No power"
    rel = w / ftp_val
    if rel < 0.56:
        return "Z1 Recovery"
    if rel < 0.76:
        return "Z2 Endurance"
    if rel < 0.91:
        return "Z3 Tempo"
    if rel < 1.06:
        return "Z4 Threshold"
    if rel < 1.21:
        return "Z5 VO2"
    return "Z6 Anaerobic"

df_f['zone'] = df_f['power_w'].apply(lambda x: classify_zone(x, ftp))
st.subheader("Per-ride zones (sample)")
st.dataframe(df_f[['date','ride_name','distance_km','power_w','zone']].sort_values('date', ascending=False).head(15))

# ---------------------------
# Sleep vs Performance insights
# ---------------------------
st.subheader("Sleep vs Performance (if sleep data provided)")
if sleep_per_day is not None and not sleep_per_day.empty:
    # merge sleep_per_day with per-ride average power per day
    rides_per_day = df_f.groupby('date_only').agg({'power_w':'mean','distance_km':'sum','speed_kmh':'mean'}).reset_index().rename(columns={'date_only':'date'})
    rides_per_day['date'] = pd.to_datetime(rides_per_day['date']).dt.date
    merged = pd.merge(rides_per_day, sleep_per_day, on='date', how='inner')
    if merged.empty:
        st.info("Sleep uploaded but no overlapping dates between sleep nights and ride dates were found.")
    else:
        st.dataframe(merged.head(20))
        # scatter chart: sleep_hours vs avg power
        chart = alt.Chart(merged).mark_circle(size=60).encode(
            x=alt.X('sleep_hours:Q', title='Sleep (hours)'),
            y=alt.Y('power_w:Q', title='Avg Power (W)'),
            tooltip=['date','sleep_hours','power_w']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
        # simple correlation
        corr = merged['sleep_hours'].corr(merged['power_w'])
        st.write(f"Pearson correlation (sleep_hours vs avg_power): {corr:.3f}")
        # quick insight
        more_sleep = merged[merged['sleep_hours'] >= 7]['power_w'].mean()
        less_sleep = merged[merged['sleep_hours'] < 7]['power_w'].mean()
        pct = (more_sleep - less_sleep) / less_sleep * 100 if pd.notna(more_sleep) and pd.notna(less_sleep) and less_sleep != 0 else np.nan
        st.write(f"Avg power after ≥7h sleep: {more_sleep:.0f} W; after <7h: {less_sleep:.0f} W; delta: {pct:.1f}%")

else:
    st.info("No sleep data provided. Upload Apple Health export (ZIP/XML) or a sleep CSV to use sleep correlation features.")

# ---------------------------
# Smart insights (simple heuristics)
# ---------------------------
st.header("Smart Insights")
insights = []

# 1) Sleep correlation insight (if stable)
if sleep_per_day is not None and not sleep_per_day.empty and 'power_w' in df_f.columns:
    # compute merged as above
    rides_per_day = df_f.groupby('date_only').agg({'power_w':'mean','distance_km':'sum'}).reset_index().rename(columns={'date_only':'date'})
    rides_per_day['date'] = pd.to_datetime(rides_per_day['date']).dt.date
    merged = pd.merge(rides_per_day, sleep_per_day, on='date', how='inner')
    if not merged.empty:
        corr = merged['sleep_hours'].corr(merged['power_w'])
        if pd.notna(corr):
            if corr > 0.15:
                insights.append(f"Positive relationship detected: sleep hours correlate with higher next-day power (r={corr:.2f}).")
            elif corr < -0.15:
                insights.append(f"Negative relationship detected: more sleep correlates with lower measured power (r={corr:.2f}).")
            else:
                insights.append(f"No strong correlation found between sleep and next-day power (r={corr:.2f}).")

# 2) Weekend vs weekday performance
if 'date' in df_f.columns and df_f['power_w'].notna().any():
    df_f['weekday'] = pd.to_datetime(df_f['date']).dt.weekday
    weekend_avg = df_f[df_f['weekday']>=5]['power_w'].mean()
    weekday_avg = df_f[df_f['weekday']<5]['power_w'].mean()
    if pd.notna(weekend_avg) and pd.notna(weekday_avg) and weekday_avg != 0:
        diff = (weekend_avg - weekday_avg) / weekday_avg * 100
        if abs(diff) > 8:
            if diff > 0:
                insights.append(f"You produce ~{diff:.0f}% more power on weekends vs weekdays.")
            else:
                insights.append(f"You produce ~{abs(diff):.0f}% less power on weekends vs weekdays.")
        else:
            insights.append("No major weekend vs weekday power difference detected.")

# 3) Recent trend (7-day rolling power trend)
if 'power_w' in df_f.columns and df_f['power_w'].notna().any():
    df_f_sorted = df_f.sort_values('date').copy()
    df_f_sorted['power_7'] = df_f_sorted['power_w'].rolling(7, min_periods=1).mean()
    if len(df_f_sorted) >= 7:
        recent = df_f_sorted['power_7'].iloc[-1]
        earlier = df_f_sorted['power_7'].iloc[max(0, len(df_f_sorted)-14)]
        if earlier and earlier > 0:
            change_pct = (recent - earlier) / earlier * 100
            if change_pct > 5:
                insights.append(f"7-day rolling power increased by {change_pct:.0f}% compared to two weeks ago — trending up.")
            elif change_pct < -5:
                insights.append(f"7-day rolling power decreased by {abs(change_pct):.0f}% compared to two weeks ago — trending down.")
            else:
                insights.append("7-day rolling power roughly stable compared to two weeks ago.")

# 4) Suggest FTP sanity check
if df_f['power_max_w'].notna().any():
    max_p = df_f['power_max_w'].dropna().max()
    est_ftp = max_p * 0.75
    insights.append(f"Estimated FTP based on observed max power: ~{int(est_ftp)} W (use this to set FTP).")

# 5) Add generic suggestions
if not insights:
    insights.append("No clear automated insights (not enough data). Try uploading more activities or sleep records.")

for i, ins in enumerate(insights, start=1):
    st.markdown(f"**{i}.** {ins}")

# ---------------------------
# Download filtered CSV
# ---------------------------
st.markdown("---")
st.subheader("Download")
st.download_button("Download filtered activity CSV", df_f.to_csv(index=False), f"{sel_activity}_filtered.csv")

st.write("Done — if you want prettier visuals (interactive tooltips, nicer palettes), I can swap to Plotly and add ride detail modals. Which visualization(s) should I upgrade next?")
