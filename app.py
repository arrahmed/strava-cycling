# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from math import isnan

st.set_page_config(page_title="HR × Power × Sleep Dashboard", layout="wide")
st.title("❤️ HR × 🚴 Power × 😴 Sleep — Correlation & Trends")

st.write(
    "Upload two CSVs: (1) cycling per-ride CSV and (2) nightly sleep CSV. "
    "This app computes HR↔Power efficiency, sleep→next-day power correlations, HR trends, and HR-based intensity load."
)

# -------------------------
# Upload boxes (two files)
# -------------------------
c1, c2 = st.columns([2, 1])
with c1:
    cy_file = st.file_uploader("1) Upload cycling CSV (required) — headers: date, moving_hours, distance_km, avg_power, np_power, if, tss", type=["csv"])
with c2:
    sl_file = st.file_uploader("2) Upload sleep CSV (optional) — headers: date, sleep_hours, sleep_quality, resting_hr, hrv", type=["csv"])

if not cy_file:
    st.info("Upload your cycling CSV to start. Use the mock CSV if testing.")
    st.stop()

# -------------------------
# Helper
# -------------------------
def safe_num(series):
    return pd.to_numeric(series, errors="coerce")

def pearson_r(x, y):
    return x.corr(y)

# -------------------------
# Load cycling CSV (we expect one row per ride)
# -------------------------
try:
    dfc = pd.read_csv(cy_file)
except Exception as e:
    st.error(f"Couldn't read cycling CSV: {e}")
    st.stop()

# Normalize column names (strip)
dfc.columns = [c.strip() for c in dfc.columns]

# Map expected columns using exact names you supplied (but also tolerate slight variants)
col_map = {}
def find(cols, candidates):
    for cand in candidates:
        if cand in cols:
            return cand
    # fuzzy lower/strip match
    lowcols = {c.lower().replace(" ", "").replace("_",""): c for c in cols}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_","")
        if key in lowcols:
            return lowcols[key]
    return None

cols = list(dfc.columns)
date_col = find(cols, ["date", "start_date", "start_date_local"])
moving_col = find(cols, ["moving_hours", "moving_time", "movinghours", "moving_time_seconds", "elapsed_time"])
dist_col = find(cols, ["distance_km", "distance", "distance_m", "distance_km"])
avg_power_col = find(cols, ["avg_power", "average_watts", "average_power", "avg_power_w"])
np_col = find(cols, ["np_power", "normalized_power", "weighted_average_watts"])
if_col = find(cols, ["if", "intensity_factor"])
tss_col = find(cols, ["tss", "training_stress_score"])
avg_hr_col = find(cols, ["average_heartrate", "avg_heartrate", "avg_hr", "average_hr"])
max_hr_col = find(cols, ["max_heartrate", "max_hr", "peak_heartrate"])
name_col = find(cols, ["name", "ride_name", "activity_name"])

# Inform detected mapping
st.subheader("Detected cycling columns")
st.write({
    "date_col": date_col,
    "moving_col": moving_col,
    "distance_col": dist_col,
    "avg_power_col": avg_power_col,
    "np_col": np_col,
    "if_col": if_col,
    "tss_col": tss_col,
    "avg_hr_col": avg_hr_col,
    "max_hr_col": max_hr_col
})

# Parse date
if date_col:
    dfc['date'] = pd.to_datetime(dfc[date_col], errors='coerce').dt.date
else:
    st.error("No date column detected in cycling CSV. Ensure there's a 'date' column.")
    st.stop()

# canonical columns
dfc['distance_km'] = safe_num(dfc[dist_col]) if dist_col else np.nan
# moving_hours: if value seems >24 assume seconds and convert
if moving_col:
    mv = safe_num(dfc[moving_col])
    if mv.max(skipna=True) is not None and mv.max(skipna=True) > 24:  # seconds likely
        dfc['moving_hours'] = mv / 3600.0
    else:
        dfc['moving_hours'] = mv
else:
    dfc['moving_hours'] = np.nan

dfc['avg_power_w'] = safe_num(dfc[avg_power_col]) if avg_power_col else np.nan
dfc['np_power_w'] = safe_num(dfc[np_col]) if np_col else np.nan
dfc['if'] = safe_num(dfc[if_col]) if if_col else np.nan
dfc['tss'] = safe_num(dfc[tss_col]) if tss_col else np.nan
dfc['avg_hr'] = safe_num(dfc[avg_hr_col]) if avg_hr_col else np.nan
dfc['max_hr'] = safe_num(dfc[max_hr_col]) if max_hr_col else np.nan
dfc['ride_name'] = dfc[name_col].astype(str) if name_col else ("ride_" + dfc.index.astype(str))

# Aggregate to per-day (mean for power/hr, sum for distance/time/tss)
daily = dfc.groupby('date').agg({
    'distance_km':'sum',
    'moving_hours':'sum',
    'avg_power_w':'mean',
    'np_power_w':'mean',
    'if':'mean',
    'tss':'sum',
    'avg_hr':'mean',
    'max_hr':'max'
}).reset_index().sort_values('date')

# show sample
st.subheader("Per-day aggregated cycling (sample)")
st.dataframe(daily.head(10))

# -------------------------
# Load sleep CSV (if provided)
# -------------------------
sleep_df = None
if sl_file:
    try:
        sleep_raw = pd.read_csv(sl_file)
        sleep_raw.columns = [c.strip() for c in sleep_raw.columns]
        # expect header: date,sleep_hours,sleep_quality,resting_hr,hrv
        sdate_col = find(list(sleep_raw.columns), ["date", "sleep_date", "end_date"])
        sh_col = find(list(sleep_raw.columns), ["sleep_hours", "duration_hours", "hours", "sleepduration"])
        sq_col = find(list(sleep_raw.columns), ["sleep_quality", "quality"])
        srhr_col = find(list(sleep_raw.columns), ["resting_hr", "restingheartrate"])
        shrv_col = find(list(sleep_raw.columns), ["hrv"])
        # normalize
        if sdate_col:
            sleep_raw['date'] = pd.to_datetime(sleep_raw[sdate_col], errors='coerce').dt.date
        else:
            sleep_raw['date'] = pd.NaT
        sleep_raw['sleep_hours'] = safe_num(sleep_raw[sh_col]) if sh_col else np.nan
        sleep_raw['sleep_quality'] = safe_num(sleep_raw[sq_col]) if sq_col else np.nan
        sleep_raw['resting_hr'] = safe_num(sleep_raw[srhr_col]) if srhr_col else np.nan
        sleep_raw['hrv'] = safe_num(sleep_raw[shrv_col]) if shrv_col else np.nan
        # aggregate per-night
        sleep_df = sleep_raw.groupby('date').agg({'sleep_hours':'sum','sleep_quality':'mean','resting_hr':'mean','hrv':'mean'}).reset_index()
        st.subheader("Sleep (per-night)")
        st.dataframe(sleep_df.head(10))
    except Exception as e:
        st.warning(f"Couldn't read/parse sleep CSV: {e}")
        sleep_df = None
else:
    st.info("No sleep file uploaded — sleep-based correlations will be disabled until a file is uploaded.")

# -------------------------
# Align sleep -> next-day rides
# -------------------------
# Convention: sleep on date D maps to rides on date D+1
if sleep_df is not None and not sleep_df.empty:
    sleep_df['next_date'] = pd.to_datetime(sleep_df['date']) + pd.Timedelta(days=1)
    sleep_df['next_date'] = sleep_df['next_date'].dt.date
    merged = daily.merge(sleep_df[['next_date','sleep_hours','sleep_quality','resting_hr','hrv']], left_on='date', right_on='next_date', how='left')
else:
    merged = daily.copy()
    merged['sleep_hours'] = np.nan
    merged['sleep_quality'] = np.nan
    merged['resting_hr'] = np.nan
    merged['hrv'] = np.nan

# -------------------------
# Sidebar: max_hr input (for intensity calc) and FTP not required here
# -------------------------
st.sidebar.header("HR & intensity settings")
user_max_hr = st.sidebar.number_input("Estimated Max HR (bpm) — used for HR zone/intensity", min_value=120, max_value=250, value=int(np.nanmean(merged['max_hr'].dropna()) if merged['max_hr'].notna().any() else 190))
st.sidebar.caption("If unknown, use 220 - age as a rough estimate.")

# -------------------------
# HR → Power efficiency
# -------------------------
st.header("HR → Power Efficiency")

if merged['avg_hr'].notna().any() and merged['avg_power_w'].notna().any():
    merged['watts_per_bpm'] = merged['avg_power_w'] / merged['avg_hr']
    # show recent trend + distribution
    st.subheader("Watts per bpm (avg_power / avg_hr) — higher = more efficient")
    chart = alt.Chart(merged).mark_line(point=True).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('watts_per_bpm:Q', title='Watts per bpm'),
        tooltip=['date','avg_power_w','avg_hr','watts_per_bpm']
    ).properties(height=300).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.write("Higher watts per bpm = more power produced for each heart-beat on average (good).")
    st.metric("Most recent watts_per_bpm", f"{merged['watts_per_bpm'].iloc[-1]:.2f}" if not isnan(merged['watts_per_bpm'].iloc[-1]) else "N/A")
else:
    st.info("Not enough avg_hr or avg_power data to compute watts_per_bpm.")

# -------------------------
# Sleep -> Power correlations (if sleep provided)
# -------------------------
st.header("Sleep → Next-day Performance Correlations")

targets = []
if merged['avg_power_w'].notna().any():
    targets.append(('avg_power_w','Average Power (W)'))
if merged['np_power_w'].notna().any():
    targets.append(('np_power_w','Normalized Power (W)'))
if merged['if'].notna().any():
    targets.append(('if','Intensity Factor'))
if merged['tss'].notna().any():
    targets.append(('tss','TSS'))

if sleep_df is None:
    st.info("Upload sleep CSV to get sleep→power correlations.")
else:
    if len(targets) == 0:
        st.info("No power/NP/IF/TSS columns found to correlate with sleep.")
    else:
        for key, pretty in targets:
            sub = merged[[key,'sleep_hours','date']].dropna()
            if sub.shape[0] < 5:
                st.write(f"Not enough overlapping days for **{pretty}** (need ≥5). Found {sub.shape[0]}.")
                continue

            corr = pearson_r(sub[key], sub['sleep_hours'])
            st.subheader(f"{pretty} ← Sleep Hours (night before)")
            st.write(f"Pearson r = **{corr:.3f}**   (n = {len(sub)})")

            # regression line
            X = sub[['sleep_hours']].values.reshape(-1,1)
            y = sub[key].values
            model = LinearRegression().fit(X,y)
            xs = np.linspace(sub['sleep_hours'].min(), sub['sleep_hours'].max(), 50)
            ys = model.predict(xs.reshape(-1,1))

            plotdf = pd.DataFrame({
                'sleep_hours': sub['sleep_hours'],
                key: sub[key],
                'date': pd.to_datetime(sub['date'])
            })

            pts = alt.Chart(plotdf).mark_circle(size=80).encode(
                x=alt.X('sleep_hours:Q', title='Sleep hours (night before)'),
                y=alt.Y(f'{key}:Q', title=pretty),
                tooltip=['date','sleep_hours', key]
            )

            line = alt.Chart(pd.DataFrame({'sleep_hours': xs, key: ys})).mark_line(color='firebrick', strokeWidth=2).encode(
                x='sleep_hours:Q', y=f'{key}:Q'
            )

            st.altair_chart((pts + line).interactive().properties(height=320), use_container_width=True)

            slope = model.coef_[0]
            intercept = model.intercept_
            st.write(f"Model: {pretty} ≈ {slope:.2f} × sleep_hours + {intercept:.1f}")
            if corr > 0.15:
                st.success(f"Insight: More sleep tends to be associated with higher {pretty} (r={corr:.2f}).")
            elif corr < -0.15:
                st.warning(f"Insight: More sleep tends to be associated with lower {pretty} (r={corr:.2f}).")
            else:
                st.info(f"No strong linear relationship detected (r={corr:.2f}).")

# -------------------------
# HR trend charts
# -------------------------
st.header("HR Trends")

if merged['avg_hr'].notna().any():
    st.subheader("Average HR over time")
    ch = alt.Chart(merged).mark_line(point=True).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('avg_hr:Q', title='Average HR (bpm)'),
        tooltip=['date','avg_hr']
    ).properties(height=300).interactive()
    st.altair_chart(ch, use_container_width=True)

if merged['max_hr'].notna().any():
    st.subheader("Max HR over time")
    ch2 = alt.Chart(merged).mark_line(point=True, color='orange').encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('max_hr:Q', title='Max HR (bpm)'),
        tooltip=['date','max_hr']
    ).properties(height=250).interactive()
    st.altair_chart(ch2, use_container_width=True)

# monthly HR summary
st.subheader("Monthly avg HR summary")
if 'month' not in merged.columns:
    merged['month'] = pd.to_datetime(merged['date']).astype('datetime64[ns]').astype('datetime64[M]')
monthly_hr = merged.groupby('month').agg({'avg_hr':'mean','max_hr':'mean'}).reset_index()
if not monthly_hr.empty:
    mchart = alt.Chart(monthly_hr).transform_fold(
        ['avg_hr','max_hr'],
        as_ = ['type','value']
    ).mark_line(point=True).encode(
        x='month:T', y='value:Q', color='type:N', tooltip=['month','type','value']
    ).properties(height=300).interactive()
    st.altair_chart(mchart, use_container_width=True)
else:
    st.info("Not enough HR data to produce monthly summary.")

# -------------------------
# HR-based intensity load
# (simple, interpretable formula)
# intensity_pct = avg_hr / user_max_hr
# hr_load = intensity_pct * moving_hours * 100  (arbitrary scaling like HR-TSS)
# -------------------------
st.header("HR-based Intensity Load (per day)")

if merged['avg_hr'].notna().any():
    merged['intensity_pct'] = merged['avg_hr'] / float(user_max_hr)
    merged['hr_load'] = merged['intensity_pct'] * merged['moving_hours'] * 100.0  # units: "HR-load"
    st.subheader("HR-load over time")
    ch_load = alt.Chart(merged).mark_bar().encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('hr_load:Q', title='HR Load'),
        tooltip=['date','hr_load','avg_hr','moving_hours']
    ).properties(height=300).interactive()
    st.altair_chart(ch_load, use_container_width=True)
    st.metric("Total HR-load", f"{merged['hr_load'].sum():.0f}")
else:
    st.info("Not enough HR data to compute HR-based intensity load.")

# -------------------------
# Downloads & wrap up
# -------------------------
st.markdown("---")
st.subheader("Download merged per-day dataset")
csv_out = merged.to_csv(index=False)
st.download_button("⬇️ Download merged CSV", data=csv_out, file_name="merged_sleep_hr_power.csv")

st.success("HR features added. Want me to (A) add per-ride tooltips, (B) compute CTL/ATL/TSB fitness curves, or (C) export a one-page PDF report? Reply with A/B/C.")
