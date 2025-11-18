# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Sleep ⇄ Cycling Correlation", layout="wide")
st.title("😴⇄🚴 Sleep vs Cycling Performance (Upload & Correlate)")

st.write(
    "Upload your cycling CSV (activities) and a sleep CSV (Apple Health / simple CSV). "
    "The app will aggregate rides per day, align the previous night's sleep to next-day rides, "
    "and compute correlations + plots (sleep_hours → next-day power / NP / IF / TSS)."
)

# -------------------------
# Utility: fuzzy column finder
# -------------------------
def find_col(df, candidates):
    cols = list(df.columns)
    low = [c.lower().replace(" ", "").replace("_", "") for c in cols]
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        for i, lc in enumerate(low):
            if key in lc or lc in key:
                return cols[i]
    return None

def safe_to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

# -------------------------
# Uploads
# -------------------------
c1, c2 = st.columns(2)
with c1:
    cy_file = st.file_uploader("Upload cycling CSV (Strava-like) — required", type=["csv"])
with c2:
    sl_file = st.file_uploader("Upload sleep CSV (Apple Health CSV or simple sleep.csv) — optional", type=["csv", "zip", "xml"])

if not cy_file:
    st.info("Upload a cycling CSV to start.")
    st.stop()

# -------------------------
# Load cycling CSV
# -------------------------
try:
    dfc = pd.read_csv(cy_file)
except Exception as e:
    st.error(f"Couldn't read cycling CSV: {e}")
    st.stop()

# Normalize header whitespace
dfc.columns = [c.strip() for c in dfc.columns]

# --- find likely columns (use your mocked column names as likely candidates) ---
date_col = find_col(dfc, ["date", "start_date", "start_date_local", "activity_date", "timestamp"])
# headers you said we have: date,moving_hours,distance_km,avg_power,np_power,if,tss
moving_col = find_col(dfc, ["moving_hours", "moving_time", "movingtime", "elapsed_time"])
dist_col = find_col(dfc, ["distance_km", "distance", "distance_m", "distance_meters"])
avg_power_col = find_col(dfc, ["avg_power", "average_watts", "average_power", "power_w", "avgwatts"])
np_col = find_col(dfc, ["np_power", "normalized_power", "np"])
if_col = find_col(dfc, ["if", "intensity_factor"])
tss_col = find_col(dfc, ["tss", "training_stress_score"])
name_col = find_col(dfc, ["name", "activity_name", "ride_name"])

# Fallback conversions for units/column shapes
# If distance is in meters (typical), convert to km
if dist_col:
    try:
        test_max = pd.to_numeric(dfc[dist_col], errors="coerce").max()
        if pd.notna(test_max) and test_max > 1000:
            dfc["distance_km"] = pd.to_numeric(dfc[dist_col], errors="coerce") / 1000.0
        else:
            dfc["distance_km"] = pd.to_numeric(dfc[dist_col], errors="coerce")
    except Exception:
        dfc["distance_km"] = pd.to_numeric(dfc[dist_col], errors="coerce")
else:
    dfc["distance_km"] = np.nan

# moving hours: if moving_hours column exists (hours), use; else convert seconds->hours
if moving_col:
    # some exports give moving_time in seconds, some in hours; detect typical scale
    mv = pd.to_numeric(dfc[moving_col], errors="coerce")
    if mv.max() > 24:  # likely in seconds
        dfc["moving_hours"] = mv / 3600.0
    else:
        dfc["moving_hours"] = mv
else:
    dfc["moving_hours"] = np.nan

# average power
if avg_power_col:
    dfc["avg_power_w"] = pd.to_numeric(dfc[avg_power_col], errors="coerce")
else:
    dfc["avg_power_w"] = np.nan

# NP
if np_col:
    dfc["np_power_w"] = pd.to_numeric(dfc[np_col], errors="coerce")
else:
    dfc["np_power_w"] = np.nan

# IF
if if_col:
    dfc["if"] = pd.to_numeric(dfc[if_col], errors="coerce")
else:
    dfc["if"] = np.nan

# TSS
if tss_col:
    dfc["tss"] = pd.to_numeric(dfc[tss_col], errors="coerce")
else:
    dfc["tss"] = np.nan

# parse date
if date_col:
    dfc["date"] = pd.to_datetime(dfc[date_col], errors="coerce").dt.date
else:
    # try best-effort: if there's a column named 'date' in simple lowercase
    if 'date' in dfc.columns:
        dfc["date"] = pd.to_datetime(dfc['date'], errors="coerce").dt.date
    else:
        st.warning("No obvious date column found — please ensure your cycling CSV contains a date column.")
        dfc["date"] = pd.NaT

# ride name fallback
if name_col:
    dfc["ride_name"] = dfc[name_col].astype(str)
else:
    dfc["ride_name"] = "ride_" + dfc.index.astype(str)

st.subheader("Cycling data (aggregating per-day)")
st.write("Detected columns and examples:")
st.write({
    "date_col": date_col,
    "moving_col": moving_col,
    "distance_km_col": dist_col,
    "avg_power_col": avg_power_col,
    "np_col": np_col,
    "if_col": if_col,
    "tss_col": tss_col
})

# -------------------------
# Aggregate cycling to per-day (mean/ sum metrics)
# -------------------------
agg_funcs = {
    "distance_km": "sum",
    "moving_hours": "sum",
    "avg_power_w": "mean",
    "np_power_w": "mean",
    "if": "mean",
    "tss": "sum"
}
# ensure columns exist
for k in list(agg_funcs.keys()):
    if k not in dfc.columns:
        dfc[k] = np.nan

daily = dfc.groupby("date").agg(agg_funcs).reset_index().sort_values("date")
daily["rides_count"] = dfc.groupby("date").size().reindex(daily["date"]).values

st.dataframe(daily.head(10))

# -------------------------
# Load sleep CSV (simple CSV expected) and normalize
# -------------------------
sleep_df = None
if sl_file:
    try:
        sleep_raw = pd.read_csv(sl_file)
        sleep_raw.columns = [c.strip() for c in sleep_raw.columns]
        # find sleep date & duration columns
        sleep_date_col = find_col(sleep_raw, ["date", "sleep_date", "end_date", "end"])
        sleep_dur_col = find_col(sleep_raw, ["sleep_hours", "duration_hours", "duration", "hours", "sleepduration"])
        sleep_quality_col = find_col(sleep_raw, ["sleep_quality", "quality"])
        hr_col_sleep = find_col(sleep_raw, ["resting_hr", "restingheart", "restingheartrate"])
        hrv_col_sleep = find_col(sleep_raw, ["hrv"])
        # create normalized dataframe
        s = sleep_raw.copy()
        if sleep_date_col:
            s["sleep_date"] = pd.to_datetime(s[sleep_date_col], errors="coerce").dt.date
        else:
            # if start/end columns exist, try to compute night end
            start_col = find_col(sleep_raw, ["start", "start_date", "starttime"])
            end_col = find_col(sleep_raw, ["end", "end_date", "endtime"])
            if start_col and end_col:
                s["sleep_date"] = pd.to_datetime(s[end_col], errors="coerce").dt.date
            else:
                s["sleep_date"] = pd.NaT
        if sleep_dur_col:
            s["sleep_hours"] = pd.to_numeric(s[sleep_dur_col], errors="coerce")
        else:
            # try to compute from start/end
            start_col = find_col(sleep_raw, ["start", "start_date", "starttime"])
            end_col = find_col(sleep_raw, ["end", "end_date", "endtime"])
            if start_col and end_col:
                s["sleep_hours"] = (pd.to_datetime(s[end_col], errors="coerce") - pd.to_datetime(s[start_col], errors="coerce")).dt.total_seconds() / 3600.0
            else:
                s["sleep_hours"] = np.nan
        if sleep_quality_col:
            s["sleep_quality"] = pd.to_numeric(s[sleep_quality_col], errors="coerce")
        else:
            s["sleep_quality"] = np.nan
        if hr_col_sleep:
            s["resting_hr"] = pd.to_numeric(s[hr_col_sleep], errors="coerce")
        else:
            s["resting_hr"] = np.nan
        if hrv_col_sleep:
            s["hrv"] = pd.to_numeric(s[hrv_col_sleep], errors="coerce")
        else:
            s["hrv"] = np.nan

        # aggregate per-night in case there are multiple records
        sleep_df = s.groupby("sleep_date").agg({"sleep_hours":"sum","sleep_quality":"mean","resting_hr":"mean","hrv":"mean"}).reset_index().rename(columns={"sleep_date":"date"})
        sleep_df["date"] = pd.to_datetime(sleep_df["date"]).dt.date
        st.subheader("Sleep (normalized per night)")
        st.dataframe(sleep_df.head())
    except Exception as e:
        st.warning(f"Couldn't parse sleep CSV: {e}")
        sleep_df = None

# -------------------------
# Align sleep -> next-day rides
# -------------------------
# We treat the sleep record for date D as the sleep that *precedes* rides on date D+1
# So shift sleep date forward by +1 day to match rides
if sleep_df is not None and not sleep_df.empty:
    sleep_df["next_date"] = pd.to_datetime(sleep_df["date"]) + pd.Timedelta(days=1)
    sleep_df["next_date"] = sleep_df["next_date"].dt.date
    merged = pd.merge(daily, sleep_df, left_on="date", right_on="next_date", how="left", suffixes=("","_sleep"))
else:
    merged = daily.copy()
    merged["sleep_hours"] = np.nan
    merged["sleep_quality"] = np.nan
    merged["resting_hr"] = np.nan
    merged["hrv"] = np.nan

# -------------------------
# Compute correlations and show charts
# -------------------------
st.header("Correlation: Sleep (night) → Next-day performance")

# metrics to test
targets = []
if merged["avg_power_w"].notna().any():
    targets.append(("avg_power_w", "Average Power (W)"))
if merged["np_power_w"].notna().any():
    targets.append(("np_power_w", "Normalized Power (W)"))
if merged["if"].notna().any():
    targets.append(("if", "Intensity Factor"))
if merged["tss"].notna().any():
    targets.append(("tss", "Training Stress Score (TSS)"))

if sleep_df is None:
    st.info("No sleep uploaded — upload a sleep CSV to compute correlations.")
else:
    if len(targets) == 0:
        st.info("No suitable power/load columns found in cycling data (avg_power/np/if/tss). Correlation can't be computed.")
    else:
        for col_key, pretty in targets:
            # drop rows without both
            sub = merged[[col_key, "sleep_hours", "date"]].dropna()
            if sub.shape[0] < 5:
                st.write(f"Not enough overlapping days for **{pretty}** (need >=5).")
                continue

            # compute Pearson correlation
            corr = sub[col_key].corr(sub["sleep_hours"])
            st.subheader(f"{pretty} ← Sleep Hours (next-day)")
            st.write(f"Pearson r = **{corr:.3f}**  (n={len(sub)})")

            # scatter + regression line via sklearn
            X = sub[["sleep_hours"]].values.reshape(-1, 1)
            y = sub[col_key].values
            model = LinearRegression().fit(X, y)
            # line points
            xs = np.linspace(min(X).item(), max(X).item(), 50)
            ys = model.predict(xs.reshape(-1, 1))

            chart_df = pd.DataFrame({
                "sleep_hours": sub["sleep_hours"],
                col_key: sub[col_key],
                "date": pd.to_datetime(sub["date"])
            })

            base = alt.Chart(chart_df).encode(
                x=alt.X("sleep_hours:Q", title="Sleep hours (night before)"),
                y=alt.Y(f"{col_key}:Q", title=pretty),
                tooltip=["date", "sleep_hours", col_key]
            )

            pts = base.mark_circle(size=80).encode(color=alt.value("#1f77b4"))
            line = alt.Chart(pd.DataFrame({"sleep_hours": xs, col_key: ys})).mark_line(color="firebrick", strokeWidth=2)
            reg = line.encode(x="sleep_hours:Q", y=f"{col_key}:Q")

            combo = (pts + reg).interactive().properties(height=320, width=700)
            st.altair_chart(combo, use_container_width=True)

            # quick textual insight
            slope = model.coef_[0]
            intercept = model.intercept_
            st.write(f"Model: {pretty} ≈ {slope:.2f} × sleep_hours + {intercept:.1f}")
            # human summary
            if corr > 0.15:
                st.success(f"Insight: More sleep tends to be associated with higher {pretty} (r={corr:.2f}).")
            elif corr < -0.15:
                st.warning(f"Insight: More sleep tends to be associated with lower {pretty} (r={corr:.2f}).")
            else:
                st.info(f"No strong linear relationship detected (r={corr:.2f}).")

# -------------------------
# Additional insights
# -------------------------
st.header("Quick Automated Insights")
insights = []
# Sleep variance vs power variance
if sleep_df is not None and 'sleep_hours' in merged.columns and merged['sleep_hours'].notna().sum() >= 8:
    # correlation with avg power if available
    if 'avg_power_w' in merged.columns and merged['avg_power_w'].notna().sum() >= 8:
        r = merged['sleep_hours'].corr(merged['avg_power_w'])
        if pd.notna(r):
            if r > 0.12:
                insights.append(f"Sleep hours positively correlate with next-day average power (r={r:.2f}).")
            elif r < -0.12:
                insights.append(f"Sleep hours negatively correlate with next-day average power (r={r:.2f}).")
            else:
                insights.append("No notable correlation between sleep hours and next-day average power.")

# resting HR & HRV suggestions
if 'resting_hr' in merged.columns and merged['resting_hr'].notna().any():
    mean_rest = merged['resting_hr'].mean()
    insights.append(f"Avg resting HR on uploaded nights: {mean_rest:.1f} bpm — unusually high nights may indicate fatigue.")

if 'hrv' in merged.columns and merged['hrv'].notna().any():
    mean_hrv = merged['hrv'].mean()
    insights.append(f"Avg HRV on uploaded nights: {mean_hrv:.1f} ms — higher is typically better recovery-wise.")

if len(insights) == 0:
    st.write("Not enough data to produce automated insights. Upload more nights or rides.")
else:
    for i, it in enumerate(insights, 1):
        st.markdown(f"**{i}.** {it}")

# -------------------------
# Download merged table
# -------------------------
st.markdown("---")
st.subheader("Download merged per-day dataset (sleep aligned to next-day rides)")
csv_out = merged.to_csv(index=False)
st.download_button("⬇️ Download merged CSV", data=csv_out, file_name="merged_sleep_cycling.csv")

st.info("Done — show this to the boss. If you want, I can (A) add per-ride detail tooltips, (B) compute CTL/ATL/TSB fitness curves, (C) produce a PDF report. Tell me which next.")
