import streamlit as st
import pandas as pd
import zipfile
import io
import xml.etree.ElementTree as ET

st.title("🚴 Cycling & Sleep Performance Dashboard")

st.write("Upload your Strava activities export, and optionally your Apple Health data (ZIP, XML, or CSV).")

# ----------------------------
# STRAVA FILE UPLOAD
# ----------------------------
strava_file = st.file_uploader(
    "Upload Strava export (.csv)",
    type=["csv"],
)

# ----------------------------
# SLEEP FILE UPLOAD
# ----------------------------
sleep_file = st.file_uploader(
    "Upload Apple Health data (ZIP/XML/CSV) – optional",
    type=["zip", "xml", "csv"],
)

# Stop until Strava is uploaded
if not strava_file:
    st.info("👆 Please upload your Strava CSV file to continue.")
    st.stop()

# Load Strava CSV
df = pd.read_csv(strava_file)
df.columns = [c.lower().replace(" ", "_") for c in df.columns]

st.subheader("Preview of Strava data")
st.dataframe(df.head())

# Detect power
has_power = "average_watts" in df.columns
has_estimated = "estimated_watts" in df.columns

if has_power:
    st.success("Detected **real power meter** data.")
elif has_estimated:
    st.warning("Using **estimated power** (no power meter detected).")
else:
    st.error("No power data found in Strava file.")

# ----------------------------
# LOAD SLEEP DATA
# ----------------------------
sleep_df = None

if sleep_file:
    ext = sleep_file.name.lower()

    # CASE 1: Apple Health ZIP export
    if ext.endswith(".zip"):
        with zipfile.ZipFile(sleep_file, "r") as z:
            xml_files = [f for f in z.namelist() if f.endswith(".xml")]
            if xml_files:
                xml_content = z.read(xml_files[0])
                root = ET.fromstring(xml_content)

                # Extract "SleepAnalysis" records
                records = []
                for record in root.findall(".//Record"):
                    if record.attrib.get("type") == "HKCategoryTypeIdentifierSleepAnalysis":
                        records.append({
                            "start": record.attrib.get("startDate"),
                            "end": record.attrib.get("endDate"),
                            "value": record.attrib.get("value")
                        })

                sleep_df = pd.DataFrame(records)
                st.success("Extracted sleep data automatically from Apple Health ZIP!")

    # CASE 2: XML file directly
    elif ext.endswith(".xml"):
        xml_content = sleep_file.read()
        root = ET.fromstring(xml_content)
        records = []
        for record in root.findall(".//Record"):
            if record.attrib.get("type") == "HKCategoryTypeIdentifierSleepAnalysis":
                records.append({
                    "start": record.attrib.get("startDate"),
                    "end": record.attrib.get("endDate"),
                    "value": record.attrib.get("value")
                })
        sleep_df = pd.DataFrame(records)
        st.success("Extracted sleep data from XML!")

    # CASE 3: CSV sleep file
    elif ext.endswith(".csv"):
        sleep_df = pd.read_csv(sleep_file)
        st.success("Loaded sleep CSV!")

# Show sleep preview
if sleep_df is not None:
    st.subheader("Preview of Sleep Data")
    st.dataframe(sleep_df.head())
else:
    st.info("No sleep data uploaded — sleep charts will be disabled.")
