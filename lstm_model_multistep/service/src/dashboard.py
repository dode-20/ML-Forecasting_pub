import types
import torch
import sys

# WORKAROUND: Remove torch.classes completely from the imported namespace
if hasattr(torch, "classes"):
    delattr(torch, "classes")
    sys.modules.pop("torch.classes", None)

import streamlit as st
import pandas as pd
import os
from model import PVForecastLSTM
from datetime import datetime, timedelta, time
import numpy as np
import plotly.express as px

# Define the data path
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/csv_files_2023_private"))

# Load historical daily data
df = pd.concat([
    pd.read_csv(
        f"{DATA_PATH}/{f}",
        sep=";", 
        decimal=".",  # Specify the decimal separator
        thousands=",",  # Specify the thousands separator
    ).assign(filename=f)
    for f in sorted(os.listdir(DATA_PATH)) if f.endswith(".csv")
], ignore_index=True)

# Rename and clean up columns
df.columns = [col.strip() for col in df.columns]
df.rename(columns={df.columns[0]: "Time"}, inplace=True)
df["timestamp"] = pd.to_datetime(
    df["filename"].str.extract(r'(\d{4}_\d{2}_\d{2})')[0] + " " +
    df["Time"].str.extract(r'(\d{2}:\d{2})')[0],
    format="%Y_%m_%d %H:%M", errors="coerce"
)
df["pv_power_w"] = pd.to_numeric(df["PV power generation / Mean values [W]"], errors="coerce")
df["pv_kwh"] = df["pv_power_w"] / 1000.0
df.dropna(subset=["timestamp", "pv_power_w"], inplace=True)

# Calculate pv_prev_year
df["prev_year"] = df["timestamp"] - pd.DateOffset(years=1)
df["pv_prev_year"] = df.set_index("timestamp")["pv_kwh"].reindex(df["prev_year"]).values
df["pv_prev_year"] = pd.to_numeric(df["pv_prev_year"], errors="coerce").fillna(0.0)

# Load the model
model = PVForecastLSTM()
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pv_model.pth")
state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()

# Streamlit dashboard
st.title("PV-Forecast Dashboard")

# Input for selecting a start date
start_date = st.date_input("Start date for the forecast", value=datetime.today().date())

# Initialize predictions and actual values
timestamps_all = []
predictions_all = []
real_values_all = {f"Actual Value {year}": [] for year in range(start_date.year - 5, start_date.year + 1)}

# Input data for the last 3 days
past_dates = [start_date - timedelta(days=i) for i in range(1, 4)]
day_data = df[df["timestamp"].dt.date.isin(past_dates)]

if len(day_data) == 96 * 3:  # Ensure 3 days of data are available
    day_data = day_data.copy()
    day_data["hour"] = day_data["timestamp"].dt.hour + day_data["timestamp"].dt.minute / 60.0
    day_data["day_of_year"] = day_data["timestamp"].dt.dayofyear
    day_data["weekday"] = day_data["timestamp"].dt.weekday

    input_day = day_data[["pv_kwh", "pv_prev_year", "hour", "day_of_year", "weekday"]].values.reshape(1, 96 * 3, 5)

    # Generate forecast
    input_tensor = torch.tensor(input_day, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(input_tensor).numpy().flatten().tolist()

    # Ensure start_date is a datetime object at midnight
    start_datetime = datetime.combine(start_date, time.min)

    # Generate timestamps for the next 7 days with 15-minute intervals
    timestamps = [
        start_datetime + timedelta(days=day_offset, minutes=m)
        for day_offset in range(7)
        for m in range(0, 24 * 60, 15)
    ]

    # Collect actual values for the last 5 years and the year of the selected date
    for year_offset in range(0, 6):  # Last 5 years + year of the selected date
        if year_offset == 0:
            year = start_date.year  # Year of the selected date
            label = f"Actual Value {year}"
        else:
            year = start_date.year - year_offset
            label = f"Actual Value {year}"

        real_values = []
        for day_offset in range(7):
            past_date = start_date + timedelta(days=day_offset - 365 * year_offset)
            day_data = df[df["timestamp"].dt.date == past_date]
            if len(day_data) == 96:
                real_values.extend(day_data["pv_kwh"].values.tolist())
            else:
                real_values.extend([None] * 96)
        real_values_all[label] = real_values

    # Remove years without available actual values (all values are None)
    real_values_all = {year: values for year, values in real_values_all.items() if any(v is not None for v in values)}

    # Ensure all lists have the same length
    assert len(timestamps) == len(y_pred), "List lengths do not match!"

    # Create a DataFrame
    df_pred = pd.DataFrame({
        "Time": timestamps,
        "Forecast [kWh]": y_pred,
    })

    # Add actual values
    for year, values in real_values_all.items():
        df_pred[year] = values

    # Convert all columns to numeric data types
    for col in df_pred.columns:
        if col != "Time":  # "Time" remains as a datetime column
            df_pred[col] = pd.to_numeric(df_pred[col], errors="coerce")

    df_pred["Time"] = pd.to_datetime(df_pred["Time"], errors="coerce")
    df_pred = df_pred.dropna(subset=["Time"])
    df_pred = df_pred.sort_values("Time")
    df_pred["Forecast [kWh]"] = pd.to_numeric(df_pred["Forecast [kWh]"], errors="coerce")

    # Visualization
    fig = px.line(
        df_pred,
        x="Time",
        y=["Forecast [kWh]"] + list(real_values_all.keys()),
        title=f"PV Forecast for the next 7 days (including actual values for the last 5 years and {start_date.year})"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Optional display of forecast data
    if st.checkbox("Show Forecast Data"):
        st.dataframe(df_pred)
else:
    st.error("Not enough data available for the last 3 days.")