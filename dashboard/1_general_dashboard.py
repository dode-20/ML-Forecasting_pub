import streamlit as st
import requests
import time
import math
from pathlib import Path

def polar_to_cartesian(cx, cy, radius, angle_deg):
    angle_rad = math.radians(angle_deg)
    x = cx + radius * math.cos(angle_rad)
    y = cy - radius * math.sin(angle_rad)
    return x, y

def semicircle_path(value_percent):
    start_x, start_y = 10, 100
    angle = 180 * (value_percent / 100)
    end_x, end_y = polar_to_cartesian(90, 100, 80, 180 - angle)
    large_arc = 1 if value_percent > 50 else 0
    return f"M10,100 A80,80 0 {large_arc},1 {end_x:.1f},{end_y:.1f}"

def led_indicator(status: str):
    color = {
        "running": "#28c76f",    # green
        "stopped": "#ea5455",    # red
        "unreachable": "#b6b6b6" # gray
    }.get(status, "#b6b6b6")

    html = f"""
    <div style='
        display: inline-block;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background-color: {color};
        margin-right: 10px;
        box-shadow: 0 0 6px {color};
    '></div><span style='font-weight: 500;'>{status}</span>
    """
    st.markdown(html, unsafe_allow_html=True)

st.set_page_config(page_title="PV Forecasting", layout="wide")
st.title("IPV - PV Forecast Dashboard")

# System Check Section

# System Check Section
st.subheader("System Check")

# --- System Resource Monitor ---
import psutil
import pandas as pd

# --- Live CPU and RAM monitor with Plotly ---
import plotly.graph_objects as go

st.markdown("CPU and RAM usage: (Dashboard container - later overall system usage)")

cpu_percent = psutil.cpu_percent()
ram_percent = psutil.virtual_memory().percent

col1, col2 = st.columns(2)

# Gauge tick marks for semicircle, every 20%
tick_marks = ""
for i in range(0, 101, 20):
    angle = 180 * (i / 100)
    x1, y1 = polar_to_cartesian(90, 100, 75, 180 - angle)
    x2, y2 = polar_to_cartesian(90, 100, 85, 180 - angle)
    tick_marks += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#888" stroke-width="2" />'

with col1:
    st.markdown(f"""
    <div style="position: relative; width: 100%; max-width: 180px; height: 100%; margin: auto; overflow: hidden;">
        <div style='text-align: center; font-size: 14px; color: #66aaff; font-weight: bold;'>CPU</div>
        <svg width="100%" height="100%" viewBox="0 0 180 100" preserveAspectRatio="xMidYMid meet">
            {tick_marks}
            <path d="M10,100 A80,80 0 0,1 170,100"
                  fill="none"
                  stroke="rgba(30,30,30,0.3)"
                  stroke-width="15"/>
            <path d="{semicircle_path(cpu_percent)}"
                  fill="none"
                  stroke="#66aaff"
                  stroke-width="15"
                  stroke-linecap="round"
                  filter="drop-shadow(0px 0px 6px #66aaff)" />
        </svg>
        <div style="position: absolute; bottom: 12px; width: 100%; text-align: center; font-size: 20px; font-weight: bold; color: #66aaff;">
            {cpu_percent:.1f} %
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="position: relative; width: 100%; max-width: 180px; height: 100%; max-height: 180px; margin: auto; overflow: hidden;">
        <div style='text-align: center; font-size: 14px; color: #ffaa66; font-weight: bold;'>RAM</div>
        <svg width="100%" height="100%" viewBox="0 0 180 100" preserveAspectRatio="xMidYMid meet">
            {tick_marks}
            <path d="M10,100 A80,80 0 0,1 170,100"
                  fill="none"
                  stroke="rgba(30,30,30,0.3)"
                  stroke-width="15"/>
            <path d="{semicircle_path(ram_percent)}"
                  fill="none"
                  stroke="#ffaa66"
                  stroke-width="15"
                  stroke-linecap="round"
                  filter="drop-shadow(0px 0px 6px #66aaff)" />
        </svg>
        <div style="position: absolute; bottom: 12px; width: 100%; text-align: center; font-size: 20px; font-weight: bold; color: #ffaa66;">
            {ram_percent:.1f} %
        </div>
    </div>
    """, unsafe_allow_html=True)

def check_api_health(url):
    try:
        response = requests.get(url, timeout=2)
        if response.ok:
            data = response.json()
            return "running" if data.get("status") == "running" else "offline"
    except:
        return "offline"
    return "offline"

st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
st.markdown("Container Status:")

system_services = {
    "Dashboard": "running",  # always expected to be running
    "Model Manager": check_api_health("http://model_manager:8008/health"),
    "InfluxDB Client": check_api_health("http://api_influxdata:8009/health")
}

status_colors = {
    "running": "#28c76f",   # green
    "offline": "#777"      # gray
}

cols_system = st.columns(4)
order = ["Dashboard", "Model Manager", "InfluxDB Client", ""]

for col, name in zip(cols_system, system_services.keys()):
    if name == "":
        col.markdown("")  # Leere Spalte
        continue
    status = system_services[name]
    color = status_colors.get(status, "#777")
    col.markdown(
        f"""
        <div style='
            display: block;
            width: 100%;
            max-width: 220px;
            min-width: 145px;
            border: 1px solid #444;
            border-radius: 8px;
            padding: 16px;
            background-color: #1e1e1e;
            height: 80px;
            position: relative;
        '>
            <div style='font-weight: bold; color: white;'>{name}</div>
            <div style='
                position: absolute;
                bottom: 15px;
                left: 16px;
                display: flex;
                align-items: center;
                gap: 6px;
            '>
                <div style='
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    background-color: {color};
                    box-shadow: 0 0 6px {color};
                '></div>
                <span style='color: white; font-size: 0.8em;'>{status}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Model container control section
st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
st.subheader("Container Control:")

def fetch_status():
    try:
        response = requests.get("http://model_manager:8008/status/lstm")
        if response.ok:
            is_running = response.json().get("running", False)
            return "running" if is_running else "stopped"
    except Exception:
        return "unreachable"
    return "unreachable"

if "container_status" not in st.session_state:
    st.session_state.container_status = fetch_status()

status_color = "#28c76f" if st.session_state.container_status == "running" else "#ea5455"
status_glow = status_color



# --- 3+1 grid layout for models and controls ---
# First row: container boxes
cols_top = st.columns(4)
model_names = ["LSTM", "XGBoost", "CNN", "..."]
model_status = {
    "LSTM": st.session_state.container_status,
    "XGBoost": "offline",
    "CNN": "offline",
    "...": "offline"
}

status_colors = {
    "running": "#28c76f",
    "stopped": "#ea5455",
    "offline": "#777"
}

for col, name in zip(cols_top, model_names):
    status = model_status[name]
    color = status_colors.get(status, "#b6b6b6")
    col.markdown(
        f"""
        <div style='
            display: block;
            width: 100%;
            max-width: 220px;
            min-width: 85px;
            border: 1px solid #444;
            border-radius: 8px;
            padding: 16px;
            background-color: #1e1e1e;
            height: 80px;
            position: relative;
        '>
            <div style='font-weight: bold; color: white;'>{name}</div>
            <div style='
                position: absolute;
                bottom: 12px;
                left: 16px;
                display: flex;
                align-items: center;
                gap: 6px;
            '>
                <div style='
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    background-color: {color};
                    box-shadow: 0 0 6px {color};
                '></div>
                <span style='color: white; font-size: 0.8em;'>{status}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Second row: container buttons
cols_bottom = st.columns(4)
for col, name in zip(cols_bottom, model_names):
    if name == "LSTM":
        button_label = "Stop" if st.session_state.container_status == "running" else "Start"
        with col:
            if st.button(button_label, key=f"{name.lower()}_button"):
                action = "stop" if st.session_state.container_status == "running" else "start"
                try:
                    response = requests.post(f"http://model_manager:8008/control/lstm", json={"action": action})
                    if response.ok:
                        with st.spinner(f"Waiting for container to {action}..."):
                            progress = st.progress(0)
                            for i in range(100):
                                time.sleep(0.1)
                                current_status = fetch_status()
                                progress.progress(i + 1)
                                if action == "start" and current_status == "running":
                                    st.session_state.container_status = "running"
                                    break
                                elif action == "stop" and current_status == "stopped":
                                    st.session_state.container_status = "stopped"
                                    break
                            progress.empty()
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        with col:
            st.button("disabled", key=f"{name.lower()}_button")

# --- Miro iframe embed for model container overview ---
import streamlit.components.v1 as components

components.html(
    '''
    <iframe width="768" height="432" 
            src="https://miro.com/app/live-embed/uXjVINWuZhc=/?focusWidget=3458764629836705251&embedMode=view_only_without_ui&embedId=380901762773" 
            frameborder="0" 
            scrolling="no" 
            allow="fullscreen; clipboard-read; clipboard-write" 
            allowfullscreen>
    </iframe>
    ''',
    height=450
)

st.markdown("---")
st.subheader("Actions")

col_train, col_forecast = st.columns([1, 1])

with col_train:
    if st.button("Train Model", use_container_width=True):
        st.switch_page("pages/2_training.py")

with col_forecast:
    if st.button("View Forecast Overview", use_container_width=True):
        st.switch_page("pages/3_forecast_overview.py")

st.markdown("---")
st.subheader("IPV Solar Modules Database")

import os
import pandas as pd
from datetime import datetime

backup_path = "backup/last_module_data.pkl"

def load_backup(path):
    if os.path.exists(path):
        df = pd.read_pickle(path)
        timestamp = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%d-%m-%y %H:%M")
        return df, timestamp
    return pd.DataFrame(), "No backup found"

def fetch_and_store_data():
    try:
        response = requests.get("http://api_influxdata:8009/modules/overview", timeout=60)
        if response.ok:
            df = pd.DataFrame(response.json())
            df.to_pickle(backup_path)
            return df, datetime.now().strftime("%d-%m-%y %H:%M")
    except Exception as e:
        st.error(f"API request failed: {e}")
    return None, None

if st.button("Reload Data"):
    with st.spinner("Fetching data from API..."):
        new_df, new_time = fetch_and_store_data()
    if new_df is not None:
        st.success("Data successfully reloaded from API.")
        df, status_time = new_df, new_time
    else:
        df, status_time = load_backup(backup_path)
else:
    df, status_time = load_backup(backup_path)

st.markdown(f"**Status: {status_time}**")
st.dataframe(df, use_container_width=True)


# --- Saved Training Settings Overview ---
from pathlib import Path
import json

st.markdown("---")
st.subheader("ML-Model Database")

# Modelltypen (Ordner in results/trained_models)
if Path("/app/results").exists():
    models_base = Path("/app/results/trained_models")
else:
    models_base = Path(__file__).parent.parent / "results" / "trained_models"

if models_base.exists():
    for model_type_dir in sorted(models_base.iterdir()):
        if model_type_dir.is_dir():
            model_type = model_type_dir.name.upper()
            st.markdown(f"#### {model_type} Models")
            model_rows = []
            for model_folder in sorted(model_type_dir.iterdir()):
                if model_folder.is_dir():
                    # Suche nach model_config und summary
                    config_files = list(model_folder.glob("model_config_*.json"))
                    summary_files = list(model_folder.glob("training_summary_*.txt"))
                    config = {}
                    summary = {}
                    if config_files:
                        with open(config_files[0], "r") as f:
                            config = json.load(f)
                    if summary_files:
                        with open(summary_files[0], "r") as f:
                            lines = f.readlines()
                        # Extrahiere relevante Infos aus der Summary
                        for line in lines:
                            if ":" in line:
                                key, value = line.split(":", 1)
                                summary[key.strip()] = value.strip()
                    # Row for the table
                    model_rows.append({
                        "Model Name": config.get("model_name", model_folder.name),
                        "Train Time": summary.get("Test time", "-"),
                        "Features": ", ".join(config.get("training_settings", {}).get("features", [])),
                        "Output": ", ".join(config.get("training_settings", {}).get("output", [])),
                        "Epochs": config.get("training_settings", {}).get("epochs", "-"),
                        "Batch Size": config.get("training_settings", {}).get("batch_size", "-"),
                        "Learning Rate": config.get("training_settings", {}).get("learning_rate", "-"),
                        "Loss Function": config.get("training_settings", {}).get("loss_function", "-"),
                        "Final Training Loss": summary.get("Final Training Loss", "-"),
                        "Final Validation Loss": summary.get("Final Validation Loss", "-"),
                        "Folder": str(model_folder)
                    })
            if model_rows:
                df_models = pd.DataFrame(model_rows)
                st.dataframe(df_models, use_container_width=True)
            else:
                st.info(f"No trained {model_type} models found.")
else:
    st.info("No trained models directory found.")

st.markdown("---")
st.subheader("ML-Model Settings Database")

# Check if running in container (results mounted as /app/results) or local development
if Path("/app/results").exists():
    settings_dir = Path("/app/results/model_configs")
else:
    settings_dir = Path(__file__).parent.parent / "results" / "model_configs"
settings_dir.mkdir(parents=True, exist_ok=True)
# Show status timestamp for settings files
settings_files = list(settings_dir.glob("*_settings.json")) + list(settings_dir.glob("*_config.json"))
settings_status = datetime.fromtimestamp(
    max(f.stat().st_mtime for f in settings_files)
).strftime("%d-%m-%y %H:%M") if settings_dir.exists() and settings_files else "No settings files"
st.markdown(f"**Status: {settings_status}**")

if settings_dir.exists():
    # Suche nach sowohl *_settings.json als auch *_config.json Dateien
    json_files = settings_files
    if json_files:
        records = []
        for file in json_files:
            try:
                with open(file, "r") as f:
                    content = json.load(f)
                    records.append(content)
            except Exception as e:
                st.warning(f"Could not load {file.name}: {e}")

        if records:
            df_settings = pd.DataFrame(records)
            # Ensure all list-like values are converted to strings for display
            for col in ["features", "output", "selected_modules"]:
                if col in df_settings.columns:
                    df_settings[col] = df_settings[col].apply(
                        lambda x: ", ".join(x) if isinstance(x, list) else (str(x) if x is not None else "")
                    )
            if "date_selection" in df_settings.columns:
                df_settings["date_selection"] = df_settings["date_selection"].apply(lambda x: json.dumps(x) if isinstance(x, dict) else str(x))
            st.dataframe(df_settings, use_container_width=True)
            if st.button("Reload Settings"):
                st.rerun()
        else:
            st.info("No valid JSON settings found.")
    else:
        st.info("No settings files available.")
else:
    st.warning("Settings directory not found.")
