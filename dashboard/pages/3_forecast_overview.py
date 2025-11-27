import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

# API Endpoints - moved to top
LSTM_API_URL = "http://lstm_model:8000"
MODEL_MANAGER_URL = "http://model_manager:8008"

st.set_page_config(page_title="Forecast Overview", layout="wide")

# Page Header
st.title("Forecast Overview")

# Sidebar for configuration
st.sidebar.header("Configuration")

# --- Load Aggregated Forecast Input Data and Forecast Config (direkter Zugriff auf gemountete Ergebnisse) ---
AGGR_CSV_PATH = "/app/results/forecast_data/aggrData/latest_forecast_input_aggr.csv"
FORECAST_CONFIG_PATH = "/app/results/forecast_data/forecast_config.json"
df_aggr = None
forecast_config = None

try:
    if os.path.exists(AGGR_CSV_PATH):
        df_aggr = pd.read_csv(AGGR_CSV_PATH)
    else:
        st.warning(f"Aggregated forecast input CSV not found: {AGGR_CSV_PATH}")
except Exception as e:
    st.error(f"Error loading aggregated forecast input CSV: {e}")

try:
    if os.path.exists(FORECAST_CONFIG_PATH):
        with open(FORECAST_CONFIG_PATH, "r") as f:
            forecast_config = json.load(f)
    else:
        st.warning(f"Forecast config not found: {FORECAST_CONFIG_PATH}")
except Exception as e:
    st.error(f"Error loading forecast config: {e}")

# --- Streamlit Expander for Aggregated Forecast Input and Forecast-Config ---
with st.expander("Show Aggregated Forecast Input (aggr CSV)"):
    if df_aggr is not None:
        st.dataframe(df_aggr, use_container_width=True)
    else:
        st.info("No aggregated forecast input loaded.")

with st.expander("Show Forecast Config (JSON)"):
    if forecast_config is not None:
        st.json(forecast_config)
    else:
        st.info("No forecast config loaded.")

# Model selection
@st.cache_data
def get_available_models():
    """Lädt verfügbare Modelle von der API oder aus dem lokalen Verzeichnis"""
    try:
        # Versuche zuerst die API
        st.write("Trying API to get models...")
        response = requests.get(f"{LSTM_API_URL}/models", timeout=10)
        st.write(f"API Response Status: {response.status_code}")
        if response.status_code == 200:
            models_data = response.json()
            st.write(f"API returned {len(models_data)} models")
            model_names = [model['model_name'] for model in models_data]
            st.write(f"Model names: {model_names}")
            st.write("--------------------------------")
            return model_names
    except Exception as e:
        st.write(f"API Error: {e}")
    
    # Fallback: Lokales Verzeichnis (Container-Pfad)
    try:
        st.write("Trying local directory...")
        # Check if running in container (results mounted as /app/results) or local development
        if Path("/app/results").exists():
            models_dir = "/app/results/trained_models/lstm"
        else:
            models_dir = str(Path(__file__).parent.parent.parent / "results" / "trained_models" / "lstm")
        st.write(f"Looking in: {models_dir}")
        st.write(f"Current working directory: {os.getcwd()}")
        if os.path.exists(models_dir):
            # Look for model directories (YYYYMMDD_HHMM_lstm/)
            model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d.endswith('_lstm')]
            model_names = [d.replace('_lstm', '') for d in model_dirs]
            st.write(f"Found model directories: {model_dirs}")
            st.write(f"Model names: {model_names}")
            return model_names
        else:
            st.write(f"Directory not found: {models_dir}")
        return []
    except Exception as e:
        st.error(f"Fehler beim Laden der Modelle: {e}")
        return []

def get_model_info(model_name: str) -> Dict:
    """Loads detailed information about a model from API"""
    try:
        response = requests.get(f"{LSTM_API_URL}/models/{model_name}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    # Fallback: Local settings file
    return load_model_settings(model_name)

def load_model_settings(model_name: str) -> Dict:
    """Loads model settings from local file"""
    try:
        # Look for model config in the new structure
        # Check if running in container (results mounted as /app/results) or local development
        if Path("/app/results").exists():
            settings_path = f"/app/results/trained_models/lstm/{model_name}_lstm/model_config_{model_name}_lstm.json"
        else:
            settings_path = str(Path(__file__).parent.parent.parent / "results" / "trained_models" / "lstm" / f"{model_name}_lstm" / f"model_config_{model_name}_lstm.json")
        
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                return json.load(f)
        
        return {}
    except Exception as e:
        st.error(f"Error loading model settings: {e}")
        return {}

available_models = get_available_models()

if not available_models:
    st.warning("No trained models found. Please train a model first.")
    st.stop()

# Model selection in sidebar
st.sidebar.subheader("Model Selection")

# Single model selection
selected_single_model = st.sidebar.selectbox(
    "Single model for detailed view:",
    available_models,
    index=0 if available_models else None
)

# Multiple models for comparison
st.sidebar.subheader("Model Comparison")
comparison_models = st.sidebar.multiselect(
    "Select models for comparison:",
    available_models,
    default=available_models[:2] if len(available_models) >= 2 else available_models
)

# Date selection
st.sidebar.subheader("Forecast Parameters")
forecast_date = st.sidebar.date_input(
    "Start date for forecast:",
    value=datetime.now().date(),
    max_value=datetime.now().date()
)

forecast_horizon = st.sidebar.slider(
    "Forecast horizon (days):",
    min_value=1,
    max_value=7,
    value=3
)

def get_forecast_data(model_name: str, start_date: datetime, horizon_days: int) -> Optional[pd.DataFrame]:
    """Fetches forecast data from API"""
    try:
        # API request to LSTM service
        forecast_request = {
            "model_name": model_name,
            "start_date": start_date.isoformat(),
            "horizon_days": horizon_days
        }
        
        response = requests.post(f"{LSTM_API_URL}/forecast", json=forecast_request, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Convert ISO timestamps to datetime
            timestamps = [datetime.fromisoformat(ts) for ts in data['timestamps']]
            
            return pd.DataFrame({
                'timestamp': timestamps,
                'forecast_power': data['forecast_values'],
                'model': model_name
            })
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error to API: {e}")
        # Fallback: Simulate data for demo purposes
        st.warning("Using simulated data as fallback...")
        
        timestamps = []
        for day in range(horizon_days):
            for hour in range(24):
                for minute in range(0, 60, 15):  # 15-minute intervals
                    dt = start_date + timedelta(days=day, hours=hour, minutes=minute)
                    timestamps.append(dt)
        
        # Simulate forecast values
        np.random.seed(hash(model_name) % 2**32)
        base_power = 0.5 + 0.3 * np.sin(np.arange(len(timestamps)) * 2 * np.pi / 96)
        noise = np.random.normal(0, 0.1, len(timestamps))
        forecast_values = np.maximum(0, base_power + noise)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'forecast_power': forecast_values,
            'model': model_name
        })
    except Exception as e:
        st.error(f"Error fetching forecast data for {model_name}: {e}")
        return None

def create_forecast_plot(df: pd.DataFrame, title: str = "Forecast") -> go.Figure:
    """Creates a Plotly graph for the forecast"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['forecast_power'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def create_comparison_plot(comparison_data: List[pd.DataFrame]) -> go.Figure:
    """Creates a comparison plot for multiple models"""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, df in enumerate(comparison_data):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['forecast_power'],
            mode='lines+markers',
            name=df['model'].iloc[0],
            line=dict(color=color, width=2),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title="Model Comparison - Forecasts",
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

# Main content
if selected_single_model:
    st.header(f"Single Model: {selected_single_model}")
    
    # Load model information
    model_info = get_model_info(selected_single_model)
    
    # Model information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", model_info.get('model_type', 'LSTM'))
    
    with col2:
        training_date = model_info.get('training_date', 'Unknown')
        st.metric("Training Date", training_date)
    
    with col3:
        st.metric("Forecast Horizon", f"{forecast_horizon} days")
    
    with col4:
        # Display performance metrics
        perf_metrics = model_info.get('performance_metrics', {})
        if 'final_val_loss' in perf_metrics and perf_metrics['final_val_loss'] is not None:
            st.metric("Val. Loss", f"{perf_metrics['final_val_loss']:.4f}")
        else:
            st.metric("Performance", "N/A")
    
    # Extended model information
    with st.expander("Detailed Model Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Features")
            features = model_info.get('input_features', [])
            if features:
                for feature in features:
                    st.write(f"• {feature}")
            else:
                st.write("No features available")
        
        with col2:
            st.subheader("Output Features")
            outputs = model_info.get('output_features', [])
            if outputs:
                for output in outputs:
                    st.write(f"• {output}")
            else:
                st.write("No outputs available")
        
        # Performance metrics
        if 'performance_metrics' in model_info:
            st.subheader("Performance Metrics")
            metrics = model_info['performance_metrics']
            
            if 'final_train_loss' in metrics and metrics['final_train_loss'] is not None:
                st.metric("Final Training Loss", f"{metrics['final_train_loss']:.4f}")
            
            if 'final_val_loss' in metrics and metrics['final_val_loss'] is not None:
                st.metric("Final Validation Loss", f"{metrics['final_val_loss']:.4f}")
            
            if 'epochs_trained' in metrics:
                st.metric("Trained Epochs", metrics['epochs_trained'])
    
    # Load forecast data
    forecast_data = get_forecast_data(selected_single_model, forecast_date, forecast_horizon)
    
    if forecast_data is not None:
        # Main plot
        fig = create_forecast_plot(forecast_data, f"Forecast: {selected_single_model}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            max_power = forecast_data['forecast_power'].max()
            st.metric("Max Power", f"{max_power:.2f} kW")
        
        with col2:
            avg_power = forecast_data['forecast_power'].mean()
            st.metric("Avg Power", f"{avg_power:.2f} kW")
        
        with col3:
            total_energy = forecast_data['forecast_power'].sum() * 0.25  # 15-minute intervals
            st.metric("Total Energy", f"{total_energy:.2f} kWh")
        
        with col4:
            peak_hour = forecast_data.loc[forecast_data['forecast_power'].idxmax(), 'timestamp']
            st.metric("Peak Time", peak_hour.strftime("%H:%M"))
        
        # Show raw data
        with st.expander("Show Raw Data"):
            st.dataframe(forecast_data)

# Model comparison
if len(comparison_models) >= 2:
    st.header("Model Comparison")
    
    # Load forecast data for all comparison models
    comparison_data = []
    for model in comparison_models:
        data = get_forecast_data(model, forecast_date, forecast_horizon)
        if data is not None:
            comparison_data.append(data)
    
    if comparison_data:
        # Comparison plot
        fig = create_comparison_plot(comparison_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison table
        st.subheader("Comparison Statistics")
        
        comparison_stats = []
        for data in comparison_data:
            model_name = data['model'].iloc[0]
            stats = {
                'Model': model_name,
                'Max Power (kW)': f"{data['forecast_power'].max():.2f}",
                'Avg Power (kW)': f"{data['forecast_power'].mean():.2f}",
                'Total Energy (kWh)': f"{data['forecast_power'].sum() * 0.25:.2f}",
                'Std Deviation': f"{data['forecast_power'].std():.2f}"
            }
            comparison_stats.append(stats)
        
        comparison_df = pd.DataFrame(comparison_stats)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Difference analysis
        if len(comparison_data) == 2:
            st.subheader("Difference Analysis")
            
            # Align data points
            df1, df2 = comparison_data[0], comparison_data[1]
            merged_df = pd.merge(df1, df2, on='timestamp', suffixes=('_1', '_2'))
            merged_df['difference'] = merged_df['forecast_power_1'] - merged_df['forecast_power_2']
            
            # Difference plot
            fig_diff = go.Figure()
            fig_diff.add_trace(go.Scatter(
                x=merged_df['timestamp'],
                y=merged_df['difference'],
                mode='lines+markers',
                name=f'Difference ({df1["model"].iloc[0]} - {df2["model"].iloc[0]})',
                line=dict(color='red', width=2),
                marker=dict(size=4)
            ))
            
            fig_diff.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig_diff.update_layout(
                title="Difference between Models",
                xaxis_title="Time",
                yaxis_title="Power Difference (kW)",
                template='plotly_white'
            )
            
            st.plotly_chart(fig_diff, use_container_width=True)
            
            # Difference statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                mean_diff = merged_df['difference'].mean()
                st.metric("Avg Difference", f"{mean_diff:.3f} kW")
            
            with col2:
                max_diff = merged_df['difference'].abs().max()
                st.metric("Max Deviation", f"{max_diff:.3f} kW")
            
            with col3:
                std_diff = merged_df['difference'].std()
                st.metric("Std Deviation", f"{std_diff:.3f} kW")

# Footer
st.markdown("---")
st.markdown("*Forecast Overview - Model Comparison and Analysis*")
