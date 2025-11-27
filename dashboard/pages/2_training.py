import streamlit as st
import json
from pathlib import Path
import requests
import time

st.set_page_config(page_title="Train Model", layout="wide")
st.title("Train a Model")

import pandas as pd
# Check if running in container (results mounted as /app/results) or local development
if Path("/app/results").exists():
    settings_path = Path("/app/results/model_configs")
else:
    settings_path = Path(__file__).parent.parent.parent / "results" / "model_configs"
settings_path.mkdir(parents=True, exist_ok=True)
# Suche nach sowohl *_settings.json als auch *_config.json Dateien
json_files = list(settings_path.glob("*_settings.json")) + list(settings_path.glob("*_config.json"))

st.subheader("Select a model to train")

# Load Model expander
with st.expander("Load Existing Model Settings"):
    mode = st.radio("Select loading method", ["File picker", "From table", "Upload file"], horizontal=True)
    loaded = {}

    if mode == "File picker":
        json_filenames = [f.name for f in json_files]
        selected_file = st.selectbox("Choose a settings file", json_filenames)
        if selected_file and st.button("Load selected file"):
            selected_path = settings_path / selected_file
            with open(selected_path, "r") as f:
                st.session_state.loaded_settings = json.load(f)
            st.success(f"Loaded settings from {selected_file}")

    elif mode == "From table":
        if json_files:
            import datetime

            table_data = []
            for f in json_files:
                with open(f, "r") as jf:
                    content = json.load(jf)
                table_data.append({
                    "Select": False,
                    "Filename": f.name,
                    "Last Modified": datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    "Model Type": content.get("model_type", ""),
                    "Module Type": content.get("module_type", ""),
                    "Features": ", ".join(content.get("features", [])),
                    "Epochs": content.get("epochs", ""),
                    "Batch Size": content.get("batch_size", ""),
                    "LR": content.get("learning_rate", ""),
                    "Loss": content.get("loss_function", "")
                })

            df = pd.DataFrame(table_data)
            edited_df = st.data_editor(
                df,
                column_config={
                    "Select": st.column_config.CheckboxColumn(required=True),
                    "Filename": st.column_config.TextColumn(disabled=True)
                },
                hide_index=True,
                use_container_width=True,
                key="settings_table"
            )

            # Nur eine Auswahl zulassen
            if (edited_df["Select"] == True).sum() > 1:
                st.warning("Please select only one file.")
                selected_rows = pd.DataFrame()
            else:
                selected_rows = edited_df[edited_df["Select"] == True]

            if st.button("Load selected table file"):
                if not selected_rows.empty:
                    selected_filename = selected_rows.iloc[0]["Filename"]
                    selected_path = settings_path / selected_filename
                    with open(selected_path, "r") as f:
                        st.session_state.loaded_settings = json.load(f)
                    # Addition: Set output_select and date_selection
                    loaded_data = st.session_state.loaded_settings
                    output_feature = loaded_data.get("output", [])
                    if isinstance(output_feature, list):
                        st.session_state.output_select = output_feature
                    else:
                        st.session_state.output_select = [output_feature]

                    date_selection = loaded_data.get("date_selection", {})
                    if isinstance(date_selection, dict):
                        st.session_state.date_selection_radio = date_selection.get("mode", "Use all historical data").capitalize()
                        if date_selection.get("mode") == "custom":
                            from datetime import date
                            start_date_str = date_selection.get("start")
                            end_date_str = date_selection.get("end")
                            if start_date_str and end_date_str:
                                st.session_state.start_date = date.fromisoformat(start_date_str)
                                st.session_state.end_date = date.fromisoformat(end_date_str)
                            else:
                                st.session_state.start_date = date.today()
                                st.session_state.end_date = date.today()
                        elif date_selection.get("mode") == "last":
                            st.session_state.last_hours_input = int(date_selection.get("value", 48))
                    st.success(f"Loaded settings from {selected_filename}")
        else:
            st.info("No saved settings files found.")

    elif mode == "Upload file":
        uploaded_file = st.file_uploader("Upload a settings file", type=["json"])
        if uploaded_file is not None:
            try:
                uploaded_settings = json.load(uploaded_file)
                st.session_state.loaded_settings = uploaded_settings
                # Ergänzung: Setze output_select und date_selection
                loaded_data = st.session_state.loaded_settings
                output_feature = loaded_data.get("output", [])
                if isinstance(output_feature, list):
                    st.session_state.output_select = output_feature
                else:
                    st.session_state.output_select = [output_feature]

                date_selection = loaded_data.get("date_selection", {})
                if isinstance(date_selection, dict):
                    st.session_state.date_selection_radio = date_selection.get("mode", "Use all historical data").capitalize()
                    if date_selection.get("mode") == "custom":
                        from datetime import date
                        start_date_str = date_selection.get("start")
                        end_date_str = date_selection.get("end")
                        if start_date_str and end_date_str:
                            st.session_state.start_date = date.fromisoformat(start_date_str)
                            st.session_state.end_date = date.fromisoformat(end_date_str)
                        else:
                            st.session_state.start_date = date.today()
                            st.session_state.end_date = date.today()
                    elif date_selection.get("mode") == "last":
                        st.session_state.last_hours_input = int(date_selection.get("value", 48))
                # optional: speichere hochgeladene Datei im settings_path
                # Check if running in container (results mounted as /app/results) or local development
                if Path("/app/results").exists():
                    settings_dir = Path("/app/results/model_configs")
                else:
                    settings_dir = Path(__file__).parent.parent.parent / "results" / "model_configs"
                settings_dir.mkdir(exist_ok=True)
                save_path = settings_dir / f"{uploaded_settings.get('model_name', 'uploaded')}_settings.json"
                with open(save_path, "w") as f:
                    json.dump(uploaded_settings, f, indent=4)
                st.success("Settings loaded from uploaded file.")
            except Exception as e:
                st.error(f"Failed to load settings: {e}")

loaded = st.session_state.get("loaded_settings", {})

model_choice_default = loaded.get("model_type", "")
module_type_default = loaded.get("module_type", "")
model_name_default = loaded.get("model_name", "")
use_all_modules_default = loaded.get("use_all_modules", "Use all modules of this type")
features_default = loaded.get("features", [])
output_default = loaded.get("output", [])
date_selection_default = loaded.get("date_selection", "Use all historical data")
use_validation_default = loaded.get("use_validation_set", "No")
epochs_default = loaded.get("epochs", 5)
batch_size_default = loaded.get("batch_size", 32)
learning_rate_default = loaded.get("learning_rate", 0.001)
shuffle_default = loaded.get("shuffle", False)
loss_function_default = loaded.get("loss_function", "RMSE")

# Vor dem Slider:
validation_set_default = loaded.get("validation_set", {})
if validation_set_default and "validation_split" in validation_set_default:
    validation_split_default = 1 - float(validation_set_default.get("validation_split", 0.15))
else:
    validation_split_default = 0.85

# Auswahl des Modells
model_choice = st.selectbox(
    "Select a model to train",
    ["", "LSTM", "XGBoost", "CNN"],
    index=["", "LSTM", "XGBoost", "CNN"].index(model_choice_default) if model_choice_default in ["", "LSTM", "XGBoost", "CNN"] else 0,
    key="model_choice"
)
# Clear session state if a new model is being manually selected (not loaded)
if "loaded_settings" in st.session_state and model_choice != st.session_state["loaded_settings"].get("model_type", ""):
    del st.session_state["loaded_settings"]
    st.rerun()
module_type = ""
if model_choice == "":
    st.warning("Please select a model first.")
    st.markdown("<div style='opacity: 0.3;'>", unsafe_allow_html=True)
else:
    module_type = st.selectbox(
        "Module type - Trainings Data",
        ["", "silicon", "perovskite", "both"],
        index=["", "silicon", "perovskite", "both"].index(module_type_default) if module_type_default in ["", "silicon", "perovskite", "both"] else 0,
        key="module_type_select"
    )

    if module_type == "":
        st.warning("Please select a module type.")
        st.markdown("<div style='opacity: 0.3;'>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='opacity: 1;'>", unsafe_allow_html=True)

        st.subheader(f"Training configuration for {model_choice} with {module_type}")

        import datetime
        suffix = {
            "silicon": "Silic",
            "perovskite": "Perovsk",
            "both": "Sili-Pero"
        }.get(module_type, "Unknown")

        default_model_name = f"{datetime.datetime.now().strftime('%d%m%y')}-{model_choice}-{suffix}_"
        model_name = st.text_input(
            "Model name",
            value=model_name_default if model_name_default else default_model_name,
            key="model_name_input"
        )

        use_all_modules = st.radio(
            "Module selection",
            ["Use all modules of this type", "Select specific modules"],
            index=["Use all modules of this type", "Select specific modules"].index(use_all_modules_default) if use_all_modules_default in ["Use all modules of this type", "Select specific modules"] else 0,
            key="use_all_modules_radio"
        )

        selected_modules = []
        if use_all_modules == "Select specific modules":
            module_options = {
                "silicon": [
                    "Atersa_1_1", "Atersa_2_1", "Atersa_3_1", "Atersa_4_1", "Atersa_5_1", "Atersa_6_1",
                    "Sanyo_2_1", "Sanyo_3_1", "Sanyo_4_1", "Sanyo_5_1",
                    "Solon_1_1", "Solon_1_2", "Solon_2_1", "Solon_2_2", "Solon_3_2",
                    "Sun_Power_1_1", "Sun_Power_2_1", "Sun_Power_3_1", "Sun_Power_4_1", "Sun_Power_5_1"
                ],
                "perovskite": [
                    "Perovskite_1_1", "Perovskite_1_2", "Perovskite_1_3",
                    "Perovskite_2_1", "Perovskite_2_2", "Perovskite_2_3"
                ],
                "both": []
            }
            if module_type in module_options and module_type != "both":
                import pandas as pd
                module_df = pd.DataFrame({
                    "Module": module_options[module_type],
                    "Include": [False] * len(module_options[module_type])
                })
                select_all = st.button("Select all modules")
                unselect_all = st.button("Unselect all modules")

                if select_all:
                    module_df["Include"] = True
                elif unselect_all:
                    module_df["Include"] = False

                edited_df = st.data_editor(module_df, hide_index=True, use_container_width=True)
                selected_modules = edited_df[edited_df["Include"]]["Module"].tolist()
            elif module_type == "both":
                st.info("Custom selection for mixed module types is not supported yet.")

        features = st.multiselect(
            "Input Features",
            ["Temp", "AmbTemp", "AmbHmd", "Irr"],
            help="Select input features used to predict the output.",
            default=features_default,
            key="features_multiselect"
        )
        # --- Time feature selection ---
        time_feature_options = ["hour", "minute", "weekday", "month", "day_of_year"]
        time_features_default = loaded.get("time_features", [])
        time_features = st.multiselect(
            "Time Features",
            time_feature_options,
            help="Select time-based features to include in the model.",
            default=time_features_default,
            key="time_features_multiselect"
        )
        output_feature = st.multiselect(
            "Output Features",
            ["P", "U", "I"],
            default=output_default if isinstance(output_default, list) else [output_default],
            help="Select one or more target values your model will predict.",
            key="output_select"
        )
        mode_list = ["Use all historical data", "Select date range", "Last X hours"]
        if isinstance(date_selection_default, dict):
            mode_str = date_selection_default.get("mode", "all")
            mode_index = mode_list.index({
                "all": "Use all historical data",
                "custom": "Select date range",
                "last": "Last X hours"
            }.get(mode_str, "Use all historical data"))
        else:
            mode_index = 0
        date_selection = st.radio(
            "Select training data range from InfluxDB Data",
            mode_list,
            index=mode_index,
            help="Choose whether to use all data, a custom date range, or only recent data.",
            key="date_selection_radio"
        )

        date_range = {}
        if date_selection == "Select date range":
            from datetime import date
            start_date_str = date_selection_default.get("start") if isinstance(date_selection_default, dict) else None
            end_date_str = date_selection_default.get("end") if isinstance(date_selection_default, dict) else None
            
            # Set minimum date to 2024-08-16 (first data written)
            min_date = date(2024, 8, 16)
            
            start_date_value = date.fromisoformat(start_date_str) if start_date_str else min_date
            end_date_value = date.fromisoformat(end_date_str) if end_date_str else date.today()

            # Ensure start_date is not before minimum date
            if start_date_value < min_date:
                start_date_value = min_date

            start_date = st.date_input(
                "Training period from",
                value=start_date_value,
                min_value=min_date,
                max_value=date.today(),
                key="start_date"
            )
            end_date = st.date_input(
                "Training period to",
                value=end_date_value,
                min_value=start_date,
                max_value=date.today(),
                key="end_date"
            )
            date_range = {
                "mode": "custom",
                "start": str(start_date),
                "end": str(end_date)
            }
        elif date_selection == "Last X hours":
            last_hours = st.number_input("How many hours back?", min_value=1, max_value=168, value=48, step=1, key="last_hours_input")
            date_range = {
                "mode": "last",
                "value": last_hours
            }
        else:
            date_range = {"mode": "all"}

        # Validation set always active and disabled
        st.write('Use validation set:')
        st.checkbox('Yes', value=True, disabled=True, key='use_validation_set')

        # Validation Split Slider (English, with help text)
        train_split = st.slider(
            'Train/Validation Split (%)',
            min_value=50, max_value=95, value=int(round(100 * validation_split_default)), step=1,
            help='Percentage of data used for training. A higher value increases training data but reduces validation accuracy insight.'
            )
        validation_split = 1 - (train_split / 100)

        num_epochs = st.number_input(
            "Number of Epochs",
            min_value=1, max_value=1000, value=epochs_default,
            help="Number of complete passes through the dataset. More epochs may improve learning but increase training time and overfitting risk.",
            key="epochs_input"
        )
        batch_size = st.number_input(
            "Batch Size",
            min_value=1, max_value=1024, value=batch_size_default,
            help="Number of samples per training update. Larger batches speed up training but may reduce generalization.",
            key="batch_size_input"
        )
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=1e-5, max_value=1.0, value=learning_rate_default, format="%.5f",
            help="Controls how much the model learns from each step. Higher values speed up learning but may destabilize convergence.",
            key="learning_rate_input"
        )
        shuffle_data = st.checkbox(
            "Shuffle training data",
            value=shuffle_default,
            help="Randomizes training data order to improve learning robustness and avoid local patterns.",
            key="shuffle_checkbox"
        )
        loss_function = st.selectbox(
            "Loss Function",
            ["RMSE", "MSE", "MAE", "MAPE"],
            index=["RMSE", "MSE", "MAE", "MAPE"].index(loss_function_default) if loss_function_default in ["RMSE", "MSE", "MAE", "MAPE"] else 0,
            help="Determines how prediction errors are measured. Choose RMSE for strong penalties on large errors, MAPE for percentage-based accuracy, etc.",
            key="loss_function_select"
        )
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("Save Settings"):
                # Check if running in container (results mounted as /app/results) or local development
                if Path("/app/results").exists():
                    settings_dir = Path("/app/results/model_configs")
                else:
                    settings_dir = Path(__file__).parent.parent.parent / "results" / "model_configs"
                settings_dir.mkdir(exist_ok=True)

                settings = {
                    "model_name": model_name,
                    "model_type": model_choice,
                    "module_type": module_type,
                    "use_all_modules": use_all_modules,
                    "selected_modules": selected_modules if use_all_modules == "Select specific modules" else "all",
                    "features": features,
                    "time_features": time_features,
                    "output": output_feature if isinstance(output_feature, list) else [output_feature],
                    "date_selection": date_range,
                    "validation_set": {
                        'use_validation_set': 'Yes',
                        'validation_split': round(validation_split, 2),
                    },
                    "epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "shuffle": shuffle_data,
                    "loss_function": loss_function
                }

                settings_file = settings_dir / f"{model_name}_settings.json"
                with open(settings_file, "w") as f:
                    json.dump(settings, f, indent=4)
                st.success(f"Settings saved to {settings_file}")
        with col2:
            if st.button("Reset"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.session_state["model_choice"] = ""
                st.rerun()

        # --- Externes Training UI ---
        st.markdown('---')
        external_training_enabled = st.checkbox('Train on external device')
        external_training_url = ''
        if external_training_enabled:
            external_training_url = st.text_input('Enter the URL where the LSTM container is reachable (e.g. http://192.168.1.100:8000)')

        if st.button("Start Training"):
            if external_training_enabled and not external_training_url:
                st.error("Please enter the external device URL!")
                st.stop()
            if external_training_enabled:
                st.info('Have you already started the container on the external device?')
            if model_name.endswith("_") or model_name.strip() == "":
                st.error("Please complete the model name before starting training.")
            elif module_type == "":
                st.error("Please select a module type. The Model will be trained only with data from this selected module type.")
            elif not features:
                st.error("Please select at least one feature.")
            elif not output_feature:
                st.error("Please select at least one output feature.")
            else:
                all_features = time_features + features
                settings = {
                    "model_name": model_name,
                    "model_type": model_choice,
                    "module_type": module_type,
                    "use_all_modules": True if use_all_modules == "Use all modules of this type" else False,
                    "selected_modules": selected_modules if use_all_modules == "Select specific modules" else "all",
                    "features": all_features,
                    "time_features": time_features,
                    "output": output_feature if isinstance(output_feature, list) else [output_feature],
                    "date_selection": date_range,
                    "validation_set": {
                        'use_validation_set': 'Yes',
                        'validation_split': round(validation_split, 2),
                    },
                    "epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "shuffle": shuffle_data,
                    "loss_function": loss_function,
                    "external_training": {
                        "enabled": external_training_enabled,
                        "url": external_training_url
                    }
                }

                # 1. Trainingsdaten asynchron vorbereiten lassen
                try:
                    response = requests.post("http://model_manager:8008/train_model", json=settings, timeout=600.0)
                    if response.status_code == 200:
                        st.success(f"Training data preparation started for model '{model_name}'.")
                        # 2. Polling auf Datenstatus
                        status_url = f"http://lstm_model:8000/data-preparation-status/{model_name}"
                        if external_training_enabled and external_training_url:
                            status_url = external_training_url.rstrip("/") + f"/data-preparation-status/{model_name}"
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        for i in range(300):  # max. 10 Minuten
                            status_resp = requests.get(status_url)
                            status_json = status_resp.json()
                            status = status_json.get("status", "unknown")
                            if status == "ready":
                                progress_bar.progress(100)
                                status_text.success("Training data ready. Starting training...")
                                break
                            elif status == "failed":
                                progress_bar.empty()
                                st.error(f"Data preparation failed: {status_json.get('error')}")
                                st.stop()
                            else:
                                progress_bar.progress(int((i/300)*100))
                                status_text.info(f"Waiting for data preparation... (Status: {status})")
                                time.sleep(2)
                        else:
                            st.error("Timeout while waiting for data preparation.")
                            st.stop()
                        # 3. Training wird wie gewohnt gestartet (das macht der model_manager nach Datenbereitstellung)
                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Failed to call model_manager API: {e}")

st.markdown("</div>", unsafe_allow_html=True)