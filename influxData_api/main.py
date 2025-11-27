from fastapi import FastAPI, Request
import pandas as pd
import os
from fastapi.responses import FileResponse

from data.module_data_splitter import ModuleDataSplitter
from data.influxDB_client import influxClient

app = FastAPI()

# Initialize the Influx client
db_client = influxClient("data/.env")

@app.get("/modules/overview")
def get_module_overview():
    df = db_client.get_module_overview()

    # Define module types by name pattern
    type_mapping = {
        "Atersa": "Silicon",
        "Sanyo": "Silicon",
        "Solon": "Silicon",
        "Sun_Power": "Silicon",
        "Perovskite": "Perovskite"
    }

    def classify(name):
        for key in type_mapping:
            if key in name:
                return type_mapping[key]
        return "Unknown"

    df["Module Type"] = df["Name"].apply(classify)
    df = df[["MAC", "Name", "Module Type", "Fields", "Active (last 24h)", "Start Timestamp", "Last Timestamp"]]
    return df.to_dict(orient="records")

# CSV-serving endpoints
def read_csv_as_dict(path):
    try:
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

@app.get("/csv/influx_export")
def get_influx_export():
    return FileResponse("data/influx_export.csv", media_type="text/csv", filename="influx_export.csv")

@app.get("/csv/perovskite_modules")
def get_perovskite_modules():
    return FileResponse("data/split_modules/perovskite_modules.csv", media_type="text/csv", filename="perovskite_modules.csv")

@app.get("/csv/silicon_modules")
def get_silicon_modules():
    return FileResponse("data/split_modules/silicon_modules.csv", media_type="text/csv", filename="silicon_modules.csv")

@app.get("/health")
def health_check():
    return {"status": "running", "service": "api_influxdata"}


# New POST route for training data queries from model container
@app.post("/query/training_data")
async def query_training_data(request: Request):
    body = await request.json()
    print("Received training data query:", body)

    model_name = body.get("model_name", "unnamed_model")
    model_type = body.get("model_type", "LSTM")
    module_type = body.get("module_type", "Silicon")
    use_all = body.get("use_all_modules", "Use all modules of this type")
    selected_modules = body.get("selected_modules", "all")
    features = body.get("features", [])
    outputs = body.get("output", [])
    date_selection = body.get("date_selection", "Use all historical data")

    # Mapping for output parameters
    output_map = {
        "Current": "I",
        "Voltage": "U",
        "Power": "P"
    }
    mapped_outputs = [output_map.get(o, o) for o in outputs]

    try:
        df = db_client.get_training_data(
            dataset_name=f"{model_name}_data",
            model_type=model_type,
            module_type=module_type,
            use_all_modules=use_all,
            selected_modules=selected_modules,
            features=features,
            outputs=mapped_outputs,
            date_selection=date_selection
        )
        return {
            "status": "success",
            "path": f"lstm_model/data/{model_name}_data.csv",
            "record_count": len(df)
        }
    except Exception as e:
        return {"error": str(e)}