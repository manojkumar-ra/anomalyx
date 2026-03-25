from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import pandas as pd
import os
import json
from detector import run_detection
from explainer import explain_anomalies
from database import save_scan, get_history, init_db

app = FastAPI(title="AnomalyX", version="1.0.0")

os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
init_db()

# keeping current dataset in memory so we dont have to re-read csv
_current = {}


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return {"error": "only csv files supported"}

    contents = await file.read()
    filepath = os.path.join("uploads", file.filename)
    with open(filepath, "wb") as f:
        f.write(contents)

    df = pd.read_csv(filepath)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) < 1:
        return {"error": "dataset needs at least 1 numeric column"}

    # had issues with numpy types not being json serializable so converting manually
    preview = json.loads(df.head(10).to_json(orient="records"))

    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "mean": round(float(df[col].mean()), 2),
            "std": round(float(df[col].std()), 2),
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2)
        }

    _current["df"] = df
    _current["filename"] = file.filename
    _current["numeric_cols"] = numeric_cols

    return {
        "filename": file.filename,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": numeric_cols,
        "all_columns": df.columns.tolist(),
        "preview": preview,
        "stats": stats
    }


class DetectRequest(BaseModel):
    columns: list[str]
    contamination: float = 0.05


@app.post("/detect")
def detect(req: DetectRequest):
    if "df" not in _current:
        return {"error": "upload a dataset first"}

    df = _current["df"]
    for col in req.columns:
        if col not in df.columns:
            return {"error": f"column '{col}' not found"}

    results = run_detection(df, req.columns, req.contamination)
    if results.get("error"):
        return {"error": results["error"]}

    # add ai explanation to results before sending back
    results["explanation"] = explain_anomalies(results["summary"], req.columns)

    save_scan(
        filename=_current["filename"],
        columns=", ".join(req.columns),
        total_rows=results["summary"]["total_rows"],
        anomalies_found=results["summary"]["total_anomalies"],
        methods_used="Isolation Forest, Autoencoder, Z-Score",
        contamination=req.contamination
    )

    return results


@app.get("/history")
def history():
    return get_history()


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
