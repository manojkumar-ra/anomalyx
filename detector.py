import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from scipy import stats


# simple autoencoder for learning normal patterns
# if it cant reconstruct a datapoint well, that point is probably an anomaly
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden = max(input_dim // 2, 2)
        bottleneck = max(hidden // 2, 1)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, bottleneck),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(data, epochs=50):
    model = Autoencoder(data.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    tensor_data = torch.FloatTensor(data)

    # train it to reconstruct normal data
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(tensor_data), tensor_data)
        loss.backward()
        optimizer.step()

    # reconstruction error = how "weird" each row is
    model.eval()
    with torch.no_grad():
        reconstructed = model(tensor_data)
        errors = torch.mean((tensor_data - reconstructed) ** 2, dim=1).numpy()

    return errors


def zscore_detection(data, threshold=3.0):
    z_scores = np.abs(stats.zscore(data, axis=0))
    is_anomaly = np.any(z_scores > threshold, axis=1)
    max_z = np.max(z_scores, axis=1)
    return is_anomaly, max_z


def run_detection(df, columns, contamination=0.05):
    try:
        data = df[columns].copy().dropna()
        if len(data) < 10:
            return {"error": "need at least 10 rows of data"}

        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        # --- isolation forest ---
        iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        iso_labels = iso.fit_predict(scaled)
        iso_scores = iso.decision_function(scaled)
        iso_anomalies = iso_labels == -1  # -1 means anomaly in sklearn

        # --- autoencoder ---
        ae_errors = train_autoencoder(scaled)
        ae_threshold = np.percentile(ae_errors, (1 - contamination) * 100)
        ae_anomalies = ae_errors > ae_threshold

        # --- z-score ---
        z_anomalies, z_scores = zscore_detection(scaled)

        # voting system - flag as anomaly if at least 2 out of 3 methods agree
        votes = iso_anomalies.astype(int) + ae_anomalies.astype(int) + z_anomalies.astype(int)
        final_anomalies = votes >= 2

        # normalize all scores to 0-1 range then combine with weights
        iso_norm = 1 - (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-10)
        ae_norm = (ae_errors - ae_errors.min()) / (ae_errors.max() - ae_errors.min() + 1e-10)
        z_norm = (z_scores - z_scores.min()) / (z_scores.max() - z_scores.min() + 1e-10)

        # weighted combination - isolation forest gets most weight since its usually most reliable
        combined_score = (iso_norm * 0.4 + ae_norm * 0.35 + z_norm * 0.25)

        # put everything into a dataframe for easy sorting
        result_df = data.reset_index(drop=True).copy()
        result_df["anomaly_score"] = np.round(combined_score, 4)
        result_df["is_anomaly"] = final_anomalies
        result_df["iso_forest"] = iso_anomalies
        result_df["autoencoder"] = ae_anomalies
        result_df["zscore"] = z_anomalies
        result_df["votes"] = votes

        anomaly_rows = result_df[result_df["is_anomaly"]].sort_values("anomaly_score", ascending=False)

        # scatter plot data - using first 2 selected columns as x and y
        scatter_data = []
        for _, row in result_df.iterrows():
            scatter_data.append({
                "x": round(float(row[columns[0]]), 2),
                "y": round(float(row[columns[1]]) if len(columns) > 1 else float(row[columns[0]]), 2),
                "is_anomaly": bool(row["is_anomaly"]),
                "score": round(float(row["anomaly_score"]), 4)
            })

        method_results = {
            "isolation_forest": {"anomalies": int(iso_anomalies.sum()), "description": "tree-based isolation of outlier points"},
            "autoencoder": {"anomalies": int(ae_anomalies.sum()), "description": "neural network reconstruction error"},
            "zscore": {"anomalies": int(z_anomalies.sum()), "description": "statistical deviation from mean"}
        }

        # histogram bins for score distribution chart
        score_bins = np.histogram(combined_score, bins=20)

        summary = {
            "total_rows": len(result_df),
            "total_anomalies": int(final_anomalies.sum()),
            "anomaly_percentage": round(final_anomalies.sum() / len(result_df) * 100, 2),
            "avg_anomaly_score": round(float(combined_score[final_anomalies].mean()), 4) if final_anomalies.sum() > 0 else 0,
            "methods": method_results
        }

        # top anomalies with which methods flagged them
        top_anomalies = []
        for _, row in anomaly_rows.head(10).iterrows():
            entry = {col: round(float(row[col]), 2) for col in columns}
            entry["anomaly_score"] = round(float(row["anomaly_score"]), 4)
            entry["methods_flagged"] = []
            if row["iso_forest"]: entry["methods_flagged"].append("Isolation Forest")
            if row["autoencoder"]: entry["methods_flagged"].append("Autoencoder")
            if row["zscore"]: entry["methods_flagged"].append("Z-Score")
            top_anomalies.append(entry)

        return {
            "summary": summary,
            "top_anomalies": top_anomalies,
            "scatter_data": scatter_data,
            "score_distribution": {
                "counts": score_bins[0].tolist(),
                "edges": [round(e, 3) for e in score_bins[1].tolist()]
            }
        }

    except Exception as e:
        return {"error": f"detection failed: {str(e)}"}
