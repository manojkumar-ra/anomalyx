![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)

# AnomalyX

Multi-method anomaly detection system that combines **Isolation Forest**, **Autoencoder** (PyTorch), and **Z-Score** analysis to detect outliers in any CSV dataset. Features a weighted voting system, interactive visualizations, and AI-powered explanations.

## Features

- Upload any CSV dataset with numeric columns
- Three detection methods running simultaneously
  - **Isolation Forest** — tree-based outlier isolation
  - **Autoencoder** — neural network reconstruction error
  - **Z-Score** — statistical deviation analysis
- Weighted voting system (anomaly if 2+ methods agree)
- Combined anomaly scoring with customizable contamination rate
- Interactive scatter plot and score distribution charts
- AI-powered explanation of results using Groq (Llama 3.3 70B)
- Scan history stored in MySQL

## Screenshots

### Upload & Dataset Overview
![Dataset Analysis](screenshots/dataset_analysis.png)

### Detection Results
![Results](screenshots/detection_results.png)

### AI Analysis
![AI Analysis](screenshots/ai_analysis.png)

## Tech Stack

- **Backend** — FastAPI, Python
- **ML** — scikit-learn (Isolation Forest), PyTorch (Autoencoder), SciPy (Z-Score)
- **AI** — Groq API with Llama 3.3 70B
- **Database** — MySQL
- **Frontend** — HTML/CSS/JS, Chart.js
- **Data** — Pandas, NumPy

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=anomalyx
```

Run the server:
```bash
python main.py
```

Open `http://localhost:8004`
