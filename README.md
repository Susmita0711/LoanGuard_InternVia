# LoanGuard – Dynamic Loan Default Prediction Engine

## Overview

**LoanGuard** is an advanced machine‑learning powered application for assessing the risk of loan default. It combines robust data preprocessing, multiple model candidates, model selection, SHAP explainability, and a professional Streamlit dashboard front‑end backed by a FastAPI service.

- **Organization**: FutureLend Capital
- **Tagline**: Smarter Risk Assessment. Responsible Lending.
- **Tech Stack**: Python, Pandas, NumPy, Scikit‑learn, XGBoost, LightGBM, SHAP, FastAPI, Pydantic, Streamlit, Matplotlib, Seaborn, Joblib, Docker, Pytest.

## Project Structure

```
LoanGuard/
├─ app.py               # Entry point for running both services via run_local.py
├─ run_local.py         # Port detection and process manager
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ .env.example
├─ .gitignore
├─ run.bat
│
├─ data/
│   └─ README.md       # Dataset description and licensing
│
├─ models/
│   └─ .gitkeep        # Tracked for empty folder
│
├─ src/
│   ├─ download_data.py
│   ├─ preprocessing.py
│   ├─ train.py
│   ├─ evaluate.py
│   ├─ predict.py
│   ├─ risk_scoring.py
│   └─ explainability.py
│
├─ api/
│   ├─ main.py
│   ├─ schemas.py
│   └─ routes.py
│
├─ dashboard/
│   ├─ home.py
│   ├─ assessment.py
│   ├─ batch_prediction.py
│   ├─ analytics.py
│   ├─ performance.py
│   └─ explainable_ai.py
│
├─ assets/
│   └─ logo.png        # Placeholder – add your logo
│
└─ tests/
    ├─ test_preprocessing.py
    ├─ test_prediction.py
    └─ test_api.py
```

## Getting Started

1. **Create a virtual environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Download the dataset** – the first run of `run_local.py` automatically downloads the UCI Credit Card Default dataset.
3. **Train models** – `run_local.py` triggers `src/train.py` which trains Logistic Regression, Random Forest, XGBoost and LightGBM (if available) and saves the best model under `models/`.
4. **Start the application** – run `run.bat` (or `python run_local.py`). The script detects free ports (default 8502 for Streamlit, 8001 for FastAPI), starts both services, and prints the URLs.

## Usage

- **Dashboard**: Open the printed Streamlit URL in a browser.
- **API**: Swagger UI is available at `<backend_url>/docs`.

## Testing

Run the test suite with:
```powershell
pytest tests
```

## Licensing & Data

The dataset is from the UCI Machine Learning Repository and is licensed under the Creative Commons Attribution‑ShareAlike 4.0 International license. See `data/README.md` for details.

---
*This project was generated autonomously by Antigravity AI.*
