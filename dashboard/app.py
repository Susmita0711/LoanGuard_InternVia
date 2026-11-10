"""dashboard/app.py
-----------------
Streamlit entry point for LoanGuard.
Shows a sidebar navigation menu and loads the selected page.
All pages are simple modules under ``dashboard/``.
"""
import os
import streamlit as st

# Retrieve backend URL (set by docker-compose env var)
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")

st.set_page_config(page_title="LoanGuard Dashboard", layout="wide")

# Sidebar navigation
pages = {
    "Home": "home",
    "Loan Risk Assessment": "assessment",
    "Batch Prediction": "batch_prediction",
    "Risk Analytics": "analytics",
    "Model Performance": "performance",
    "Explainable AI": "explainable_ai",
    "About": "about",
}

st.sidebar.title("LoanGuard Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))
module_name = f"dashboard.{pages[selection]}"
module = __import__(module_name, fromlist=["run"])
module.run(BACKEND_URL)

