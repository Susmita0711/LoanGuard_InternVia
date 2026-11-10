"""main.py
-----------------
FastAPI application entry point. Includes CORS configuration and route registration.
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router

app = FastAPI(title="LoanGuard API", version="1.0.0")

# Allow only the Streamlit frontend origin (configured via env or default localhost:8502)
frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:8502")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

# Optional health endpoint at root for quick checks
@app.get("/health", include_in_schema=False)
def root_health():
    return {"status": "healthy"}

