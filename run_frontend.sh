#!/usr/bin/env bash
cd "$(dirname "$0")"
API_URL=${API_URL:-http://localhost:8000} streamlit run frontend/app.py
