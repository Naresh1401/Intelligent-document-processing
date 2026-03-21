#!/bin/bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
sleep 3
streamlit run ui/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
