# 🛡️ Real-Life Violence Detection — FastAPI + Streamlit

```
violence-detection-app/
├── backend/            FastAPI + Pydantic service
│   ├── main.py         endpoints  (/predict, /health, /docs)
│   ├── schemas.py      Pydantic request/response models
│   ├── video_utils.py  frame extraction (same as training notebook)
│   ├── model_service.py loads the .h5 model once
│   └── config.py       image size, sequence length, class names, limits
├── frontend/
│   ├── app.py          Streamlit UI
│   └── .streamlit/config.toml
├── models/             Real_Life_Violence_Detation_MoBiLSTM_model.h5
├── requirements-backend.txt
└── requirements-frontend.txt
```

## Run

```bash
# 1) Backend  (terminal 1)
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements-backend.txt -r requirements-frontend.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 2) Frontend (terminal 2)
streamlit run frontend/app.py
```

* API docs (Swagger): http://localhost:8000/docs
* Streamlit UI:       http://localhost:8501
* Health check:       http://localhost:8000/health

## Test the API without the UI

```bash
curl -X POST "http://localhost:8000/predict?threshold=0.5" -F "file=@my_video.mp4"
```

## Configuration (environment variables)

| Variable        | Default                      | Meaning                          |
|-----------------|------------------------------|----------------------------------|
| `MODEL_PATH`    | `models/…MoBiLSTM_model.h5`  | Path to the model file           |
| `MAX_UPLOAD_MB` | `200`                        | Max upload size                  |
| `MAX_SEGMENTS`  | `8`                          | Max timeline segments            |
| `API_URL`       | `http://localhost:8000`      | Backend URL used by Streamlit    |

## Troubleshooting

* **Model fails to load** – `/health` shows `model_not_loaded` and the server log has the reason.
  Use a TensorFlow version close to the one you trained with on Kaggle
  (check with `import tensorflow as tf; print(tf.__version__)` in your notebook).
* **Video won't open** – re-encode it: `ffmpeg -i in.mov -c:v libx264 out.mp4`.
