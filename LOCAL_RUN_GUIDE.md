# 🚀 Local Execution Guide

## Prerequisites

1. **Docker Desktop** running (for RabbitMQ and Backend)
2. **Python** installed with dependencies

## Quick Start

### 1. Start Docker Services

```cmd
docker-compose up -d rabbitmq backend
```

This starts:
- RabbitMQ on `localhost:5672` (management UI: `localhost:15672`)
- Backend API on `localhost:8000`

### 2. Install Frontend Dependencies

```cmd
pip install -r requirements-frontend.txt
```

Or with UV:
```cmd
uv pip install -r requirements-frontend.txt
```

### 3. Run Streamlit Frontend Locally

```cmd
streamlit run frontend/app.py
```

The app will open at: **http://localhost:8501**

## 📁 New Project Structure

```
MLOPS/
├── frontend/          ← Streamlit app (run locally)
│   ├── app.py
│   ├── components/
│   └── utils/
├── backend/           ← FastAPI (runs in Docker)
│   ├── main.py
│   └── services/
├── streaming/         ← Shared RabbitMQ code
│   ├── producer.py
│   └── consumer.py
├── app/              ← Docker-only structure
└── data/
    └── cars.csv
```

## ⚙️ Configuration

The frontend automatically detects environment:
- **Docker**: Uses `http://backend:8000`
- **Local**: Change to `http://localhost:8000` if needed (line 27 in `frontend/app.py`)

## 🔧 Troubleshooting

### Import Errors
Make sure you're running from project root:
```cmd
cd W:\MyLab\Egyetem\PE_MI_OPS\MLOPS
streamlit run frontend/app.py
```

### Backend Connection Issues
Update `frontend/app.py` line 27:
```python
api_client = APIClient(base_url="http://localhost:8000")
```

### RabbitMQ Connection
For local execution, update RabbitMQ host in `frontend/app.py` line 140:
```python
conn = RabbitMQConnection(host="localhost", port=5672)
```

## 📊 Using the App

1. **Home Page**: View system info and current loaded model
2. **Model Training**: Upload CSV, select target column, train model
3. **Prediction**: Select model, upload data, get predictions

## 🐳 Full Docker Mode

To run everything in Docker:
```cmd
docker-compose up -d
```

Access:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- RabbitMQ Management: http://localhost:15672
