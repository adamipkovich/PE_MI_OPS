from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import logging
import sys
import os

# Add project root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if APP_ROOT not in sys.path:
    sys.path.append(APP_ROOT)

from streaming import RabbitMQConnection, Consumer
from services.model_storage import ModelStorage
from services.training_service import TrainingService
from services.prediction_service import PredictionService

# Logging beállítása
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
storage = ModelStorage()
training_service = TrainingService(storage)
prediction_service = None
rabbit_connection = None
consumer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan management."""
    global rabbit_connection, consumer, prediction_service
    
    logger.info("🚀 Backend indítása...")
    
    # Startup: RabbitMQ kapcsolat
    rabbit_connection = RabbitMQConnection(host="rabbitmq", port=5672)
    if rabbit_connection.connect():
        consumer = Consumer(rabbit_connection)
        prediction_service = PredictionService(storage, consumer)
        logger.info("✅ RabbitMQ kapcsolat létrehozva")
    else:
        logger.warning("⚠️ RabbitMQ kapcsolat sikertelen - prediction service korlátozott")
        prediction_service = PredictionService(storage, None)
    
    yield
    
    # Shutdown: Kapcsolat lezárása
    if rabbit_connection:
        rabbit_connection.disconnect()
    logger.info("👋 Backend leállítva")

app = FastAPI(
    title="ML Backend API",
    description="Machine Learning Backend SOLID elvekkel",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic modellek
class TrainRequest(BaseModel):
    data_path: str
    target_column: str
    model_id: Optional[str] = None

# ===== ENDPOINTS =====

@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "healthy",
        "service": "ML Backend",
        "version": "1.0.0"
    }

@app.get("/models")
async def list_models():
    """Összes model listázása."""
    models = storage.list_models()
    return {"models": models, "count": len(models)}

@app.get("/models/current")
async def get_current_model():
    """Aktuális betöltött model."""
    if prediction_service is None or prediction_service.current_model_id is None:
        return {"current_model": None}
    
    return {
        "current_model": prediction_service.current_model_id,
        "features": prediction_service.current_features
    }

@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Model információk lekérése."""
    metadata = storage.get_model_metadata(model_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Model nem található")
    return metadata

@app.post("/models/{model_id}/load")
async def load_model(model_id: str):
    """Model betöltése a prediction service-be."""
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Prediction service nem elérhető")
    
    success = prediction_service.load_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model betöltés sikertelen")
    
    return {
        "message": f"Model betöltve: {model_id}",
        "current_model": model_id
    }

@app.post("/train")
async def train_model(request: TrainRequest):
    """
    Model tanítás.
    
    Body:
    {
        "data_path": "data/cars.csv",
        "target_column": "Origin",
        "model_id": "optional_custom_id"
    }
    """
    try:
        # Adatok betöltése
        data = pd.read_csv(request.data_path, sep=";")
        
        # Feature és target szétválasztás
        y = data[request.target_column]
        X = data.drop(columns=[request.target_column, "Car"] if "Car" in data.columns else [request.target_column])
        
        # Tanítás
        result = training_service.train_decision_tree(X, y, model_id=request.model_id)
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        return result
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Adat fájl nem található")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/{queue_name}")
async def predict_from_queue(queue_name: str):
    """
    Predikció RabbitMQ queue-ból.
    
    Először küldd el az adatokat a queue-ba, majd hívd meg ezt az endpointot.
    """
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Prediction service nem elérhető")
    
    if prediction_service.current_model is None:
        raise HTTPException(status_code=400, detail="Nincs betöltött model! Használd a /models/{id}/load endpointot.")
    
    result = prediction_service.predict_from_queue(queue_name)
    if result is None:
        raise HTTPException(status_code=404, detail="Nincs üzenet a queue-ban vagy hiba történt")
    
    return {"predictions": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
