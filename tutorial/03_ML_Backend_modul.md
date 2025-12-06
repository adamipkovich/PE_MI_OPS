# 03 - ML Backend Modul (Training + Prediction + FastAPI)

## Mi az ML Backend és miért van rá szükség?

### A Probléma

Egy machine learning rendszerben szét kell választani a különböző felelősségeket:

1. **Model tanítás** - Adatok feldolgozása, model training, mentés
2. **Model predikció** - Betanított model használata új adatokon
3. **API interfész** - Külső kommunikáció (frontend, más szolgáltatások)
4. **Model management** - Modellek tárolása, verziózása, metaadatok

### A Megoldás: Modularizált Backend

```
FastAPI (API Layer)
    ↓
Services (Business Logic)
    ├── TrainingService → Model tanítás
    └── PredictionService → Predikció
    ↓
Storage (Data Layer)
    └── Lokális fájlrendszer (pickle/joblib)
```

---

## Architektúra Áttekintés

### Komponensek

1. **FastAPI App** - HTTP API endpoints
2. **TrainingService** - Model tanítás logika
3. **PredictionService** - Predikció logika RabbitMQ integrációval
4. **Model Storage** - Lokális fájlok + metadata.json

### Adatfolyam

```
POST /train
    → TrainingService.train()
    → Model mentése (pickle)
    → Metadata frissítése

GET /predict/{queue}
    → RabbitMQ queue-ból adatok olvasása
    → PredictionService.predict()
    → Eredmények visszaküldése
```

---

## 📊 Data Upload & Processing Workflow

### Training Workflow - Teljes Folyamat

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING WORKFLOW                       │
└─────────────────────────────────────────────────────────────────┘

1. Frontend (Streamlit)
   │
   ├── User feltölt egy CSV fájlt (pl. cars.csv)
   │   └── File uploader: st.file_uploader()
   │
   ├── CSV mentése temp file-ba
   │   └── df.to_csv("temp_training_data.csv")
   │
   └── API hívás POST /train
       │
       ▼
2. Backend (FastAPI)
   │
   ├── Endpoint: @app.post("/train")
   │   └── Request body: {data_path, target_column, model_id?}
   │
   ├── CSV betöltése
   │   └── pd.read_csv(data_path, sep=";")
   │
   ├── Feature/Target szétválasztás
   │   ├── y = data[target_column]
   │   └── X = data.drop(target_column)
   │
   ├── TrainingService.train_decision_tree()
   │   ├── GridSearchCV futtatás
   │   ├── Best model kiválasztás
   │   └── Metrikák számítása
   │
   └── ModelStorage.save_model()
       ├── Model → pickle file
       └── Metadata → metadata.json
       │
       ▼
3. Response
   │
   └── JSON vissza a frontend-nek
       ├── model_id
       ├── accuracy, precision, f1, recall
       └── best_params
```

**Példa Request:**
```json
POST http://localhost:8000/train
Content-Type: application/json

{
  "data_path": "data/cars.csv",
  "target_column": "Origin",
  "model_id": "my_awesome_model"
}
```

**Példa Response:**
```json
{
  "model_id": "my_awesome_model",
  "success": true,
  "accuracy": 0.8765,
  "precision": 0.8723,
  "f1_score": 0.8744,
  "recall": 0.8765,
  "best_params": {
    "criterion": "gini",
    "ccp_alpha": 0.01,
    "max_depth": 5
  },
  "feature_names": ["MPG", "Cylinders", "Displacement", ...],
  "n_samples": 398,
  "n_features": 8
}
```

---

### Prediction Workflow - Teljes Folyamat

```
┌─────────────────────────────────────────────────────────────────┐
│                   PREDICTION WORKFLOW                            │
└─────────────────────────────────────────────────────────────────┘

1. Frontend (Streamlit)
   │
   ├── User feltölt CSV-t predikáláshoz
   │   └── File uploader: st.file_uploader()
   │
   ├── RabbitMQ Producer létrehozás
   │   ├── conn = RabbitMQConnection(host="rabbitmq")
   │   └── producer = Producer(conn)
   │
   ├── Adatok küldése queue-ba
   │   └── producer.send_message("predictions", {"data": df.to_dict()})
   │
   └── API hívás GET /predict/{queue_name}
       │
       ▼
2. Backend (FastAPI)
   │
   ├── Endpoint: @app.get("/predict/predictions")
   │
   ├── Ellenőrzés: Van betöltött model?
   │   └── if prediction_service.current_model is None → ERROR
   │
   ├── PredictionService.predict_from_queue()
   │   │
   │   ├── Consumer.get_message("predictions")
   │   │   └── RabbitMQ queue-ból olvas
   │   │
   │   ├── JSON → DataFrame konverzió
   │   │   └── pd.DataFrame.from_dict(message['data'])
   │   │
   │   ├── Feature validáció
   │   │   └── Megvannak-e a szükséges oszlopok?
   │   │
   │   └── Model.predict(X)
   │       └── Predictions hozzáadása DataFrame-hez
   │
   └── Response: {"predictions": [...]}
       │
       ▼
3. Frontend (Streamlit)
   │
   ├── Eredmények megjelenítése
   │   ├── Táblázat: st.dataframe()
   │   ├── Metrikák: accuracy, precision, f1
   │   └── Confusion Matrix: matplotlib
   │
   └── ✅ Kész!
```

**Lépésről lépésre példa:**

**1. Model betöltése:**
```bash
curl -X POST "http://localhost:8000/models/my_awesome_model/load"
```

**2. Adatok RabbitMQ-ba küldése (Python):**
```python
from app.streaming import RabbitMQConnection, Producer
import pandas as pd

# Adatok betöltése
df = pd.read_csv("data/test.csv", sep=";")

# Producer
conn = RabbitMQConnection(host="rabbitmq", port=5672)
conn.connect()
producer = Producer(conn)

# Küldés
producer.send_message("predictions", {"data": df.to_dict()})
conn.disconnect()
```

**3. Predikció kérés:**
```bash
curl "http://localhost:8000/predict/predictions"
```

**Példa Response:**
```json
{
  "predictions": [
    {
      "MPG": 18.0,
      "Cylinders": 8,
      "Displacement": 307.0,
      "Horsepower": 130.0,
      "Weight": 3504,
      "Acceleration": 12.0,
      "Year": 70,
      "prediction": "US"
    },
    {
      "MPG": 15.0,
      "Cylinders": 8,
      "Displacement": 350.0,
      "Horsepower": 165.0,
      "Weight": 3693,
      "Acceleration": 11.5,
      "Year": 70,
      "prediction": "US"
    }
  ]
}
```

---

### 🔑 Kulcs Különbségek: Training vs Prediction

| Szempont | Training | Prediction |
|----------|----------|------------|
| **Input módszer** | File path (CSV a szerveren) | RabbitMQ queue |
| **HTTP metódus** | POST | GET |
| **Adatok tárolása** | Model → pickle file | Nincs tárolás |
| **Időigény** | Hosszú (GridSearch) | Gyors |
| **Output** | Metrikák + model_id | Predictions lista |
| **Side effect** | Model file létrehozás | Nincs |

---

### 🛠️ Gyakorlati Tippek

**1. CSV formátum követelmények:**
- Elválasztó: `;` (pontosvessző)
- Encoding: UTF-8
- Header: Kötelező (első sor)
- Hiányzó értékek: Kerülendők (NaN kezelés szükséges)

**2. Feature konzisztencia:**
- Training és prediction során **ugyanazok** a feature-ök szükségesek
- Feature nevek **pontosan** megegyeznek
- Feature sorrend: nem számít (DataFrame oszlopnevek alapján)

**3. Error handling:**
```python
# Streamlit frontend example
try:
    result = api_client.train_model(...)
    if result and result.get('success'):
        st.success("✅ Sikeres!")
    else:
        st.error(f"❌ Hiba: {result.get('error')}")
except Exception as e:
    st.error(f"❌ Váratlan hiba: {e}")
```

---

## SOLID Elvek Alkalmazása

### 1. **Single Responsibility Principle (SRP)**

Minden service egy felelősséggel:
- `TrainingService` → csak tanítás
- `PredictionService` → csak predikció
- `ModelStorage` → csak mentés/betöltés

### 2. **Dependency Inversion Principle (DIP)**

Service-ek függnek absztrakciótól:
```python
class PredictionService:
    def __init__(self, consumer: Consumer):  # Absztrakció!
        self.consumer = consumer
```

### 3. **Interface Segregation Principle (ISP)**

Külön interfészek különböző funkciókhoz:
```python
class Trainable(ABC):
    @abstractmethod
    def train(self, X, y): pass

class Predictable(ABC):
    @abstractmethod
    def predict(self, X): pass
```

---

## Implementáció Lépésről Lépésre

### Lépés 1: Model Storage (Mentés/Betöltés)

**Fájl:** `app/backend/services/model_storage.py`

**Miért csináljuk?**
- Modellek perzisztens tárolása
- Metadata kezelés (accuracy, feature names, stb.)
- Verziózás támogatása

```python
import json
import joblib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelStorage:
    """
    Model tárolás és betöltés kezelése.
    
    SOLID elvek:
    - SRP: Csak model I/O műveletek
    """
    
    def __init__(self, storage_dir: str = "backend/models/saved"):
        """
        ModelStorage inicializálása.
        
        Args:
            storage_dir: Modellek tárolási mappája
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_dir / "metadata.json"
        
        # Metadata inicializálása
        if not self.metadata_file.exists():
            self._save_metadata({"models": []})
    
    def _load_metadata(self) -> Dict:
        """Metadata betöltése."""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Metadata betöltés sikertelen: {e}")
            return {"models": []}
    
    def _save_metadata(self, metadata: Dict) -> bool:
        """Metadata mentése."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"❌ Metadata mentés sikertelen: {e}")
            return False
    
    def save_model(
        self, 
        model: Any, 
        model_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Model mentése metadata-val.
        
        Args:
            model: A mentendő model
            model_id: Egyedi model azonosító
            metadata: Model metaadatok (accuracy, features, stb.)
            
        Returns:
            True ha sikeres
        """
        try:
            # Model fájl mentése
            model_path = self.storage_dir / f"{model_id}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"💾 Model mentve: {model_path}")
            
            # Metadata frissítése
            all_metadata = self._load_metadata()
            model_info = {
                "id": model_id,
                "path": str(model_path),
                "created_at": datetime.now().isoformat(),
                **metadata
            }
            all_metadata["models"].append(model_info)
            self._save_metadata(all_metadata)
            
            logger.info(f"✅ Model és metadata mentve: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model mentés sikertelen: {e}")
            return False
    
    def load_model(self, model_id: str) -> Optional[Any]:
        """
        Model betöltése ID alapján.
        
        Args:
            model_id: Model azonosító
            
        Returns:
            A betöltött model vagy None
        """
        try:
            model_path = self.storage_dir / f"{model_id}.pkl"
            if not model_path.exists():
                logger.error(f"❌ Model nem található: {model_id}")
                return None
            
            model = joblib.load(model_path)
            logger.info(f"✅ Model betöltve: {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Model betöltés sikertelen: {e}")
            return None
    
    def get_model_metadata(self, model_id: str) -> Optional[Dict]:
        """Model metadata lekérése."""
        metadata = self._load_metadata()
        for model_info in metadata["models"]:
            if model_info["id"] == model_id:
                return model_info
        return None
    
    def list_models(self) -> List[Dict]:
        """Összes model listázása."""
        metadata = self._load_metadata()
        return metadata.get("models", [])
```

---

### Lépés 2: Training Service

**Fájl:** `app/backend/services/training_service.py`

**Miért csináljuk?**
- Model tanítás üzleti logikája
- GridSearch / hyperparameter tuning
- Automatikus model mentés

```python
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from .model_storage import ModelStorage

logger = logging.getLogger(__name__)

class TrainingService:
    """
    Model training service.
    
    SOLID elvek:
    - SRP: Csak model tanítás
    - DIP: Függ ModelStorage absztrakciótól
    """
    
    def __init__(self, storage: ModelStorage):
        """
        TrainingService inicializálása.
        
        Args:
            storage: Model storage instance (dependency injection)
        """
        self.storage = storage
    
    def train_decision_tree(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_id: Optional[str] = None,
        hyperparams: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Decision Tree tanítása GridSearch-el.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_id: Model azonosító (ha None, generálódik)
            hyperparams: Hyperparaméterek GridSearch-hoz
            
        Returns:
            Dictionary eredményekkel (model_id, score, stb.)
        """
        try:
            logger.info("🚀 Model tanítás indítása...")
            
            # Model ID generálása ha nincs
            if model_id is None:
                from datetime import datetime
                model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Default hyperparaméterek
            if hyperparams is None:
                hyperparams = {
                    'criterion': ['gini', 'entropy'],
                    'ccp_alpha': [0.01, 0.001],
                    'max_depth': np.arange(2, 8, 1)
                }
            
            # GridSearch tanítás
            clf = GridSearchCV(
                DecisionTreeClassifier(random_state=42),
                hyperparams,
                cv=5,
                scoring='accuracy'
            )
            clf.fit(X, y)
            
            # Legjobb model
            best_model = clf.best_estimator_
            best_score = clf.best_score_
            
            # Predikciók
            y_pred = best_model.predict(X)
            
            # Metrikák számítása
            metrics = {
                "accuracy": float(accuracy_score(y, y_pred)),
                "precision": float(precision_score(y, y_pred, average='micro')),
                "f1_score": float(f1_score(y, y_pred, average='micro')),
                "recall": float(recall_score(y, y_pred, average='micro')),
                "best_params": clf.best_params_,
                "feature_names": X.columns.tolist(),
                "n_samples": len(X),
                "n_features": len(X.columns)
            }
            
            # Model mentése
            self.storage.save_model(
                model=best_model,
                model_id=model_id,
                metadata=metrics
            )
            
            logger.info(f"✅ Model tanítás sikeres! Accuracy: {metrics['accuracy']:.4f}")
            
            return {
                "model_id": model_id,
                "success": True,
                **metrics
            }
            
        except Exception as e:
            logger.error(f"❌ Model tanítás sikertelen: {e}")
            return {
                "success": False,
                "error": str(e)
            }
```

---

### Lépés 3: Prediction Service

**Fájl:** `app/backend/services/prediction_service.py`

**Miért csináljuk?**
- Predikció logika
- RabbitMQ integráció
- Model cache kezelés

```python
import logging
import pandas as pd
from typing import Dict, Any, Optional
from .model_storage import ModelStorage
import sys
sys.path.append('..')
from app.streaming import Consumer

logger = logging.getLogger(__name__)

class PredictionService:
    """
    Prediction service RabbitMQ integrációval.
    
    SOLID elvek:
    - SRP: Csak predikció
    - DIP: Függ ModelStorage és Consumer absztrakciótól
    """
    
    def __init__(self, storage: ModelStorage, consumer: Optional[Consumer] = None):
        """
        PredictionService inicializálása.
        
        Args:
            storage: Model storage instance
            consumer: RabbitMQ consumer instance (opcionális)
        """
        self.storage = storage
        self.consumer = consumer
        self.current_model = None
        self.current_model_id = None
        self.current_features = None
    
    def load_model(self, model_id: str) -> bool:
        """
        Model betöltése a cache-be.
        
        Args:
            model_id: Model azonosító
            
        Returns:
            True ha sikeres
        """
        try:
            # Model betöltése
            model = self.storage.load_model(model_id)
            if model is None:
                return False
            
            # Metadata betöltése
            metadata = self.storage.get_model_metadata(model_id)
            if metadata is None:
                logger.error(f"❌ Metadata nem található: {model_id}")
                return False
            
            # Cache frissítése
            self.current_model = model
            self.current_model_id = model_id
            self.current_features = metadata.get("feature_names", [])
            
            logger.info(f"✅ Model betöltve cache-be: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model betöltés sikertelen: {e}")
            return False
    
    def predict(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Predikció a betöltött modellel.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame eredményekkel vagy None
        """
        try:
            if self.current_model is None:
                logger.error("❌ Nincs betöltött model!")
                return None
            
            # Feature validáció
            if self.current_features:
                if not all(f in data.columns for f in self.current_features):
                    logger.error(f"❌ Hiányzó feature-ök: {self.current_features}")
                    return None
                
                # Feature-ök kiválasztása
                X = data[self.current_features]
            else:
                X = data
            
            # Predikció
            predictions = self.current_model.predict(X)
            
            # Eredmények DataFrame-be
            result = data.copy()
            result['prediction'] = predictions
            
            logger.info(f"✅ Predikció sikeres: {len(predictions)} minta")
            return result
            
        except Exception as e:
            logger.error(f"❌ Predikció sikertelen: {e}")
            return None
    
    def predict_from_queue(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """
        Predikció RabbitMQ queue-ból.
        
        Args:
            queue_name: Queue neve
            
        Returns:
            Dictionary eredményekkel vagy None
        """
        try:
            if self.consumer is None:
                logger.error("❌ Consumer nincs konfigurálva!")
                return None
            
            # Üzenet fogadása
            message = self.consumer.get_message(queue_name)
            if message is None:
                return None
            
            # DataFrame konvertálás
            if 'data' in message:
                df = pd.DataFrame.from_dict(message['data'])
            else:
                df = pd.DataFrame.from_dict(message)
            
            # Predikció
            result_df = self.predict(df)
            if result_df is None:
                return None
            
            # JSON konvertálás
            return result_df.to_dict(orient='records')
            
        except Exception as e:
            logger.error(f"❌ Queue predikció sikertelen: {e}")
            return None
```

---

### Lépés 4: FastAPI Endpoints

**Fájl:** `backend/main.py`

**Miért csináljuk?**
- RESTful API interfész
- Dependency injection FastAPI-val
- Lifespan management (kapcsolatok)

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import sys
sys.path.append('..')

from app.streaming import RabbitMQConnection, Consumer
from services.model_storage import ModelStorage
from services.training_service import TrainingService
from services.prediction_service import PredictionService

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
    
    # Startup: RabbitMQ kapcsolat
    rabbit_connection = RabbitMQConnection(host="rabbitmq", port=5672)
    if rabbit_connection.connect():
        consumer = Consumer(rabbit_connection)
        prediction_service = PredictionService(storage, consumer)
    
    yield
    
    # Shutdown: Kapcsolat lezárása
    if rabbit_connection:
        rabbit_connection.disconnect()

app = FastAPI(
    title="ML Backend API",
    description="Machine Learning Backend SOLID-dal",
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

@app.get("/models/current")
async def get_current_model():
    """Aktuális betöltött model."""
    if prediction_service is None or prediction_service.current_model_id is None:
        return {"current_model": None}
    
    return {
        "current_model": prediction_service.current_model_id,
        "features": prediction_service.current_features
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
```

---

### Lépés 5: __init__.py fájlok

**`backend/__init__.py`**
```python
# Üres fájl a package-hoz
```

**`backend/services/__init__.py`**
```python
from .model_storage import ModelStorage
from .training_service import TrainingService
from .prediction_service import PredictionService

__all__ = ['ModelStorage', 'TrainingService', 'PredictionService']
```

---

## SOLID Elvek Összefoglalása

### ✅ Mit értünk el?

1. **Single Responsibility (SRP)**
   - `ModelStorage` → csak I/O
   - `TrainingService` → csak tanítás
   - `PredictionService` → csak predikció
   - `main.py` → csak API routing

2. **Dependency Inversion (DIP)**
   - Service-ek dependency injection-nel kapják a függőségeket
   - Könnyű tesztelés mock objektumokkal

3. **Interface Segregation (ISP)**
   - Kis, specifikus interfészek
   - Service-ek csak azt használják, amire szükségük van

---

## Tesztelés

### 1. Model Tanítás

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "data/cars.csv",
    "target_column": "Origin"
  }'
```

### 2. Modellek Listázása

```bash
curl "http://localhost:8000/models"
```

### 3. Model Betöltése

```bash
curl -X POST "http://localhost:8000/models/model_20251206_100000/load"
```

### 4. Predikció (RabbitMQ-n keresztül)

Python script:
```python
from app.streaming import RabbitMQConnection, Producer
import pandas as pd

# Producer
conn = RabbitMQConnection()
conn.connect()
producer = Producer(conn)

# Adatok küldése
data = pd.read_csv("data/cars.csv", sep=";")
producer.send_message("predictions", {"data": data.to_dict()})
conn.disconnect()
```

Majd:
```bash
curl "http://localhost:8000/predict/predictions"
```

---

## Következő Lépés

Most, hogy van egy tiszta, SOLID alapú ML backend modulunk, a következő tutorialban létrehozzuk a **Frontend modult (Streamlit)**, ahol:
- User-friendly UI-t építünk
- RabbitMQ-n keresztül kommunikálunk
- Eredményeket vizualizáljuk

👉 Folytasd a `04_Frontend_modul.md` fájllal!
