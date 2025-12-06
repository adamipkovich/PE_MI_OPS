# 04 - Frontend Modul (Streamlit)

## Mi a Frontend és miért van rá szükség?

### A Probléma

A backend API önmagában nem felhasználóbarát:
1. **Technikai knowledge** - curl parancsok, HTTP kérések ismerete szükséges
2. **Vizualizáció hiánya** - Nincs szemléletes megjelenítés
3. **Workflow bonyolultsága** - Több lépés manuális végrehajtása
4. **Eredmények értelmezése** - JSON válaszok nehezen olvashatók

### A Megoldás: Streamlit UI

**Streamlit** egy Python-based framework interaktív web applikációk gyors fejlesztésére.

```
Streamlit UI (Frontend)
    ↓
Backend API (FastAPI)
    ↓
RabbitMQ (Message Queue)
    ↓
ML Backend (Training/Prediction)
```

---

## Architektúra Áttekintés

### Komponensek

1. **UI Komponensek** - Vizuális elemek (gombok, input mezők, grafikonok)
2. **API Client** - Backend kommunikáció
3. **RabbitMQ Producer** - Adatok küldése queue-ba
4. **Visualization** - Eredmények megjelenítése (táblázatok, metrikák, confusion matrix)

### User Workflow

```
1. Model Training
   → CSV feltöltés
   → Target column kiválasztás
   → Train button
   → Eredmények megjelenítése

2. Prediction
   → Model kiválasztás
   → CSV feltöltés predikáláshoz
   → Adatok küldése RabbitMQ-ba
   → Predikció kérés
   → Eredmények vizualizálása
```

---

## SOLID Elvek Alkalmazása

### 1. **Single Responsibility Principle (SRP)**

Minden komponens egy felelősséggel:
```python
class ModelSelector:
    def render(self): ...  # Csak model választó UI

class DataUploader:
    def render(self): ...  # Csak adatok feltöltése

class ResultsDisplay:
    def render(self): ...  # Csak eredmények megjelenítése
```

### 2. **Dependency Inversion Principle (DIP)**

UI komponensek függnek API client absztrakciótól:
```python
class APIClient:
    def get_models(self): ...
    def train_model(self): ...
    def predict(self): ...
```

---

## Implementáció Lépésről Lépésre

### Lépés 1: API Client

**Fájl:** `app/frontend/utils/api_client.py`

**Miért csináljuk?**
- Központi API kommunikáció
- Error handling
- Type-safe interface

```python
import requests
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class APIClient:
    """
    Backend API kommunikáció.
    
    SOLID elvek:
    - SRP: Csak API hívások
    """
    
    def __init__(self, base_url: str = "http://backend:8000"):
        """
        API Client inicializálása.
        
        Args:
            base_url: Backend API URL
        """
        self.base_url = base_url
    
    def get_models(self) -> Optional[List[Dict[str, Any]]]:
        """Modellek listázása."""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            logger.error(f"❌ Model listázás sikertelen: {e}")
            return None
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Model információk lekérése."""
        try:
            response = requests.get(f"{self.base_url}/models/{model_id}", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Model info lekérés sikertelen: {e}")
            return None
    
    def load_model(self, model_id: str) -> bool:
        """Model betöltése."""
        try:
            response = requests.post(f"{self.base_url}/models/{model_id}/load", timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"❌ Model betöltés sikertelen: {e}")
            return False
    
    def get_current_model(self) -> Optional[Dict[str, Any]]:
        """Aktuális model lekérése."""
        try:
            response = requests.get(f"{self.base_url}/models/current", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Current model lekérés sikertelen: {e}")
            return None
    
    def train_model(self, data_path: str, target_column: str, model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Model tanítás."""
        try:
            payload = {
                "data_path": data_path,
                "target_column": target_column
            }
            if model_id:
                payload["model_id"] = model_id
            
            response = requests.post(f"{self.base_url}/train", json=payload, timeout=300)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Model tanítás sikertelen: {e}")
            return None
    
    def predict(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """Predikció RabbitMQ queue-ból."""
        try:
            response = requests.get(f"{self.base_url}/predict/{queue_name}", timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Predikció sikertelen: {e}")
            return None
```

---

### Lépés 2: UI Komponensek

**Fájl:** `app/frontend/components/model_selector.py`

```python
import streamlit as st
from typing import List, Dict, Any, Optional

class ModelSelector:
    """
    Model választó komponens.
    
    SOLID elvek:
    - SRP: Csak model választás UI
    """
    
    @staticmethod
    def render(models: List[Dict[str, Any]]) -> Optional[str]:
        """
        Model választó megjelenítése.
        
        Args:
            models: Elérhető modellek listája
            
        Returns:
            Választott model ID vagy None
        """
        if not models:
            st.warning("⚠️ Nincsenek elérhető modellek!")
            return None
        
        st.subheader("📦 Model Kiválasztása")
        
        # Model opciók
        model_options = {
            f"{m['id']} (Accuracy: {m.get('accuracy', 'N/A'):.2%})": m['id']
            for m in models
        }
        
        selected = st.selectbox(
            "Válassz modelt:",
            options=list(model_options.keys())
        )
        
        return model_options[selected] if selected else None
```

**Fájl:** `app/frontend/components/data_uploader.py`

```python
import streamlit as st
import pandas as pd
from typing import Optional

class DataUploader:
    """
    Adat feltöltő komponens.
    
    SOLID elvek:
    - SRP: Csak adatok feltöltése
    """
    
    @staticmethod
    def render(label: str = "CSV fájl feltöltése") -> Optional[pd.DataFrame]:
        """
        Adat feltöltő megjelenítése.
        
        Args:
            label: Input field label
            
        Returns:
            Feltöltött DataFrame vagy None
        """
        st.subheader("📁 " + label)
        
        uploaded_file = st.file_uploader(
            "Válassz CSV fájlt:",
            type=['csv'],
            help="CSV fájl ; elválasztóval"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, sep=";")
                st.success(f"✅ Fájl betöltve: {len(df)} sor, {len(df.columns)} oszlop")
                
                # Előnézet
                with st.expander("📊 Adatok előnézete"):
                    st.dataframe(df.head(10))
                
                return df
                
            except Exception as e:
                st.error(f"❌ Hiba a fájl betöltésekor: {e}")
                return None
        
        return None
```

**Fájl:** `app/frontend/components/results_display.py`

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, f1_score, recall_score

class ResultsDisplay:
    """
    Eredmények megjelenítő komponens.
    
    SOLID elvek:
    - SRP: Csak eredmények vizualizálása
    """
    
    @staticmethod
    def show_training_results(results: dict):
        """Training eredmények megjelenítése."""
        st.subheader("🎯 Training Eredmények")
        
        # Metrikák
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{results.get('accuracy', 0):.2%}")
        with col2:
            st.metric("Precision", f"{results.get('precision', 0):.2%}")
        with col3:
            st.metric("F1 Score", f"{results.get('f1_score', 0):.2%}")
        with col4:
            st.metric("Recall", f"{results.get('recall', 0):.2%}")
        
        # Best params
        if 'best_params' in results:
            with st.expander("⚙️ Best Hyperparameters"):
                st.json(results['best_params'])
        
        # Model info
        st.info(f"📊 Model ID: `{results.get('model_id')}`")
        st.info(f"📈 Samples: {results.get('n_samples')}, Features: {results.get('n_features')}")
    
    @staticmethod
    def show_prediction_results(df: pd.DataFrame, original_df: pd.DataFrame = None):
        """Predikció eredmények megjelenítése."""
        st.subheader("🔮 Predikció Eredmények")
        
        # Eredmények táblázat
        st.dataframe(df)
        
        # Ha van ground truth (Origin oszlop)
        if original_df is not None and 'Origin' in original_df.columns:
            st.subheader("📊 Értékelési Metrikák")
            
            y_true = original_df['Origin']
            y_pred = df['prediction']
            
            # Metrikák
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
            with col2:
                st.metric("Precision", f"{precision_score(y_true, y_pred, average='micro'):.2%}")
            with col3:
                st.metric("F1 Score", f"{f1_score(y_true, y_pred, average='micro'):.2%}")
            with col4:
                st.metric("Recall", f"{recall_score(y_true, y_pred, average='micro'):.2%}")
            
            # Confusion Matrix
            st.subheader("🔥 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, cmap='Blues')
            st.pyplot(fig)
```

---

### Lépés 3: Main App

**Fájl:** `frontend/app.py`

```python
import streamlit as st
import pandas as pd
import sys
sys.path.append('..')

from app.streaming import RabbitMQConnection, Producer
from utils.api_client import APIClient
from components.model_selector import ModelSelector
from components.data_uploader import DataUploader
from components.results_display import ResultsDisplay

# Page config
st.set_page_config(
    page_title="ML Platform",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 Machine Learning Platform")
st.markdown("*SOLID alapelvekkel tervezett ML rendszer*")

# API Client
api_client = APIClient(base_url="http://backend:8000")

# Sidebar navigation
page = st.sidebar.radio(
    "🧭 Navigáció",
    ["🏠 Home", "🎓 Model Training", "🔮 Prediction"]
)

# ===== HOME PAGE =====
if page == "🏠 Home":
    st.header("Üdvözöllek a ML Platform-on!")
    
    st.markdown("""
    ### 🎯 Funkciók
    
    1. **Model Training** - Decision Tree modellek tanítása
    2. **Prediction** - Predikció RabbitMQ-n keresztül
    3. **Visualization** - Eredmények megjelenítése
    
    ### 🏗️ Architektúra
    
    - **Frontend**: Streamlit (SOLID: SRP)
    - **Backend**: FastAPI (SOLID: DIP, ISP)
    - **Streaming**: RabbitMQ (SOLID: DIP)
    - **ML**: Scikit-learn
    
    ### 📚 SOLID Elvek
    
    Minden modul a SOLID elvek szerint van megtervezve, ami biztosítja a:
    - ✅ Karbantarthatóságot
    - ✅ Tesztelhetőséget
    - ✅ Bővíthetőséget
    """)
    
    # Current model
    st.divider()
    st.subheader("📦 Aktuális Model")
    current = api_client.get_current_model()
    if current and current.get('current_model'):
        st.success(f"✅ Betöltött model: `{current['current_model']}`")
        st.info(f"🔧 Features: {', '.join(current.get('features', []))}")
    else:
        st.warning("⚠️ Nincs betöltött model")

# ===== TRAINING PAGE =====
elif page == "🎓 Model Training":
    st.header("🎓 Model Training")
    
    # Data upload
    df = DataUploader.render("Training adat feltöltése")
    
    if df is not None:
        # Target column selection
        st.subheader("🎯 Target Column Kiválasztása")
        target_col = st.selectbox("Válassz target oszlopot:", df.columns.tolist())
        
        # Model ID (optional)
        model_id = st.text_input("Model ID (opcionális):", placeholder="model_custom_name")
        
        # Train button
        if st.button("🚀 Model Tanítás Indítása", type="primary"):
            with st.spinner("⏳ Model tanítás folyamatban..."):
                # Save uploaded file temporarily
                temp_path = "temp_training_data.csv"
                df.to_csv(temp_path, sep=";", index=False)
                
                # Train
                result = api_client.train_model(
                    data_path=temp_path,
                    target_column=target_col,
                    model_id=model_id if model_id else None
                )
                
                if result and result.get('success'):
                    st.success("✅ Model tanítás sikeres!")
                    ResultsDisplay.show_training_results(result)
                else:
                    st.error(f"❌ Model tanítás sikertelen: {result.get('error') if result else 'Unknown error'}")

# ===== PREDICTION PAGE =====
elif page == "🔮 Prediction":
    st.header("🔮 Prediction")
    
    # Model selection
    models = api_client.get_models()
    if models:
        selected_model_id = ModelSelector.render(models)
        
        if selected_model_id:
            # Load model
            if st.button("📥 Model Betöltése"):
                with st.spinner("⏳ Model betöltése..."):
                    if api_client.load_model(selected_model_id):
                        st.success(f"✅ Model betöltve: {selected_model_id}")
                    else:
                        st.error("❌ Model betöltés sikertelen")
    
    st.divider()
    
    # Data upload for prediction
    pred_df = DataUploader.render("Predikáláshoz adat feltöltése")
    
    if pred_df is not None:
        # RabbitMQ connection
        queue_name = "predictions"
        
        if st.button("📤 Adatok Küldése és Predikció", type="primary"):
            with st.spinner("⏳ Adatok küldése RabbitMQ-ba..."):
                try:
                    # RabbitMQ producer
                    conn = RabbitMQConnection(host="rabbitmq", port=5672)
                    if conn.connect():
                        producer = Producer(conn)
                        
                        # Adatok küldése
                        producer.send_message(queue_name, {"data": pred_df.to_dict()})
                        st.success("✅ Adatok elküldve RabbitMQ-ba")
                        
                        conn.disconnect()
                        
                        # Predikció kérés
                        with st.spinner("⏳ Predikció folyamatban..."):
                            result = api_client.predict(queue_name)
                            
                            if result and 'predictions' in result:
                                result_df = pd.DataFrame(result['predictions'])
                                ResultsDisplay.show_prediction_results(result_df, pred_df)
                            else:
                                st.error("❌ Predikció sikertelen")
                    else:
                        st.error("❌ RabbitMQ kapcsolat sikertelen")
                        
                except Exception as e:
                    st.error(f"❌ Hiba történt: {e}")
```

---

### Lépés 4: __init__.py fájlok

**`frontend/__init__.py`**
```python
# Frontend package
```

**`frontend/utils/__init__.py`**
```python
from .api_client import APIClient

__all__ = ['APIClient']
```

**`frontend/components/__init__.py`**
```python
from .model_selector import ModelSelector
from .data_uploader import DataUploader
from .results_display import ResultsDisplay

__all__ = ['ModelSelector', 'DataUploader', 'ResultsDisplay']
```

---

## SOLID Elvek Összefoglalása

### ✅ Mit értünk el?

1. **Single Responsibility (SRP)**
   - `APIClient` → csak API kommunikáció
   - `ModelSelector` → csak model választás
   - `DataUploader` → csak adatok feltöltése
   - `ResultsDisplay` → csak megjelenítés

2. **Dependency Inversion (DIP)**
   - App komponensek függnek APIClient-től
   - Könnyű tesztelés mock API-val

---

## Tesztelés

### Lokális futtatás

```bash
# Terminal
streamlit run frontend/app.py
```

Böngésző: `http://localhost:8501`

---

## Következő Lépés

Most, hogy minden modul kész van, a következő tutorialban **Docker konténerizációt** végzünk:
- Dockerfile-ok minden service-hez
- docker-compose.yml orchestration
- Volume management
- Network configuration

👉 Folytasd az `05_Docker_setup.md` fájllal!
