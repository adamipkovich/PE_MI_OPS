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
