import logging
import pandas as pd
from typing import Dict, Any, Optional
from .model_storage import ModelStorage
from src.streaming import Consumer

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
