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
