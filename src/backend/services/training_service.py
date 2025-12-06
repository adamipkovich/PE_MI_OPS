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
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            # Metrikák számítása
            metrics = {
                "accuracy": float(accuracy_score(y, y_pred)),
                "precision": float(precision_score(y, y_pred, average='micro')),
                "f1_score": float(f1_score(y, y_pred, average='micro')),
                "recall": float(recall_score(y, y_pred, average='micro')),
                "best_params": convert_numpy_types(clf.best_params_),
                "feature_names": X.columns.tolist(),
                "n_samples": int(len(X)),
                "n_features": int(len(X.columns))
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
