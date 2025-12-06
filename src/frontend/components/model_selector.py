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
