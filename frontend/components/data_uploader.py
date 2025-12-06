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
