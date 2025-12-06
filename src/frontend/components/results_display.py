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
