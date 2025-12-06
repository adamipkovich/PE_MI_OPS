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
