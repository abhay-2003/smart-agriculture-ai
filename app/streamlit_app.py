import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from PIL import Image
import sys

# Add root directory to sys.path so we can import config/services
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from config.settings import MODELS_DIR, OPENWEATHER_API_KEY, GEMINI_API_KEY, DISEASE_CLASSES, SOIL_DICT, CROP_DICT
from utils.validation import validate_and_clip_crop_inputs, validate_and_clip_fertilizer_inputs
from utils.image_processing import is_leaf_image, preprocess_image
from services.weather_service import get_current_weather, get_weather_forecast_trends
from services.crop_service import get_crop_recommendation
from services.fertilizer_service import get_fertilizer_recommendation
from services.disease_service import run_disease_detection
from services.chatbot_service import AgricultureChatbot

# -----------------------------
# Configuration & Custom CSS (Dark/Modern Mode)
# -----------------------------
st.set_page_config(page_title="Smart Agriculture AI", page_icon="🌱", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Global modern UI settings */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .header-banner {
        background: linear-gradient(135deg, #2e7d32, #4caf50);
        padding: 30px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .header-banner h1 {
        margin: 0;
        padding: 0;
        font-weight: 800;
        color: white;
    }
    .header-banner h2 {
        margin: 0;
        padding: 0;
        font-weight: 700;
        color: white;
    }
    .header-banner p {
        margin-top: 10px;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid rgba(128,128,128,0.2);
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Global State for Chatbot Context
# -----------------------------
if "farm_context" not in st.session_state:
    st.session_state.farm_context = {
        "location": "Unknown",
        "last_recommended_crop": "",
        "soil_n": 0,
        "soil_p": 0,
        "soil_k": 0
    }

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    crop_path = os.path.join(MODELS_DIR, "crop_model.pkl")
    fert_path = os.path.join(MODELS_DIR, "fertilizer_model.pkl")
    disease_path = os.path.join(MODELS_DIR, "plant_disease_model.h5")
    
    crop_m = joblib.load(crop_path)
    fert_m = joblib.load(fert_path)
    disease_m = tf.keras.models.load_model(disease_path, compile=False)
    return crop_m, fert_m, disease_m

try:
    crop_model, fertilizer_model, disease_model = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure models are trained and saved in the 'models' directory.")
    st.stop()

# -----------------------------
# Sidebar Navigation
# -----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=120)
    st.title("Smart Agri AI")
    st.markdown("---")
    
    options = {
        "🏠 Dashboard": "home",
        "🌱 Crop Recommendation": "crop",
        "🌾 Fertilizer Configurator": "fertilizer",
        "🔍 Plant Disease Scanner": "disease",
        "🌦️ Weather Intelligence": "weather",
        "🤖 Farm AI Assistant": "chat"
    }
    
    choice_label = st.radio("Navigation", list(options.keys()))
    choice = options[choice_label]

    st.markdown("---")
    st.markdown("### 📊 Active Farm Status")
    
    # Mini dashboard in sidebar
    if st.session_state.farm_context["location"] != "Unknown":
        st.markdown(f"📍 **{st.session_state.farm_context['location']}**")
        st.markdown(f"🌾 **Crop:** {st.session_state.farm_context['last_recommended_crop'] or 'None yet'}")
        
        # Micro NPK visual
        st.caption("Soil Nutrients (NPK)")
        col_n, col_p, col_k = st.columns(3)
        col_n.metric("N", st.session_state.farm_context["soil_n"])
        col_p.metric("P", st.session_state.farm_context["soil_p"])
        col_k.metric("K", st.session_state.farm_context["soil_k"])
    else:
        st.info("Awaiting field data input... Run a Crop Prediction to activate this panel.")

# -----------------------------
# Page Modules
# -----------------------------
if choice == "home":
    st.markdown('<div class="header-banner"><h1>Welcome to Smart Agriculture AI System 🌱</h1><p>Your unified platform for AI-driven farming insights, yield maximization, and risk mitigation.</p></div>', unsafe_allow_html=True)
    
    st.image("https://images.unsplash.com/photo-1625246333195-78d9c38ad449?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", use_column_width=True)
    
    st.markdown("### 🌟 Platform Capabilities")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><h4>🌱 Multi-Crop Prediction</h4><p>Forecast the top 3 most profitable crops for your specific location, considering predictive seasonal weather patterns and exhaustive soil profiling.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h4>🌾 Explicit Fertilizer Rx</h4><p>Input land area and soil properties to get real fertilizer prescriptions (like Urea or DAP) and precisely computed bulk quantities explaining nutrient deficiencies.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><h4>🔍 Disease Treatment AI</h4><p>Instantly diagnose infections on plant leaves and generate multi-step treatment plans, prevention tips, and fertilizer adjustments dynamically.</p></div>', unsafe_allow_html=True)

elif choice == "crop":
    st.markdown('<div class="header-banner"><h2>🌱 AI Crop Recommendation</h2><p>Data-driven crop selection based on soil nutrients and geographical weather data.</p></div>', unsafe_allow_html=True)
    
    st.write("Provide details of your land and the intended growing period.")
    
    st.markdown("#### 1. Location & Weather Planning")
    wc1, wc2 = st.columns(2)
    with wc1:
        city = st.text_input("Farm Location (City)", "New Delhi", help="We fetch historical/forecast weather data for this region.")
    with wc2:
        period_days = st.selectbox("Growing Period (Days)", [30, 60, 90, 120], index=2, help="We aggregate weather averages over this period.")
        
    st.markdown("#### 2. Soil Nutrients (NPK)")
    nc1, nc2, nc3, nc4 = st.columns(4)
    with nc1:
        n = st.number_input("Nitrogen (N)", 0, 200, 50, help="Typical values range from 0 to 150.")
    with nc2:
        p = st.number_input("Phosphorus (P)", 0, 200, 50, help="Typical values range from 0 to 150.")
    with nc3:
        k = st.number_input("Potassium (K)", 0, 200, 50, help="Typical values range from 0 to 150.")
    with nc4:
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, help="Most crops prefer pH 5.5 to 7.5.")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Predict Optimal Crop ✨", use_container_width=True):
        if not OPENWEATHER_API_KEY:
            st.error("OpenWeather API key is missing. Cannot fetch weather data.")
        else:
            with st.spinner("Fetching weather trends and computing crop probabilities..."):
                trends, err = get_weather_forecast_trends(city, OPENWEATHER_API_KEY, days=period_days)
                if err:
                    st.error(f"Failed to fetch weather data: {err}")
                else:
                    # Update session context
                    st.session_state.farm_context["location"] = city
                    st.session_state.farm_context["soil_n"] = n
                    st.session_state.farm_context["soil_p"] = p
                    st.session_state.farm_context["soil_k"] = k
                    
                    # Compute averages
                    raw_temp = sum(t["temperature"] for t in trends) / len(trends)
                    raw_hum = sum(t["humidity"] for t in trends) / len(trends)
                    raw_rain = sum(t["rainfall"] for t in trends)
                    
                    st.info(f"**Aggregated Climate for {city} over next {period_days} days:**  \n🌡️ **Avg Temp:** {raw_temp:.1f}°C | 💧 **Avg Humidity:** {raw_hum:.1f}% | 🌧️ **Total Rain Forecast:** {raw_rain:.1f}mm")
                    
                    # Validate and strictly clip out of bound values with graceful warnings
                    val_temp, val_hum, val_rain, val_ph, n_c, p_c, k_c, warnings = validate_and_clip_crop_inputs(
                        n, p, k, ph, raw_temp, raw_hum, raw_rain
                    )
                    
                    if warnings:
                        for w in warnings:
                            st.warning(f"⚠️ {w}")
                            
                    # Get Multi-crop probabilities
                    top_3_crops, importance, insight_text = get_crop_recommendation(
                        crop_model, n_c, p_c, k_c, val_temp, val_hum, val_ph, val_rain
                    )
                    
                    # Save best to session
                    best_crop = top_3_crops[0]['crop'].capitalize()
                    st.session_state.farm_context["last_recommended_crop"] = best_crop
                    
                    st.success(f"### 🏆 Primary Recommendation: **{best_crop}** ({top_3_crops[0]['probability']:.1f}%)")
                    st.markdown(f"> 🌱 **Agronomic Insight:** {insight_text}")
                    
                    st.markdown("#### 📊 Suitability Scores")
                    for crop_data in top_3_crops:
                        c_name = crop_data['crop'].capitalize()
                        c_prob = crop_data['probability']
                        
                        colA, colB = st.columns([1, 4])
                        with colA:
                            if c_name == best_crop:
                                st.write(f"**🥇 {c_name}**")
                            else:
                                st.write(f"**{c_name}**")
                        with colB:
                            st.progress(min(100, int(c_prob)))
                            st.caption(f"Suitability: {c_prob:.1f}%")

elif choice == "fertilizer":
    st.markdown('<div class="header-banner"><h2>🌾 Fertilizer Configurator</h2><p>Calculate exactly which fertilizer and how much you need for your plot.</p></div>', unsafe_allow_html=True)
    
    st.markdown("#### 1. Farm Details")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        land_area = st.number_input("Land Area", min_value=0.1, max_value=1000.0, value=4.0, step=0.5, help="Specify in Acres.")
        area_unit = st.selectbox("Unit", ["Acres", "Hectares"])
    with f2:
        crop_type = st.selectbox("Crop Type", list(CROP_DICT.keys()))
    with f3:
        soil_type = st.selectbox("Soil Type", list(SOIL_DICT.keys()))
    with f4:
        moisture = st.number_input("Soil Moisture (%)", 0, 100, 30, help="Current water retention in soil.")

    st.markdown("#### 2. Environmental & Soil Nutrients")
    e1, e2, e3, e4, e5 = st.columns(5)
    with e1:
        temperature = st.number_input("Field Temp (°C)", 0, 50, 25)
    with e2:
        humidity = st.number_input("Field Hum (%)", 0, 100, 50)
    with e3:
        n = st.number_input("N Level", 0, 200, 20)
    with e4:
        p = st.number_input("P Level", 0, 200, 20)
    with e5:
        k = st.number_input("K Level", 0, 200, 20)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Generate Fertilizer Prescription 📋", use_container_width=True):
        computed_area = land_area if area_unit == "Acres" else land_area * 2.471
        
        # Clip out of bounds
        t_c, h_c, m_c, n_c, p_c, k_c, a_c, warnings = validate_and_clip_fertilizer_inputs(
            n, p, k, temperature, humidity, moisture, computed_area
        )
        
        for w in warnings:
            st.warning(f"⚠️ {w}")
            
        if moisture == 0 and temperature > 40:
            st.warning("⚠️ Extreme dry conditions detected. Ensure proper irrigation before fertilizer application to prevent chemical burning.")
            
        soil_enc = SOIL_DICT[soil_type]
        crop_enc = CROP_DICT[crop_type]
        
        result = get_fertilizer_recommendation(
            fertilizer_model, t_c, h_c, m_c, 
            soil_enc, crop_enc, n_c, p_c, k_c, a_c
        )
        
        st.success(f"### 🌾 Recommended Fertilizer: **{result['fertilizer']}**")
        st.markdown(f"> 🧪 **Soil Analysis:** {result['explanation']}")
        
        sc1, sc2 = st.columns(2)
        with sc1:
            st.metric(label="Application Rate", value=f"{result['qty_per_acre']} kg/acre")
        with sc2:
            st.metric(label="Total Bulk Required", value=f"{result['total_qty']} kg", delta=f"For {computed_area:.1f} acres", delta_color="off")
                
elif choice == "disease":
    st.markdown('<div class="header-banner"><h2>🔍 Plant Disease Scanner</h2><p>Upload a clear image of a symptomatic leaf for an instant ML diagnosis.</p></div>', unsafe_allow_html=True)
    
    st.info("💡 **Instructions:** Ensure the image focuses closely on a single leaf with visible symptoms. The AI will reject generic non-plant images.")
    
    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            if st.button("Diagnose Disease 🔬", use_container_width=True):
                with st.spinner("Analyzing image features..."):
                    if not is_leaf_image(file_bytes):
                        st.error("❌ **Validation Failed:** The uploaded image does not appear to be a plant leaf. Please upload a clear image of a leaf.")
                    else:
                        preprocessed_img = preprocess_image(image)
                        disease, confidence = run_disease_detection(disease_model, preprocessed_img, DISEASE_CLASSES)
                        
                        if confidence < 0.60:
                            st.warning(f"⚠️ **Low Confidence ({confidence*100:.2f}%).** Unable to confidently identify disease. Best guess is **{disease}**, but please try another clearer image.")
                        else:
                            st.success(f"### 🦠 Diagnosis: **{disease}**")
                            st.info(f"🧬 Model Confidence: **{confidence*100:.2f}%**")
                            
                            if "Healthy" not in disease and GEMINI_API_KEY:
                                st.markdown("---")
                                st.markdown("### 🚑 AI Treatment Subroutine")
                                with st.spinner("Drafting multi-step treatment plan..."):
                                    bot = AgricultureChatbot(GEMINI_API_KEY)
                                    prompt = f"The crop has been diagnosed with '{disease}'. Provide a concise 3-part plan formatted with markdown: 1. Immediate Treatment Steps 2. Prevention Tips 3. Fertilizer/Soil Adjustments."
                                    advice = bot.get_response(prompt, [])
                                    st.write(advice)

elif choice == "weather":
    st.markdown('<div class="header-banner"><h2>🌦️ Weather Intelligence</h2><p>Track historical and forecasted weather trends to plan farming activities seamlessly.</p></div>', unsafe_allow_html=True)
    
    if not OPENWEATHER_API_KEY:
        st.error("OpenWeather API Key is missing. Check `.env` context.")
    else:
        wc1, wc2 = st.columns([1, 2])
        with wc1:
            city = st.text_input("Enter City Name:", "New Delhi")
            days = st.slider("Forecast/Trend Window (Days)", min_value=5, max_value=30, value=15)
            fetch_btn = st.button("Fetch Intelligence 📡")
            
        with wc2:
            st.info("Visualizes computed trend variance to help anticipate planting and harvesting risks over the selected timeline.")
            
        if fetch_btn:
            with st.spinner("Compiling meteorological data..."):
                current, err1 = get_current_weather(city, OPENWEATHER_API_KEY)
                trends, err2 = get_weather_forecast_trends(city, OPENWEATHER_API_KEY, days=days)
                
                if err1 or err2:
                    st.error(f"Error fetching data: {err1 or err2}")
                else:
                    st.markdown(f"### Current Condition in {city.title()}: **{current['weather'][0]['description'].title()}**")
                    
                    raw_temp = sum(t["temperature"] for t in trends) / len(trends)
                    raw_hum = sum(t["humidity"] for t in trends) / len(trends)
                    raw_rain = sum(t["rainfall"] for t in trends)
                    
                    st.markdown("---")
                    st.markdown(f"### 📈 {days}-Day Farming Insight Period")
                    mc1, mc2, mc3 = st.columns(3)
                    # Use delta as visual flare
                    mc1.metric("Average Temperature", f"{raw_temp:.1f} °C", "Stable")
                    mc2.metric("Average Humidity", f"{raw_hum:.1f} %", "Optimal")
                    mc3.metric("Total Rainfall Capacity", f"{raw_rain:.1f} mm", "+ Expected")
                    
                    if raw_rain > 100:
                        st.info("💡 **Farming Insight:** High rainfall levels support water-intensive crops (like Rice/Paddy). Ensure drainage systems are clear.")
                    elif raw_temp > 35 and raw_rain < 20:
                        st.warning("⚠️ **Farming Insight:** Hot and arid conditions expected. Intense irrigation might be required. Consider drought-resistant seeds.")
                    else:
                        st.info("💡 **Farming Insight:** A balanced, moderate weather trend suitable for a vast variety of staple crops like Wheat or Maize.")
                    
                    st.markdown("---")
                    
                    df = pd.DataFrame(trends)
                    df.set_index("date", inplace=True)
                    
                    st.write("**Temperature Trend (°C)**")
                    st.line_chart(df["temperature"], use_container_width=True, color="#ff4b4b")
                    
                    st.write("**Humidity Trend (%)**")
                    st.line_chart(df["humidity"], use_container_width=True, color="#4b8bff")
                    
                    st.write("**Sporadic Rainfall Estimates (mm)**")
                    st.bar_chart(df["rainfall"], use_container_width=True, color="#2e7d32")

elif choice == "chat":
    st.markdown('<div class="header-banner"><h2>🤖 Farm AI Assistant</h2><p>Ask our expert agricultural LLM about crops, treatments, soil science, or farming practices.</p></div>', unsafe_allow_html=True)
    
    if not GEMINI_API_KEY:
        st.warning("Please configure your Gemini API Key in `.env` to use the chatbot.")
    else:
        bot = AgricultureChatbot(GEMINI_API_KEY)
        
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm your AI Agriculture Expert. How can I assist you with your farming decisions today?"}
            ]
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("E.g., What crop grows best in sandy soil? Which crop should I grow?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Consulting agricultural datasets..."):
                    # Pass history securely to the bot
                    history_filtered = [msg for msg in st.session_state.messages[:-1] if msg["role"] in ["user", "assistant"]]
                    
                    # Inject farm context invisibly if interacting directly
                    ctx = st.session_state.farm_context
                    context_injection = ""
                    if ctx["last_recommended_crop"] != "":
                        context_injection = f" [Invisible Context Injection to help you answer specific questions: The user's last generated crop recommendation was '{ctx['last_recommended_crop']}', located near '{ctx['location']}', with Nitrogen: {ctx['soil_n']}, Phosphorus: {ctx['soil_p']}, Potassium: {ctx['soil_k']}. If they ask 'which crop should I grow', refer to this context.]"
                    
                    response_text = bot.get_response(prompt + context_injection, history_filtered)
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
