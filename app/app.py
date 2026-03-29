import streamlit as st
import numpy as np
import joblib
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from PIL import Image
import cv2
import requests
import google.generativeai as genai

# -----------------------------
# Configuration & CSS
# -----------------------------
st.set_page_config(page_title="Smart Agriculture AI", page_icon="🌱", layout="wide")

st.markdown("""
<style>
    .main {background-color: #f4f8f4;}
    h1 {color: #2e7d32;}
    h2, h3 {color: #1b5e20;}
    .stButton>button {
        background-color: #4caf50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #388e3c;
    }
    .css-1d391kg {background-color: #e8f5e9;}
    .stTextInput>div>div>input {border-radius: 5px;}
    .stNumberInput>div>div>input {border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Models
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

@st.cache_resource
def load_models():
    crop_path = os.path.join(BASE_DIR, "models", "crop_model.pkl")
    fert_path = os.path.join(BASE_DIR, "models", "fertilizer_model.pkl")
    disease_path = os.path.join(BASE_DIR, "models", "plant_disease_model.h5")
    
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
# Metadata
# -----------------------------
DISEASE_CLASSES = [
    "Pepper Bell Bacterial Spot", "Pepper Bell Healthy", "Potato Early Blight",
    "Potato Healthy", "Potato Late Blight", "Tomato Bacterial Spot",
    "Tomato Early Blight", "Tomato Healthy", "Tomato Late Blight",
    "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites",
    "Tomato Target Spot", "Tomato Mosaic Virus", "Tomato Yellow Leaf Curl Virus"
]

SOIL_DICT = {"Sandy":0, "Loamy":1, "Black":2, "Red":3, "Clayey":4}
CROP_DICT = {"Wheat":0, "Rice":1, "Maize":2, "Sugarcane":3, "Cotton":4}

# -----------------------------
# Helper Functions
# -----------------------------
def is_leaf_image(image_bytes):
    """OpenCV heuristic to check if the uploaded image is likely a leaf."""
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color range for green/yellow/brown hues common in leaves
    lower_bound = np.array([10, 20, 20])
    upper_bound = np.array([100, 255, 255])
    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    ratio = cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])
    
    return ratio > 0.01  # At least 1% of the image must be leaf-colored

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=100)
st.sidebar.title("Smart Agri AI")

options = ["🏠 Home Dashboard", "🌱 Crop Recommendation", "🌾 Fertilizer Recommendation", 
           "🔍 Plant Disease Detection", "🌦️ Weather Intelligence", "🤖 AI Chatbot"]
choice = st.sidebar.radio("Navigate", options)

# API Keys input in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("API Configuration")
openweather_key = st.sidebar.text_input("OpenWeather API Key", value="bcbf6fafca81e73d04a8bf33fb05dd54", type="password")
gemini_key = st.sidebar.text_input("Gemini API Key", value="AIzaSyDDiRQcn3YQqHwUxA1AD07AruTJo_KY-Lo", type="password")

# -----------------------------
# Page Modules
# -----------------------------
if choice == "🏠 Home Dashboard":
    st.title("Welcome to Smart Agriculture AI System 🌱")
    st.markdown("""
    This comprehensive platform empowers farmers with AI-driven insights to maximize yield and minimize losses.
    
    ### Features:
    - **Crop Recommendation:** Discover the optimal crop for your soil and environment.
    - **Fertilizer Recommendation:** Get precise fertilizer suggestions based on soil metrics.
    - **Disease Detection:** Upload a leaf image to instantly diagnose plant diseases.
    - **Weather Intelligence:** Get real-time weather data to plan farming activities.
    - **AI Chatbot:** Ask any farming-related questions to our intelligent assistant.
    """)
    st.image("https://images.unsplash.com/photo-1625246333195-78d9c38ad449?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", use_column_width=True)

elif choice == "🌱 Crop Recommendation":
    st.title("Crop Recommendation System")
    st.write("Enter the environmental and soil parameters to find the best crop.")
    
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Nitrogen (N)", 0, 200, 50)
        p = st.number_input("Phosphorus (P)", 0, 200, 50)
        k = st.number_input("Potassium (K)", 0, 200, 50)
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    with col2:
        temperature = st.number_input("Temperature (°C)", -10.0, 60.0, 25.0)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)
        rainfall = st.number_input("Rainfall (mm)", -10.0, 500.0, 100.0)
        
    if st.button("Predict Crop ✨"):
        # Validations
        if temperature <= 0:
            st.error("Temperature is unrealistic for crop growth.")
        elif ph < 3 or ph > 10:
            st.error("Soil pH is outside the realistic range (3-10).")
        elif rainfall < 0:
            st.error("Rainfall cannot be negative.")
        else:
            input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
            prediction = crop_model.predict(input_data)
            st.success(f"**Recommended Crop:** {prediction[0].upper()}")

elif choice == "🌾 Fertilizer Recommendation":
    st.title("Fertilizer Recommendation System")
    
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input("Temperature (°C)", 0, 50, 25)
        humidity = st.number_input("Humidity (%)", 0, 100, 50)
        moisture = st.number_input("Moisture (%)", 0, 100, 30)
        soil_type = st.selectbox("Soil Type", list(SOIL_DICT.keys()))
    with col2:
        crop_type = st.selectbox("Crop Type", list(CROP_DICT.keys()))
        nitrogen = st.number_input("Nitrogen Level", 0, 200, 20)
        potassium = st.number_input("Potassium Level", 0, 200, 20)
        phosphorus = st.number_input("Phosphorus Level", 0, 200, 20)
        
    if st.button("Recommend Fertilizer ✨"):
        if moisture == 0 and temperature > 40:
            st.warning("Extreme dry conditions detected. Ensure proper irrigation.")
            
        soil_enc = SOIL_DICT[soil_type]
        crop_enc = CROP_DICT[crop_type]
        input_data = np.array([[temperature, humidity, moisture, soil_enc, crop_enc, nitrogen, potassium, phosphorus]])
        
        prediction = fertilizer_model.predict(input_data)
        st.success(f"**Recommended Fertilizer:** {prediction[0]}")

elif choice == "🔍 Plant Disease Detection":
    st.title("Plant Disease Detection")
    st.write("Upload a clear image of a plant leaf.")
    
    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Diagnose Disease 🔬"):
            # Step 1: Leaf Detection Heuristic
            if not is_leaf_image(file_bytes):
                st.error("❌ The uploaded image does not appear to be a clear leaf. Please upload a valid plant leaf image.")
            else:
                img = image.resize((224,224))
                img_array = np.array(img)
                img_array = img_array / 255.0  # Must match training setup
                img_array = np.expand_dims(img_array, axis=0)
                
                prediction = disease_model.predict(img_array)[0]
                class_index = np.argmax(prediction)
                confidence = prediction[class_index]
                
                disease = DISEASE_CLASSES[class_index]
                # Step 2: Confidence Threshold Check
                if confidence < 0.60:
                    st.warning(f"⚠️ **Low Confidence ({confidence*100:.2f}%).** Best guess: **{disease}**. Please try another clearer image.")
                else:
                    st.success(f"**Diagnosis:** {disease}")
                    st.info(f"Confidence: {confidence*100:.2f}%")

elif choice == "🌦️ Weather Intelligence":
    st.title("Weather-Based Farming Suggestions")
    
    if not openweather_key:
        st.warning("Please enter your OpenWeather API Key in the sidebar.")
    else:
        city = st.text_input("Enter City Name:", "New Delhi")
        if st.button("Get Weather"):
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={openweather_key}&units=metric"
            try:
                response = requests.get(url).json()
                if response.get("cod") != 200:
                    st.error(f"Error: {response.get('message')}")
                else:
                    temp = response["main"]["temp"]
                    humidity = response["main"]["humidity"]
                    desc = response["weather"][0]["description"].title()
                    
                    st.subheader(f"Weather in {city}")
                    st.write(f"**Condition:** {desc}")
                    st.write(f"**Temperature:** {temp} °C")
                    st.write(f"**Humidity:** {humidity} %")
                    
                    # Basic AI-like rule suggestions
                    st.markdown("### Farming Advice:")
                    if temp > 35:
                        st.info("High temperature detected. Ensure adequate watering and shading for sensitive crops.")
                    elif temp < 10:
                        st.info("Low temperature detected. Protect crops from frost.")
                    if humidity > 80:
                        st.info("High humidity increases the risk of fungal diseases. Monitor plants closely.")
                        
            except Exception as e:
                st.error("Failed to fetch weather data. Check your connection or API key.")

elif choice == "🤖 AI Chatbot":
    st.title("Agriculture AI Assistant")
    st.write("Ask me anything about farming, soil, crops, or plant diseases!")
    
    if not gemini_key:
        st.warning("Please enter your Gemini API Key in the sidebar to use the chatbot.")
    else:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("E.g., What is the best fertilizer for tomatoes?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                system_instruction = "You are an agriculture expert helping farmers. Provide practical advice."
                full_prompt = f"{system_instruction}\nUser: {prompt}"
                try:
                    response = model.generate_content(full_prompt)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Error generating response: {e}")