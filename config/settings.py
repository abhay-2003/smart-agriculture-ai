import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# API Keys
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Constants
DISEASE_CLASSES = [
    "Pepper Bell Bacterial Spot", "Pepper Bell Healthy", "Potato Early Blight",
    "Potato Healthy", "Potato Late Blight", "Tomato Bacterial Spot",
    "Tomato Early Blight", "Tomato Healthy", "Tomato Late Blight",
    "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites",
    "Tomato Target Spot", "Tomato Mosaic Virus", "Tomato Yellow Leaf Curl Virus"
]

SOIL_DICT = {"Sandy":0, "Loamy":1, "Black":2, "Red":3, "Clayey":4}
CROP_DICT = {"Wheat":0, "Rice":1, "Maize":2, "Sugarcane":3, "Cotton":4}
