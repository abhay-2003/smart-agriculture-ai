import numpy as np
import cv2

def is_leaf_image(image_bytes):
    """OpenCV heuristic to check if the uploaded image is likely a leaf."""
    try:
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            return False
            
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define color range for green/yellow/brown hues common in leaves
        lower_bound = np.array([20, 20, 20])
        upper_bound = np.array([100, 255, 255])
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        ratio = cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])
        
        return ratio > 0.02  # At least 2% of the image must be leaf-colored
    except Exception as e:
        return False

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction."""
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0  # Must match training setup
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
