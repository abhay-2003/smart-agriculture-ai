import numpy as np

def run_disease_detection(model, preprocessed_image, class_names):
    """Runs prediction and validates confidence."""
    prediction = model.predict(preprocessed_image)[0]
    class_index = np.argmax(prediction)
    confidence = float(prediction[class_index])
    
    disease = class_names[class_index]
    
    return disease, confidence
