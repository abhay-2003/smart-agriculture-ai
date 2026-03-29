import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    base_dir = r"c:\Users\abhay\Desktop\smart_agriculture_project"
    data_path = os.path.join(base_dir, "dataset", "Fertilizer Prediction.csv")
    model_dir = os.path.join(base_dir, "models")
    
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("Dropping missing values")
        df = df.dropna()
        
    # Mapping dictionaries from app.py to ensure the model uses exactly the same label encoding
    soil_dict = {"Sandy":0, "Loamy":1, "Black":2, "Red":3, "Clayey":4}
    crop_dict = {"Wheat":0, "Rice":1, "Maize":2, "Sugarcane":3, "Cotton":4}
    
    # Strip spaces from column names just in case
    df.columns = df.columns.str.strip()
    
    # Print unique values correctly aligned
    print("Unique Soil Types in dataset:", df['Soil Type'].unique())
    print("Unique Crop Types in dataset:", df['Crop Type'].unique())
    
    # Apply mapping
    # Note: If there are other classes in the CSV not mapped in app.py, this will create NaN. 
    # We will encode dynamically if they don't match exactly.
    df['Soil Type'] = df['Soil Type'].map(soil_dict)
    df['Crop Type'] = df['Crop Type'].map(crop_dict)
    
    # Drop rows if they had unmapped values
    if df.isnull().sum().sum() > 0:
        print("Warning: some soil or crop types were not mapped. Dropping NaNs.")
        df = df.dropna()
        
    X = df[['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
    y = df['Fertilizer Name']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training RandomForest model for Fertilizer Recommendation...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    
    model_path = os.path.join(model_dir, "fertilizer_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    main()
