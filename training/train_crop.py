import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    base_dir = r"c:\Users\abhay\Desktop\smart_agriculture_project"
    data_path = os.path.join(base_dir, "dataset", "Crop_recommendation.csv")
    model_dir = os.path.join(base_dir, "models")
    
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Features:", list(df.columns))
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("Dropping missing values")
        df = df.dropna()
        
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training RandomForest model for Crop Recommendation...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    # print(classification_report(y_test, y_pred))
    
    model_path = os.path.join(model_dir, "crop_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    main()
