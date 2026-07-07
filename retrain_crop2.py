import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

def retrain():
    csv_path = 'Data/Crop_recommendation.csv'
    model_path = 'models/crop2.pkl'
    
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Sort labels alphabetically to define the 1-22 class mapping
    unique_labels = sorted(df['label'].unique())
    label_to_index = {label: i + 1 for i, label in enumerate(unique_labels)}
    
    print("Alphabetical mapping definition:")
    for label, idx in label_to_index.items():
        print(f"  {label:15} -> {idx}")
        
    # Encode label column
    df['encoded_label'] = df['label'].map(label_to_index)
    
    # Features: N, P, K, temperature, humidity, ph, rainfall
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['encoded_label']
    
    print("Training RandomForestClassifier locally for crop2.pkl...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print(f"Saving retrained compatible model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print("Retraining completed successfully! crop2.pkl is now compatible with scikit-learn 1.4.0.")

if __name__ == '__main__':
    retrain()
