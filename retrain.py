import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset (adjust the filename if yours is named differently)
df = pd.read_csv('Data/Crop_recommendation.csv')

# Extract the exact features in the order expected by app.py
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Train a new Random Forest model
model = RandomForestClassifier(n_estimators=20, random_state=0)
model.fit(X, y)

# Save the new model to overwrite the old incompatible one
with open('models/RandomForest.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model successfully retrained and saved for scikit-learn 1.4.0!")