import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Create a Synthetic Dataset (For demonstration)
# In a real scenario, you would load this: df = pd.read_csv('heart.csv')
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.randint(29, 78, n_samples),
    'sex': np.random.randint(0, 2, n_samples),
    'cp': np.random.randint(0, 4, n_samples),          # Chest Pain Type
    'trestbps': np.random.randint(94, 201, n_samples), # Resting Blood Pressure
    'chol': np.random.randint(126, 565, n_samples),    # Cholesterol
    'fbs': np.random.randint(0, 2, n_samples),         # Fasting Blood Sugar
    'restecg': np.random.randint(0, 3, n_samples),     # Resting ECG
    'thalach': np.random.randint(71, 203, n_samples),  # Max Heart Rate
    'exang': np.random.randint(0, 2, n_samples),       # Exercise Induced Angina
    'oldpeak': np.random.uniform(0.0, 6.2, n_samples), # ST Depression
    'slope': np.random.randint(0, 3, n_samples),       # Slope of peak exercise ST
    'ca': np.random.randint(0, 5, n_samples),          # Major vessels colored by flourosopy
    'thal': np.random.randint(0, 4, n_samples),        # Thalassemia
    'target': np.random.randint(0, 2, n_samples)       # 0 = No Disease, 1 = Disease
}

df = pd.DataFrame(data)

# 2. Preprocessing
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Save the Model
filename = 'heart_attack_model.pkl'
pickle.dump(model, open(filename, 'wb'))

print(f"Model trained and saved as {filename}")
print(f"Accuracy on dummy test set: {model.score(X_test, y_test) * 100:.2f}%")