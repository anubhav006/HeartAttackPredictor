import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# 1. Load the Real Dataset
csv_file = 'heart.csv'

if not os.path.exists(csv_file):
    print("   Error: 'heart.csv' not found.")
    print("   Please download it from Kaggle and put it in this folder.")
    exit()

df = pd.read_csv(csv_file)
print(f"✅ Data Loaded: {len(df)} patient records found.")

# 2. Rename columns to match our project (if needed)
# The standard UCI names are usually: 
# age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
# We ensure the column names match what our app expects.
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
              'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# 3. Preprocessing
# Check for any missing values and fill them (rare in this specific dataset but good practice)
df.fillna(df.mean(), inplace=True)

X = df.drop('target', axis=1) # Features (all columns except target)
y = df['target']              # Target variable (0 = Healthy, 1 = Heart Disease)

# 4. Split Data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
# We increase n_estimators for better performance on real data
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate Accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("------------------------------------------------")
print(f" Model Trained Successfully!")
print(f" Real-World Accuracy: {accuracy * 100:.2f}%")
print("------------------------------------------------")
print("Classification Report:")
print(classification_report(y_test, predictions))

# 7. Save the Model
filename = 'heart_attack_model.pkl'
pickle.dump(model, open(filename, 'wb'))
print(f" Model saved as '{filename}'")