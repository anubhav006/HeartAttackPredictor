from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import sqlite3
import datetime

app = Flask(__name__)

# Load the trained model
try:
    model = pickle.load(open('heart_attack_model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: Model file not found. Please run model.py first.")
    exit()

# --- DATABASE SETUP ---
def init_db():
    """Creates the database and table if they don't exist."""
    conn = sqlite3.connect('heart_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            age INTEGER,
            sex INTEGER,
            cp INTEGER,
            trestbps INTEGER,
            chol INTEGER,
            fbs INTEGER,
            restecg INTEGER,
            thalach INTEGER,
            exang INTEGER,
            oldpeak REAL,
            slope INTEGER,
            ca INTEGER,
            thal INTEGER,
            prediction_result TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized.")

# Initialize DB on startup
init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get data from JSON request
        data = request.get_json()
        
        # Extract features
        features = [
            float(data['age']),
            float(data['sex']),
            float(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            float(data['fbs']),
            float(data['restecg']),
            float(data['thalach']),
            float(data['exang']),
            float(data['oldpeak']),
            float(data['slope']),
            float(data['ca']),
            float(data['thal'])
        ]
        
        # 2. Make Prediction
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        output = int(prediction[0])

        if output == 1:
            result_text = "High Risk"
        else:
            result_text = "Low Risk"

        # 3. Save to Database
        try:
            conn = sqlite3.connect('heart_data.db')
            cursor = conn.cursor()
            
            # Prepare data for insertion
            insert_query = '''
                INSERT INTO predictions (
                    timestamp, age, sex, cp, trestbps, chol, fbs, restecg, 
                    thalach, exang, oldpeak, slope, ca, thal, prediction_result
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            # Create a tuple of values to insert
            record_values = (
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data['age'], data['sex'], data['cp'], data['trestbps'], 
                data['chol'], data['fbs'], data['restecg'], data['thalach'], 
                data['exang'], data['oldpeak'], data['slope'], data['ca'], 
                data['thal'], result_text
            )
            
            cursor.execute(insert_query, record_values)
            conn.commit()
            conn.close()
            print("Data saved to database successfully.")
            
        except Exception as db_error:
            print(f"Database Error: {db_error}")

        # 4. Return result to frontend
        return jsonify({'prediction_text': result_text})

    except Exception as e:
        return jsonify({'prediction_text': f"Error: {str(e)}"})

# New Route: View Data (Optional - for testing)
@app.route('/view_data')
def view_data():
    conn = sqlite3.connect('heart_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()
    return jsonify(rows)

if __name__ == "__main__":
    app.run(debug=True)