from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import pickle
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = 'super_secure_heartguard_key_123'

# --- FLASK LOGIN SETUP ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('heart_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, username FROM users WHERE id = ?", (user_id,))
    user_data = cursor.fetchone()
    conn.close()
    if user_data:
        return User(id=user_data[0], username=user_data[1])
    return None

# --- DATABASE INITIALIZATION ---
def init_db():
    conn = sqlite3.connect('heart_data.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL)''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        timestamp TEXT,
                        patient_name TEXT,
                        age REAL, sex REAL, cp REAL, trestbps REAL, chol REAL, 
                        fbs REAL, restecg REAL, thalach REAL, exang REAL, 
                        oldpeak REAL, slope REAL, ca REAL, thal REAL,
                        prediction_result TEXT,
                        FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

init_db()

# --- LOAD ML MODEL ---
try:
    model = pickle.load(open('heart_attack_model.pkl', 'rb'))
    print("Model loaded successfully!")
except Exception as e:
    print("Warning: Model file not found. Ensure 'heart_attack_model.pkl' is in the folder.")
    model = None

# --- ROUTES ---
@app.route('/')
def home():
    name = current_user.username if current_user.is_authenticated else "Guest Assessor"
    return render_template('index.html', name=name)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('heart_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            user_obj = User(id=user[0], username=user[1])
            login_user(user_obj)
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))
            
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    hashed_password = generate_password_hash(password)

    conn = sqlite3.connect('heart_data.db')
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        flash('Account created successfully! Please log in.', 'success')
    except sqlite3.IntegrityError:
        flash('Username already exists. Try another one.', 'error')
    finally:
        conn.close()
    return redirect(url_for('login'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/profile')
@login_required
def profile():
    conn = sqlite3.connect('heart_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, patient_name, prediction_result, age, trestbps FROM predictions WHERE user_id = ? ORDER BY id DESC", (current_user.id,))
    history = cursor.fetchall()
    conn.close()
    return render_template('profile.html', name=current_user.username, history=history)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # 1. Get 7 inputs from the frontend
        patient_name = data.get('patient_name', 'Unknown')
        age = float(data.get('age'))
        sex = float(data.get('sex'))
        cp = float(data.get('cp'))
        trestbps = float(data.get('trestbps'))
        chol = float(data.get('chol'))
        fbs = float(data.get('fbs'))
        thalach = float(data.get('thalach'))

        # 2. ULTRA-SAFE Parameters for missing fields
        restecg = 1.0  
        exang = 0.0    
        oldpeak = 0.0  
        slope = 2.0    
        ca = 0.0       
        thal = 2.0     

        # 3. Combine in exact training order
        feature_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        print("\n" + "="*40)
        print("--- NEW PREDICTION TRIGGERED ---")
        print("Inputs going to model:", feature_values)

        # 4. Predict
        if model:
            final_features = [np.array(feature_values)]
            prediction = model.predict(final_features)
            
            print("Model Output Value:", prediction[0])
            print("="*40 + "\n")
            
            # --- THE DECISION LOGIC ---
            # Agar output hamesha ulta aa raha hai, toh bas '== 1' ko '== 0' kar dijiye.
            if prediction[0] == 0:
                result = "High Risk Detected" 
            else:
                result = "Normal Profile"
        else:
            result = "Error: Model missing"

        # 5. Save to database if logged in
        if current_user.is_authenticated:
            conn = sqlite3.connect('heart_data.db')
            cursor = conn.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute('''INSERT INTO predictions 
                              (user_id, timestamp, patient_name, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, prediction_result) 
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                           (current_user.id, timestamp, patient_name, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, result))
            conn.commit()
            conn.close()

        return jsonify({'prediction_text': result})

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'prediction_text': "Error processing data"}), 400

if __name__ == '__main__':
    app.run(debug=True)