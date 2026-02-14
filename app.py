from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import numpy as np
import sqlite3
import datetime
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random string for security

# --- LOGIN SETUP ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Redirect here if user isn't logged in

# Load Model
try:
    model = pickle.load(open('heart_attack_model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: Model file not found. Run model.py first.")
    exit()

# --- DATABASE SETUP ---
DB_NAME = "heart_data.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 1. Create Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')

    # 2. Create Predictions Table (Added user_id column)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TEXT,
            age INTEGER, sex INTEGER, cp INTEGER, trestbps INTEGER,
            chol INTEGER, fbs INTEGER, restecg INTEGER, thalach INTEGER,
            exang INTEGER, oldpeak REAL, slope INTEGER, ca INTEGER,
            thal INTEGER, prediction_result TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

if not os.path.exists(DB_NAME):
    init_db()

# --- USER CLASS ---
class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user_data = cursor.fetchone()
    conn.close()
    if user_data:
        return User(id=user_data[0], username=user_data[1], password_hash=user_data[2])
    return None

# --- ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user_data = cursor.fetchone()
        conn.close()

        if user_data and check_password_hash(user_data[2], password):
            user_obj = User(id=user_data[0], username=user_data[1], password_hash=user_data[2])
            login_user(user_obj)
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')

    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    hashed_password = generate_password_hash(password, method='scrypt')

    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        conn.close()
        flash('Account created! Please log in.')
    except sqlite3.IntegrityError:
        flash('Username already exists.')
    
    return redirect(url_for('login'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    return render_template('index.html', name=current_user.username)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.get_json()
        features = [
            float(data['age']), float(data['sex']), float(data['cp']),
            float(data['trestbps']), float(data['chol']), float(data['fbs']),
            float(data['restecg']), float(data['thalach']), float(data['exang']),
            float(data['oldpeak']), float(data['slope']), float(data['ca']),
            float(data['thal'])
        ]
        
        prediction = model.predict([np.array(features)])
        result_text = "High Risk" if prediction[0] == 1 else "Low Risk"

        # Save to DB with current_user.id
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (
                user_id, timestamp, age, sex, cp, trestbps, chol, fbs, restecg, 
                thalach, exang, oldpeak, slope, ca, thal, prediction_result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            current_user.id, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data['age'], data['sex'], data['cp'], data['trestbps'], 
            data['chol'], data['fbs'], data['restecg'], data['thalach'], 
            data['exang'], data['oldpeak'], data['slope'], data['ca'], 
            data['thal'], result_text
        ))
        conn.commit()
        conn.close()

        return jsonify({'prediction_text': result_text})

    except Exception as e:
        return jsonify({'prediction_text': f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)