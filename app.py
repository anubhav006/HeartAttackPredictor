from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('heart_attack_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from JSON request
        data = request.get_json()
        
        # Extract features in the correct order
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
        
        # Convert to numpy array and reshape for prediction
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        output = int(prediction[0])

        if output == 1:
            result = "High Risk of Heart Attack"
        else:
            result = "Low Risk / Normal"

        return jsonify({'prediction_text': result})

    except Exception as e:
        return jsonify({'prediction_text': f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)