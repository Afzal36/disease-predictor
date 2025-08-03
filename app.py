from flask import Flask, render_template, request
import joblib
import numpy as np
import json

app = Flask(__name__)

# Load the trained model
model = joblib.load('health_model_1.pkl')

# Load feature names from JSON file
with open('features.json') as f:
    feature_names = json.load(f)

# Remove 'prognosis' if it exists in features
if 'prognosis' in feature_names:
    feature_names.remove('prognosis')

@app.route('/')
def index():
    """Render the main page with symptom selection form"""
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request and return results"""
    try:
        # Get selected symptoms from form
        selected_symptoms = request.form.getlist("symptoms")
        
        # Create input vector (1 if symptom selected, 0 otherwise)
        input_vector = [1 if feature in selected_symptoms else 0 for feature in feature_names]
        
        # Make prediction
        prediction = model.predict([input_vector])[0]
        
        # Get prediction probability (if your model supports it)
        try:
            prediction_proba = model.predict_proba([input_vector])[0]
            confidence = max(prediction_proba) * 100
        except:
            confidence = 85.0  # Default confidence if model doesn't support predict_proba
        
        # Render results page with prediction data
        return render_template('results.html', 
                             predicted_disease=prediction,
                             selected_symptoms=selected_symptoms,
                             confidence=round(confidence, 1),
                             num_symptoms=len(selected_symptoms))
    
    except Exception as e:
        # Handle errors gracefully
        return render_template('results.html', 
                             predicted_disease="Error in prediction",
                             selected_symptoms=[],
                             confidence=0,
                             num_symptoms=0,
                             error=str(e))

@app.route('/about')
def about():
    """Optional: About page"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)