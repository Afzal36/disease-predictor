from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('health_model_1.pkl')

# List of 133 symptoms from your dataset
import json

with open('features.json') as f:
    feature_names = json.load(f)
feature_names.remove('prognosis')

@app.route('/')
def index():
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    selected = request.form.getlist("symptoms")  # list of selected symptoms

    input_vector = [1 if feature in selected else 0 for feature in feature_names]
    prediction = model.predict([input_vector])[0]

    return f"Predicted disease: <b>{prediction}</b>"

if __name__ == '__main__':
    app.run(debug=True)
