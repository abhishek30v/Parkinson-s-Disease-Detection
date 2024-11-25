# src/app.py
from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add the project root directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.main import predict_status_from_file  # Adjusted import path

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the model
    model_path = 'D:/Parkinson-s-Disease-Detection/model/best_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Preprocess input data
    input_data = request.form['input_data']
    input_df = pd.read_csv(pd.compat.StringIO(input_data))
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_df)
    
    # Make predictions
    predictions = model.predict(input_scaled)
    status_labels = {0: 'Normal', 1: 'Early Stage', 2: 'Advanced Stage'}
    predicted_statuses = [status_labels[prediction] for prediction in predictions]
    
    return render_template('result.html', predictions=predicted_statuses)

if __name__ == '__main__':
    app.run(debug=True)
