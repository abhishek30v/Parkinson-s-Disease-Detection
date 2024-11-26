import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_model(filepath):
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

def load_sample_data(filepath):
    sample_data = pd.read_csv(filepath)
    return sample_data

def load_scaler(filepath):
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def main():
    # Define file paths
    model_path = 'D:/Parkinson-s-Disease-Detection/model/best_model.pkl'
    scaler_path = 'D:/Parkinson-s-Disease-Detection/model/scaler.pkl'
    sample_input_path = 'D:/Parkinson-s-Disease-Detection/dataset/sample_input.csv'
    
    # Load the saved model
    model = load_model(model_path)
    
    # Load the saved scaler
    scaler = load_scaler(scaler_path)
    
    # Load and preprocess the sample data
    sample_data = load_sample_data(sample_input_path)
    sample_data_scaled = scaler.transform(sample_data)  # Use transform, not fit_transform

    # Make predictions using the loaded model
    predictions = model.predict(sample_data_scaled)
    
    # Map predictions to status labels
    status_labels = {0: 'Normal', 1: 'Early Stage', 2: 'Advanced Stage'}
    predicted_statuses = [status_labels[prediction] for prediction in predictions]
    
    # Print the predicted statuses for the sample input
    print("\nPredicted Disease Conditions for Sample Input:")
    for i, status in enumerate(predicted_statuses):
        print(f"Sample {i+1}: {status}")

    expected_labels = ['Normal', 'Early Stage', 'Advanced Stage', 'Normal', 'Early Stage']  # Adjust this based on your actual expected labels
    
    if len(expected_labels) == len(predicted_statuses):
        accuracy = accuracy_score(expected_labels, predicted_statuses)
        report = classification_report(expected_labels, predicted_statuses, target_names=['Normal', 'Early Stage', 'Advanced Stage'], zero_division=0)
        cm = confusion_matrix(expected_labels, predicted_statuses)

        # Print evaluation results
        print(f"\nAccuracy: {accuracy:.2f}")
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", cm)
    else:
        print(f"\nInconsistent number of samples in expected_labels and predicted statuses.")

if __name__ == "__main__":
    main()
