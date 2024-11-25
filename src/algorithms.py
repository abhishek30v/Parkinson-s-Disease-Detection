# src/algorithms.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from utils import load_data

def evaluate_algorithms():
   # df = pd.read_csv('D:/Parkinson-s-Disease-Detection/dataset/parkinsons_raw.data', header=0)
   # df.to_csv('D:/Parkinson-s-Disease-Detection/dataset/parkinsons.csv', index=False)
    data, preprocessed_df, scaler = load_data('D:/Parkinson-s-Disease-Detection/dataset/parkinsons.csv')
    
    X_train, X_test, y_train, y_test = train_test_split(data['features'], data['labels'], test_size=0.2, random_state=42)
    
    algorithms = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(C=8, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(eval_metric='mlogloss', tree_method='hist')
    }
    
    accuracies = {}
    for name, algo in algorithms.items():
        algo.fit(X_train, y_train)
        predictions = algo.predict(X_test)
        accuracies[name] = accuracy_score(y_test, predictions)
        print(f"{name} Test Set Accuracy: {accuracies[name]:.2f}")
        
        # Print classification report
        print(f"Classification Report for {name}:\n", classification_report(y_test, predictions))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    # Explicitly print the accuracies dictionary
    print("\nAccuracy Comparison:")
    for name, accuracy in accuracies.items():
        print(f"{name}: {accuracy:.2f}")
    
    return accuracies

if __name__ == "__main__":
    evaluate_algorithms()
