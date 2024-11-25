import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from algorithms import evaluate_algorithms
from utils import load_data

def preprocess_data(df):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.drop(['sex', 'subject#'], axis=1))
    preprocessed_df = pd.DataFrame(scaled_features, columns=df.columns.drop(['sex', 'subject#']))
    preprocessed_df['sex'] = df['sex'].values
    return preprocessed_df, scaler

def evaluate_clusters(preprocessed_df):
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(preprocessed_df.drop('sex', axis=1))
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()

def apply_clustering(preprocessed_df):
    kmeans = KMeans(n_clusters=3, random_state=42)
    preprocessed_df['cluster'] = kmeans.fit_predict(preprocessed_df.drop('sex', axis=1))
    return preprocessed_df, kmeans.cluster_centers_

def visualize_clusters(preprocessed_df):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=preprocessed_df.iloc[:, 0], y=preprocessed_df.iloc[:, 1], hue=preprocessed_df['cluster'], palette='viridis')
    plt.title('KMeans Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(title='Cluster')
    plt.show()

def train_best_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_status_from_file(file_path, model, scaler):
    sample_input_df = pd.read_csv(file_path)
    sample_input_scaled = scaler.transform(sample_input_df)
    predictions = model.predict(sample_input_scaled)
    status_labels = {0: 'Normal', 1: 'Early Stage', 2: 'Advanced Stage'}
    predicted_statuses = [status_labels[prediction] for prediction in predictions]
    return predicted_statuses

def main():
    # Define file paths
    raw_data_path = 'D:/Parkinson-s-Disease-Detection/dataset/parkinsons_raw.data'
    csv_data_path = 'D:/Parkinson-s-Disease-Detection/dataset/parkinsons.csv'
    model_path = 'D:/Parkinson-s-Disease-Detection/model/best_model.pkl'
    scaler_path = 'D:/Parkinson-s-Disease-Detection/model/scaler.pkl'
    sample_input_path = 'D:/Parkinson-s-Disease-Detection/dataset/sample_input.csv'
    
    # Load data
    df = pd.read_csv(raw_data_path, header=0)
    df.to_csv(csv_data_path, index=False)
    df = pd.read_csv(csv_data_path)
    
    print("First few rows of the dataset:")
    print(df.head(10))
    print("\nDataset Information:")
    print(df.info())
    print("\nMissing Values Summary:")
    print(df.isnull().sum())

    sns.countplot(data=df, x='sex')
    plt.title('Count of Sex in the Dataset')
    plt.xlabel('Sex (0 = Female, 1 = Male)')
    plt.ylabel('Count')
    plt.show()

    df.hist(bins=15, figsize=(15, 15), edgecolor='black')
    plt.suptitle("Feature Distributions")
    plt.show()

    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix Heatmap")
    plt.show()
    
    preprocessed_df, scaler = preprocess_data(df)
    print(preprocessed_df.head())

    evaluate_clusters(preprocessed_df)
    preprocessed_df, cluster_centers = apply_clustering(preprocessed_df)
    print(preprocessed_df.head())
    print("Cluster Centers:\n", cluster_centers)

    visualize_clusters(preprocessed_df)
    
    df['cluster'] = preprocessed_df['cluster']
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='cluster')
    plt.title('Distribution of Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.show()
    
    X = preprocessed_df.drop(['sex', 'cluster'], axis=1).values
    y = preprocessed_df['cluster'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training and testing sets created.")
    
    model = train_best_model(X_train, y_train)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save the scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    #sample inputs->
    # sample_df = preprocessed_df.drop(['sex', 'cluster'], axis=1).sample(n=10, random_state=42)
    # sample_df.to_csv(sample_input_path, index=False)
    # print("Sample input data:")
    # print(sample_df)
    
    print("\nEvaluating different algorithms:")
    evaluate_algorithms()
    
     # Predict and print the results for the new input data
    predicted_statuses = predict_status_from_file(sample_input_path, model, scaler)
    
    print("\nFinal Predicted Disease Conditions for Sample Input:")
    for i, status in enumerate(predicted_statuses):
        print(f"Sample {i+1}: {status}")

if __name__ == "__main__":
    main()
