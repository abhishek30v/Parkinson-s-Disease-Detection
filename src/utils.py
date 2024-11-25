import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_data(filepath):
    df = pd.read_csv(filepath)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.drop(['sex', 'subject#'], axis=1))
    preprocessed_df = pd.DataFrame(scaled_features, columns=df.columns.drop(['sex', 'subject#']))
    preprocessed_df['sex'] = df['sex'].values
    
    # Apply KMeans clustering as in the main file
    kmeans = KMeans(n_clusters=3, random_state=42)
    preprocessed_df['cluster'] = kmeans.fit_predict(preprocessed_df.drop('sex', axis=1))
    
    X = preprocessed_df.drop(['sex', 'cluster'], axis=1).values
    y = preprocessed_df['cluster'].values
    
    data = {'features': X, 'labels': y}
    return data, preprocessed_df, scaler
