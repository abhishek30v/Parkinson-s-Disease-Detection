import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv('D:\Parkinson-s-Disease-Detection\dataset\parkinsons_raw.data', header=0)

df.to_csv(r'D:\Parkinson-s-Disease-Detection\dataset\parkinsons.csv', index=False)
df = pd.read_csv(r'D:\Parkinson-s-Disease-Detection\dataset\parkinsons.csv')

# Data exploration
print("First few rows of the dataset:")
print(df.head(10))
print("\nDataset Information:")
print(df.info())
print("\nMissing Values Summary:")
print(df.isnull().sum())

# Visualize the distribution of 'gender'
sns.countplot(data=df, x='sex')
plt.title('Count of Sex in the Dataset')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.ylabel('Count')
plt.show()

# Plot the distribution of all features
df.hist(bins=15, figsize=(15, 15), edgecolor='black')
plt.suptitle("Feature Distributions")
plt.show()

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

# Standardize the features (excluding 'sex' and 'subject#')
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(['sex', 'subject#'], axis=1))

# Convert scaled features back into a DataFrame
preprocessed_df = pd.DataFrame(scaled_features, columns=df.columns.drop(['sex', 'subject#']))
preprocessed_df['sex'] = df['sex'].values
print(preprocessed_df.head())

# Determine the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(preprocessed_df.drop('sex', axis=1))
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve to determine the optimal number of clusters
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Apply KMeans with the optimal number of clusters
kmeans = KMeans(n_clusters=3, random_state=42)
preprocessed_df['cluster'] = kmeans.fit_predict(preprocessed_df.drop('sex', axis=1))
print(preprocessed_df.head())

# Visualize the clusters based on the first two principal components or features
plt.figure(figsize=(10, 8))
sns.scatterplot(x=preprocessed_df.iloc[:, 0], y=preprocessed_df.iloc[:, 1], hue=preprocessed_df['cluster'], palette='viridis')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(title='Cluster')
plt.show()

# Examine cluster centers
cluster_centers = kmeans.cluster_centers_
print("Cluster Centers:\n", cluster_centers)

# Add cluster labels to the original DataFrame for comparison
df['cluster'] = preprocessed_df['cluster']

# Visualize the distribution of clusters
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='cluster')
plt.title('Distribution of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()

# Define features and labels for classification
X = preprocessed_df.drop(['sex', 'cluster'], axis=1).values
y = preprocessed_df['cluster'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training and testing sets created.")

# Train and evaluate Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print(f"Logistic Regression Accuracy: {accuracy_log_reg:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_log_reg))
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
plt.figure(figsize=(5, 5))
sns.heatmap(cm_log_reg, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Train and evaluate Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_tree = decision_tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Decision Tree Accuracy: {accuracy_tree:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_tree))
cm_tree = confusion_matrix(y_test, y_pred_tree)
plt.figure(figsize=(5, 5))
sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Blues')
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Train and evaluate Support Vector Machine (SVM)
svm = SVC(C=8, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(5, 5))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Train and evaluate Random Forest Classifier
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Train and evaluate XGBoost Classifier
xgb = XGBClassifier(eval_metric='mlogloss', tree_method='hist')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(5, 5))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Randomly sample 5 rows from the preprocessed DataFrame
sample_df = preprocessed_df.drop(['sex', 'cluster'], axis=1).sample(n=5, random_state=42)

# Save the sample to a CSV file
sample_df.to_csv('D:\Parkinson-s-Disease-Detection\dataset\sample_input.csv', index=False)
print("Sample input data:")
print(sample_df)

# Summarize accuracies
accuracy_results = {
    'Logistic Regression': accuracy_log_reg,
    'Decision Tree': accuracy_tree,
    'SVM': accuracy_svm,
    'Random Forest': accuracy_rf,
    'XGBoost': accuracy_xgb
}
accuracy_df = pd.DataFrame(list(accuracy_results.items()), columns=['Algorithm', 'Accuracy'])
print("\nAccuracy Comparison:")
print(accuracy_df)

# Function to predict status for each row in the sample input
def predict_status_from_file(file_path, model):
    sample_input_df = pd.read_csv(file_path)
    sample_input_scaled = scaler.transform(sample_input_df)
    predictions = model.predict(sample_input_scaled)
    status_labels = {0: 'Normal', 1: 'Early Stage', 2: 'Advanced Stage'}
    predicted_statuses = [status_labels[prediction] for prediction in predictions]
    return predicted_statuses

# Make predictions for the sample input
predicted_statuses = predict_status_from_file('sample_input.csv', random_forest)
print("\n\nPredicted statuses for the sample input:")
for i, status in enumerate(predicted_statuses):
    print(f"Sample {i+1}: {status}")