'''Build a decision tree classifier to predict whether a customer will purchase a product or 
service based on their demographic and behavioral data. Use a dataset such as the Bank Marketing 
dataset from the UCI Machine Learning Repository.'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset_path = "bank-additional.csv"
df = pd.read_csv(dataset_path, delimiter=';')
print("Data loaded successfully.")
print(df.head())

# Label encoding for categorical variables, excluding the target 'y'
label_encoders = {col: LabelEncoder().fit(df[col]) for col in 
                  df.select_dtypes(include=['object']).columns if col != 'y'}
df_encoded = df.copy()
for col, le in label_encoders.items():
    df_encoded[col] = le.transform(df_encoded[col])

# Encode the target variable 'y'
df_encoded['y'] = df_encoded['y'].map({'yes': 1, 'no': 0})

# Split the data into features (X) and target (y)
X = df_encoded.drop('y', axis=1)
y = df_encoded['y']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree classifier with limited depth
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot Feature Importances
importances_dt = clf.feature_importances_
feature_importance_df_dt = pd.DataFrame({'Feature': X_train.columns, 
                            'Importance': importances_dt}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df_dt, color='skyblue')
plt.xlabel('Importance')
plt.title('Decision Tree Feature Importance')
plt.tight_layout()
plt.show()

# Plotting bar plots for categorical features
categorical_columns = df.select_dtypes(include=['object']).columns
num_features = len(categorical_columns)
num_cols = 3 
num_rows = (num_features + num_cols - 1) // num_cols  
fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(16, 12))  
fig.suptitle('Bar Plots of Categorical Features', fontsize=16)
for i, feature in enumerate(categorical_columns):
    ax = axs[i // num_cols, i % num_cols]
    sns.countplot(x=feature, data=df, palette='plasma', ax=ax)
    ax.set_title(f'{feature}', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')  
for j in range(len(categorical_columns), num_rows * num_cols):
    fig.delaxes(axs[j // num_cols, j % num_cols])
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Plotting box plots for numerical features
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
num_features = len(numerical_columns)
num_cols = 3  
num_rows = (num_features + num_cols - 1) // num_cols
fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 6))
fig.suptitle('Box Plots of Numerical Features', fontsize=18)
for i, feature in enumerate(numerical_columns):
    ax = axs[i // num_cols, i % num_cols]
    sns.boxplot(x=df[feature], ax=ax)
    ax.set_title(f'{feature}', fontsize=10)

# Hide any unused subplots
for j in range(len(numerical_columns), num_rows * num_cols):
    fig.delaxes(axs[j // num_cols, j % num_cols])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Plot correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df_encoded.corr(), annot=True, cmap='rainbow', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Visualize the decision tree (limited depth)
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X_train.columns, class_names=['No', 'Yes'], filled=True, 
          rounded=True,fontsize=10)
plt.title('Decision Tree Visualization (Depth=5)')
plt.tight_layout()
plt.show()

# Visualize the entire decision tree
clf_full = DecisionTreeClassifier(random_state=42)  # No max_depth to see the full tree
clf_full.fit(X_train, y_train)

plt.figure(figsize=(12, 8))
plot_tree(clf_full, feature_names=X_train.columns, class_names=['No', 'Yes'], filled=True, rounded=True)
plt.title('Full Decision Tree Visualization')
plt.tight_layout()
plt.show()
