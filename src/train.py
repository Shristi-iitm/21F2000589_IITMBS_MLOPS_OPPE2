import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Ensure the output directory exists
os.makedirs('model', exist_ok=True)

# Load the training data
train_df = pd.read_csv('data/prepared/train.csv')

# Check the data structure
print("Dataset shape:", train_df.shape)
print("Column names:", train_df.columns.tolist())
print("Data types:")
print(train_df.dtypes)

# Handle categorical variables
categorical_columns = train_df.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns found:", categorical_columns)

# Remove target from categorical columns if present
if 'target' in categorical_columns:
    categorical_columns.remove('target')

# Encode categorical variables
label_encoders = {}
train_df_encoded = train_df.copy()

for col in categorical_columns:
    le = LabelEncoder()
    train_df_encoded[col] = le.fit_transform(train_df[col])
    label_encoders[col] = le

# Define features (X) and target (y)
X_train = train_df_encoded.drop('target', axis=1)
y_train = train_df_encoded['target']

print("Features shape:", X_train.shape)
print("Target shape:", y_train.shape)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and encoders
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Model trained and saved to model/model.pkl")
print("Label encoders saved to model/label_encoders.pkl")
