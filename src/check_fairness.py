import pandas as pd
import pickle
from fairlearn.metrics import MetricFrame, demographic_parity_difference
from sklearn.metrics import accuracy_score

# Load model and test data
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

test_df = pd.read_csv('data/prepared/test.csv')

# Target and sensitive feature
y_test = test_df['target']
sensitive_feature = test_df['gender']

# Encode gender numerically for the model
test_df_encoded = test_df.copy()
test_df_encoded['gender'] = test_df_encoded['gender'].map({'male': 0, 'female': 1})

# X_test with encoded gender
X_test = test_df_encoded.drop('target', axis=1)

# Get model predictions
y_pred = model.predict(X_test)

# If predictions are strings, convert to numeric (binary)
# Example: treat one class (say "setosa") as positive (1), others as 0
y_pred_binary = (y_pred == "setosa").astype(int)
y_test_binary = (y_test == "setosa").astype(int)

# Define metrics
metrics = {
    "selection_rate": lambda y_true, y_pred: y_pred.mean(),
    "accuracy": accuracy_score,
}

# Fairness metrics
metric_frame = MetricFrame(metrics=metrics,
                           y_true=y_test_binary,
                           y_pred=y_pred_binary,
                           sensitive_features=sensitive_feature)

dpd = demographic_parity_difference(y_true=y_test_binary,
                                    y_pred=y_pred_binary,
                                    sensitive_features=sensitive_feature)

print("Fairness metrics by gender (female, male):")
print(metric_frame.by_group)
print(f"\nDemographic Parity Difference: {dpd:.4f}")

