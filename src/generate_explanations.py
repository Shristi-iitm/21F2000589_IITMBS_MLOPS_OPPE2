import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# Load the model and some data
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

train_df = pd.read_csv('data/prepared/train.csv')
X_train = train_df.drop('target', axis=1)

# Encode categorical columns (like 'male', 'female') to numeric
X_train_encoded = pd.get_dummies(X_train)

# --- Use SHAP TreeExplainer on the encoded data ---
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train_encoded)

# Generate and save summary plot
# For classifiers shap_values is a list, pick class 1 if binary classification
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X_train_encoded, show=False)
else:
    shap.summary_plot(shap_values, X_train_encoded, show=False)

plt.savefig('shap_summary_plot.png', bbox_inches='tight')
plt.close()

print("SHAP summary plot saved to shap_summary_plot.png")
print("Based on the plot, you can describe in plain English which features are most important for predicting heart disease.")

