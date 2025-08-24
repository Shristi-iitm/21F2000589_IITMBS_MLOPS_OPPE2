import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Ensure the output directory exists
os.makedirs('data/prepared', exist_ok=True)

# Read the data
df = pd.read_csv('data/heart.csv')

# Split the data
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Save the splits
train.to_csv('data/prepared/train.csv', index=False)
test.to_csv('data/prepared/test.csv', index=False)

print("Data split into train and test sets successfully.")
