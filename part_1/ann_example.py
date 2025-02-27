import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

print(f"Tensorflow version: {tf.__version__}")

print("DATA PREPROCESSING")
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(f" X Values:  \n{X}")
print(f" y Values: \n{y}")

# Encoding categorical data
le = LabelEncoder()
print(f"Label Encoding  Gender Column Before: \n{X[:, 2]}")
X[:, 2] = le.fit_transform(X[:, 2])
print(f"Label Encoding  Gender Column After: \n{X[:, 2]}")

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough")
X = np.array(ct.fit_transform(X))
print(X)