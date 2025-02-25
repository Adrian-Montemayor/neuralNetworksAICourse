import numpy as np
import pandas as pd
import tensorflow as tf

print(f"Tensorflow version: {tf.__version__}")
### Data preprocessing ### 

dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)