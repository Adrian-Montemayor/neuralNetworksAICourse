import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

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

print(f"HotEncoder before: {X}")
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough")
X = np.array(ct.fit_transform(X))
print(f"HotEncoder after: {X}")

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)


# Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


print("Initializing the ANN...")
ann = tf.keras.models.Sequential()

print("Adding the input layer and the first hidden layer...")
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
print("Adding the second hidden layer...")
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
print("Adding the output layer...")
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

print("Training the ANN")

print("Compiling the ANN")
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("Training the ANN on the Training set")
ann.fit(X_train, y_train, batch_size=32, epochs=100)

prediction = ann.predict(sc.transform([[1,0,0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
print("Should we say goodbye to that customer?")
print("No" if prediction >= .5  else "Yes")