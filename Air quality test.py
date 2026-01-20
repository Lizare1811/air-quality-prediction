#Test

import pandas as pd
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("C:/Users/meena/OneDrive/Desktop/Skola tech/DL/Air quality prediction/air_quality_dataset.csv")

# -------- Label Encoding (same columns as training) --------
cat_cols = ['Location_Type', 'Source_Label']   # use YOUR categorical column names

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# -------- Features & Target --------
X = df.drop('PM2.5', axis=1)
y = df['PM2.5']

# -------- Feature Scaling --------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------- Load Model --------
json_file = open("PM25_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights("PM25_model.h5")

print("Loaded model from disk")

# -------- Predictions --------
predictions = model.predict(X_scaled)

# Show sample predictions
for i in range(10, 15):
    print(
        f"Predicted PM2.5: {predictions[i][0]:.2f} | Actual: {y.iloc[i]:.2f}"
    )
