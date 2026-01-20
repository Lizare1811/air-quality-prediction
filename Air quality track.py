#Air quality predcition

#Track

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#Load the dataset
df = pd.read_csv("C:/Users/meena/OneDrive/Desktop/Skola tech/DL/Air quality prediction/air_quality_dataset.csv")  

#Display first few rows
print(df.head())

#Basic dataset understanding (numeric + categorical)

print("Dataset Shape:")
print(df.shape)

print("\nDataset Info:")
df.info()

print("\nNumeric Columns:")
print(df.describe())

print("\nCategorical Columns:")
print(df.describe(include='object'))

#EDA
#Distribution of target variable (PM2.5)
#Shows spread of pollution levels
plt.figure(figsize=(6,4))
plt.hist(df['PM2.5'], bins=30)
plt.title("Distribution of PM2.5")
plt.xlabel("PM2.5")
plt.ylabel("Frequency")
plt.show()

#PM10 vs PM2.5 (key relationship)
#Shows correlation visually
plt.figure(figsize=(6,4))
plt.scatter(df['PM10'], df['PM2.5'])
plt.title("PM10 vs PM2.5")
plt.xlabel("PM10")
plt.ylabel("PM2.5")
plt.show()

#PM2.5 across Location Types
#Shows urban vs rural vs industrial impact
df.boxplot(column='PM2.5', by='Location_Type',figsize=(6,4))
plt.title("PM2.5 by Location Type")
plt.suptitle("")  # remove extra title
plt.xlabel("Location Type")
plt.ylabel("PM2.5")
plt.show()

#Null Values
print(df.isnull().sum())  
print("No null values in the dataset") 


#Encoding
categorical_cols = ['Location_Type', 'Source_Label']

le = LabelEncoder()
for col in ['Location_Type', 'Source_Label']:
    df[col] = le.fit_transform(df[col])

print("\nDataset after Label Encoding:")
print(df.head())

#Features and Target

#Target variable
y = df['PM2.5']  

#Feature variables (all other numeric columns except PM2.5)
features = df.columns.tolist()   #all columns
features.remove('PM2.5')        #remove target

X = df[features]  #Features for ANN

print("Features (X) columns:\n", X.columns)
print("\nTarget (y) column:\n", y.name)

#train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training and testing data ready")

#Scaling
scaler = StandardScaler()

#Fit on training features and transform both train and test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed")

#Build ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

#Train the model
history = model.fit(X_train_scaled, y_train, 
                    validation_split=0.2, 
                    epochs=100, 
                    batch_size=16, 
                    verbose=1)

#Evaluate on test data
loss, mae = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

#Model Save
model_json = model.to_json()
with open("PM25_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("PM25_model.h5")
print("Saved model to disk (architecture + weights)")




