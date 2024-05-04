import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib


data = pd.read_csv("Crop_Dataset.csv")


numerical_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
data = data[numerical_features + ["Label_Encoded"]]


X = data.drop(columns=["Label_Encoded"])  
y = data["Label_Encoded"] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)