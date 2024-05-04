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


model = SVC(kernel='linear', C=1.0)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


joblib.dump(model, "crop_recommendation_model.joblib")

def predict_crop():
    N = float(input("Enter the value for 'N' (Nitrogen): "))
    P = float(input("Enter the value for 'P' (Phosphorus): "))
    K = float(input("Enter the value for 'K' (Potassium): "))
    temperature = float(input("Enter the value for 'temperature': "))
    humidity = float(input("Enter the value for 'humidity': "))
    ph = float(input("Enter the value for 'ph': "))
    rainfall = float(input("Enter the value for 'rainfall': "))

    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=numerical_features)
    input_data_scaled = scaler.transform(input_data)
    predicted_crop = model.predict(input_data_scaled)[0]
    return predicted_crop

# Predict crop based on user input
predicted_crop = predict_crop()
print("Predicted crop:", predicted_crop)