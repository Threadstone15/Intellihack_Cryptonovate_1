import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib


data = pd.read_csv("Crop_Dataset.csv")


parameters = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
data = data[parameters + ["Label_Encoded"]]


X = data.drop(columns=["Label_Encoded"])  
y = data["Label_Encoded"] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = SVC(kernel='linear', C=1.0, probability=True)
model.fit(X_train_scaled, y_train)


y_prediction = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_prediction)
print("Accuracy:", accuracy)


joblib.dump(model, "crop_recommendation_model.joblib")

def predict_most_suitable_crops():
    N = float(input("What is the Nitrogen level (N) : "))
    P = float(input("What is the Phosphorus level (P): "))
    K = float(input("What is the Pottasium level (K): "))
    temperature = float(input("What is the temperature in (°C): "))
    humidity = float(input("What is the humidity as a percentage: "))
    ph = float(input("What is the pH value: "))
    rainfall = float(input("What is the rainfall level in (cm): "))

    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=parameters)
    input_data_scaled = scaler.transform(input_data)
    probabilities = model.predict_proba(input_data_scaled)[0]  # Get probabilities for each class
    probabilities_dict = {get_name(i): prob for i, prob in enumerate(probabilities)}
    sorted_probabilities = sorted(probabilities_dict.items(), key=lambda x: x[1], reverse=True)  # Sort probabilities

    # Select the top three crops
    top_three_crops = [crop for crop, prob in sorted_probabilities[:3]]
    return top_three_crops


def get_name(key):
    my_dict = {
        0: "wheat",
        1: "barley",
        2: "lettuce",
        3: "spinach",
        4: "cauliflower",
        5: "brussels_sprouts",
        6: "cabbage",
        7: "beans",
        8: "peas",
        9: "turnips",
        10: "carrots",
        11: "beets",
        12: "cherries",
        13: "plums",
        14: "raspberries",
        15: "pears",
        16: "blackcurrants",
        17: "strawberries",
        18: "apples",
        19: "potatoes",
        20: "rapeseed",
        21: "tomatoes"
    }
    
    return my_dict[key]


predicted_crops = predict_most_suitable_crops()
print("Top three predicted crops:", predicted_crops)