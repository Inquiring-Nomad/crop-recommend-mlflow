import pandas as pd

import pickle
import numpy as np


def predict(*args):
    features = [item for item in args]

    model = pickle.load(open("model.pkl", 'rb'))
    scaler = pickle.load(open("scaler/scaler.pkl", 'rb'))
    encoder = pickle.load(open("labelencoder/labelencoder.pkl", 'rb'))
    scaled_features = scaler.transform(np.array(features).reshape(1, -1))
    prediction = model.predict(scaled_features)
    return prediction
    return encoder.inverse_transform(prediction)[0]


df = pd.read_csv("../data/Crop_recommendation.csv")
df = df.iloc[17]
print(predict(df['N'],df['P'],df['K'],df['temperature'],df['humidity'],df['ph'],df['rainfall']))
