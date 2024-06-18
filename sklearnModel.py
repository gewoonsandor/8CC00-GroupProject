# Imports
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm

import defaults
from sklearn.model_selection import train_test_split


def run_model(inhibitor):
    x = df[defaults.get_descriptors(inhibitor)]
    y = df[inhibitor].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100))
    ])

    # Train model
    model.fit(x_train, y_train)

    # Make prediction using trained model
    predictions = model.predict(x_test)
    rounded_predictions = (predictions >= 0.5).astype(int)
    defaults.calculate_scores(rounded_predictions, y_test, inhibitor)

    with open(f"./models/model_{inhibitor}.pkl", "wb") as f:
        f.write(pickle.dumps(model))


if __name__ == "__main__":
    df = pd.read_csv('./datasets/prepared_data.csv')
    for inhibitor in defaults.y_data:
        run_model(inhibitor)
