import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

def train_model(X, Y):
    Y = np.asarray(Y).ravel()
    Y_noisy = Y + np.random.normal(0, 0.05, size=len(Y))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_noisy, test_size=0.2)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        verbosity=1,
        objective="reg:squarederror"
    )

    model.fit(X_train, Y_train)
    preds=model.predict(X_test)
    rmse= np.sqrt(mean_squared_error(Y_test, preds))
    print(f"Model trained. Test RMSE: {rmse:.2f}")
    return model


def save_model(model, path="tm_xgboost.model"):
    with open(path, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    X, Y = extract_data(file)

    model = train_model(X, Y)
    save_model(model)