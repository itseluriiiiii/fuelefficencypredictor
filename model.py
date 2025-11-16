import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

DATA_CSV = "data/processed_fuel.csv"
MODEL_OUT = "model.joblib"
FEATURES_FILE = "feature_columns.json"

def load():
    df = pd.read_csv(DATA_CSV)
    df = df.dropna(subset=['kmpl'])
    return df

def prepare(df):
    y = df['kmpl'].astype(float)
    X = df[['engine_cc','horsepower','weight_kg','road_type','avg_speed_kmph','ac_on','tyre_psi','traffic_level','payload_kg']].copy()
    X['engine_cc'] = X['engine_cc'].fillna(X['engine_cc'].median())
    X['horsepower'] = X['horsepower'].fillna(X['horsepower'].median())
    X['weight_kg'] = X['weight_kg'].fillna(X['weight_kg'].median())

    cat_cols = ['road_type','traffic_level','ac_on'] 
    num_cols = [c for c in X.columns if c not in cat_cols]

    return X, y, num_cols, cat_cols

def train():
    df = load()
    X, y, num_cols, cat_cols = prepare(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ], 
        remainder='drop'
    )

    model = Pipeline([
        ('preproc', preprocessor),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    print("Training model...")
    model.fit(X_train, y_train)
    
    print("Predicting on test set...")
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"MAE = {mae:.3f} kmpl, RMSE = {rmse:.3f} kmpl")

    joblib.dump(model, MODEL_OUT)
    print("Saved model to", MODEL_OUT)

    feature_info = {
        "num_cols": num_cols,
        "cat_cols": cat_cols
    }
    
    with open(FEATURES_FILE, "w") as f:
        json.dump(feature_info, f)
    print("Saved feature info to", FEATURES_FILE)


if __name__ == "__main__":
    train()
