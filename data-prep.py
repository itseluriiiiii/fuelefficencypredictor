import os
import pandas as pd
import numpy as np

DATA_DIR = "data"
OUT_CSV = os.path.join(DATA_DIR, "processed_fuel.csv")


def load_auto_mpg():
    path = os.path.join(DATA_DIR, "auto-mpg.csv")
    if not os.path.exists(path):
        print("Auto MPG not found.")
        return None

    df = pd.read_csv(path, encoding="latin1")
    df["kmpl"] = df["mpg"] * 0.425
    df["engine_cc"] = df["displacement"]
    df["weight_kg"] = df["weight"] * 0.453
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    df["fuel_type"] = "Petrol"

    return df[["engine_cc", "horsepower", "weight_kg", "fuel_type", "kmpl"]]


def load_fuel_consumption_ratings():
    path = os.path.join(DATA_DIR, "Fuel_Consumption_Ratings_2023.csv")
    if not os.path.exists(path):
        print("Fuel Consumption Ratings not found.")
        return None

    df = pd.read_csv(path, encoding="latin1")
    col = [c for c in df.columns if "L/100" in c][0]
    df["kmpl"] = 100 / df[col]
    df["engine_cc"] = df.get("Engine Size(L)", np.nan) * 1000
    df["horsepower"] = df.get("Horsepower", np.nan)
    df["weight_kg"] = np.nan
    df["fuel_type"] = df.get("Fuel Type", "Petrol")

    return df[["engine_cc", "horsepower", "weight_kg", "fuel_type", "kmpl"]]


def load_epa():
    path = os.path.join(DATA_DIR, "vehicles.csv")
    if not os.path.exists(path):
        print("EPA vehicles data not found.")
        return None

    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    mpg_col = "comb08" if "comb08" in df.columns else None
    
    if mpg_col is None:
        print("EPA MPG column not found.")
        return None

    df["kmpl"] = df[mpg_col] * 0.425
    df["engine_cc"] = df.get("displ", np.nan) * 1000
    df["horsepower"] = df.get("hp", np.nan)
    df["weight_kg"] = df.get("wgt", np.nan)
    df["fuel_type"] = df.get("fuelType", "Petrol")

    return df[["engine_cc", "horsepower", "weight_kg", "fuel_type", "kmpl"]]


def add_synthetic_features(df):
    n = len(df)
    rng = np.random.default_rng(42)

    df["road_type"] = rng.choice(["city", "highway", "hilly"], size=n, p=[0.5, 0.4, 0.1])
    
    df["avg_speed_kmph"] = np.where(
        df["road_type"] == "city",
        rng.normal(35, 5, size=n),
        np.where(df["road_type"] == "highway",
                 rng.normal(90, 10, size=n),
                 rng.normal(50, 8, size=n))
    )
    
    df["ac_on"] = rng.choice([0, 1], size=n, p=[0.4, 0.6])
    df["tyre_psi"] = rng.normal(33, 2, size=n).round(1)
    df["traffic_level"] = rng.choice(["low", "medium", "high"], size=n, p=[0.4, 0.4, 0.2])
    df["payload_kg"] = rng.normal(120, 40, size=n).clip(0, 400)

    return df


if __name__ == "__main__":
    auto = load_auto_mpg()
    fcr = load_fuel_consumption_ratings()
    epa = load_epa()

    all_dfs = [df for df in [auto, fcr, epa] if df is not None]
    final = pd.concat(all_dfs, ignore_index=True)

    final["engine_cc"] = pd.to_numeric(final["engine_cc"], errors="coerce")
    final["horsepower"] = pd.to_numeric(final["horsepower"], errors="coerce")
    final["kmpl"] = pd.to_numeric(final["kmpl"], errors="coerce").fillna(final["kmpl"].median())

    final = add_synthetic_features(final)
    final.to_csv(OUT_CSV, index=False)
    
    print("Processed dataset saved to:", OUT_CSV)
    print(final.head())
