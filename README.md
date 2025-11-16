# 🚗 Vehicle Fuel Efficiency Predictor
Machine Learning Project using Kaggle Datasets + Synthetic Driving Context. This project predicts vehicle fuel efficiency (km/L) using a combination of real-world vehicle specifications and synthetic driving-condition features.
It merges multiple Kaggle datasets, cleans them, converts units, standardizes fields, and generates a unified dataset suitable for modern machine learning models.

# 🔍 Overview
ML models you can train with the final dataset include:
- Linear Regression
- Random Forest Regression
- XGBoost
- Neural Networks

# 📊 Datasets Used
Place all raw dataset files.
- Auto MPG Dataset	-auto-mpg.csv	(mpg→kmpl conversion, engine size, horsepower)
- Fuel Consumption Ratings (Canada)	- Fuel_Consumption_Ratings.csv	(L/100km→kmpl conversion)
- EPA Vehicles Dataset -	vehicles.csv	(Combined MPG→kmpl + engine & horsepower)
All datasets are publicly available on Kaggle.

# 📁 Project Structure

```project/
├── data/
│   ├── auto-mpg.csv
│   ├── Fuel_Consumption_Ratings.csv
│   ├── vehicles.csv
│   └── processed_fuel.csv   # output
│
└── data-prep.py
└── app.py
└── model.py
````

# ⚙️ Installation
- Install dependencies:
```python -m pip install -r requirements.txt```
- Runing the app
  ```streamlit run app.py```
