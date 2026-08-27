import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Sample training data
np.random.seed(42)

n = 1000

year = np.random.randint(2010, 2027, n)
km = np.random.randint(5000, 200000, n)
present_price = np.random.uniform(3, 50, n)
owners = np.random.randint(0, 4, n)

fuel = np.random.randint(0, 3, n)
transmission = np.random.randint(0, 2, n)

current_year = 2026
car_age = current_year - year

# Generate sample target prices
selling_price = (
    present_price * 0.75
    - car_age * 0.30
    - (km / 100000) * 1.2
    - owners * 0.30
    + fuel * 0.20
    + transmission * 0.50
)

selling_price += np.random.normal(0, 0.8, n)

selling_price = np.maximum(selling_price, 0.5)

# Dataset
data = pd.DataFrame({
    "Year": year,
    "KM_Driven": km,
    "Present_Price": present_price,
    "Fuel_Type": fuel,
    "Transmission": transmission,
    "Owners": owners,
    "Selling_Price": selling_price
})

# Input and Output
X = data[
    [
        "Year",
        "KM_Driven",
        "Present_Price",
        "Fuel_Type",
        "Transmission",
        "Owners"
    ]
]

y = data["Selling_Price"]

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# Save model as Pickle file
with open("car_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained successfully!")
print("car_price_model.pkl created successfully!")