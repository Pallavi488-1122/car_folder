from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("car_prediction.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from form
    year = int(request.form["year"])
    km = float(request.form["km"])
    present_price = float(request.form["present_price"])
    fuel = request.form["fuel"]
    transmission = request.form["transmission"]
    owners = int(request.form["owners"])

    # Convert Fuel Type into numbers
    fuel_dict = {
        "Petrol": 0,
        "Diesel": 1,
        "CNG": 2
    }

    fuel_value = fuel_dict[fuel]

    # Convert Transmission into numbers
    transmission_dict = {
        "Manual": 0,
        "Automatic": 1
    }

    transmission_value = transmission_dict[transmission]

    # Create input data
    input_data = np.array([
        [
            year,
            km,
            present_price,
            fuel_value,
            transmission_value,
            owners
        ]
    ])

    # Predict price
    prediction = model.predict(input_data)[0]

    # Minimum price
    prediction = max(prediction, 0.5)

    return render_template(
        "result.html",
        prediction=round(prediction, 2),
        year=year,
        km=int(km),
        fuel=fuel,
        transmission=transmission,
        owners=owners
    )


if __name__ == "__main__":
    app.run(debug=True)