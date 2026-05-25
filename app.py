from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load dataset
car = pd.read_csv("quikr_car.csv")

# Load trained model
model = joblib.load("model.pkl")

# Dropdown values
companies = sorted(car["company"].dropna().unique())
fuel_types = sorted(car["fuel_type"].dropna().unique())
car_models = sorted(car["name"].dropna().unique())

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = ""

    if request.method == "POST":

        # Get form data
        name = request.form["name"]
        company = request.form["company"]
        year = int(name.split()[0])   # Take year from model name
        kms_driven = int(request.form["kms_driven"])
        fuel_type = request.form["fuel_type"]

        # Create dataframe
        input_data = pd.DataFrame(
            [[name, company, year, kms_driven, fuel_type]],
            columns=[
                "name",
                "company",
                "year",
                "kms_driven",
                "fuel_type"
            ]
        )

        # Predict price
        price = model.predict(input_data)[0]

        prediction = f"Estimated Price: ₹ {int(price)}"

    return render_template(
        "index.html",
        companies=companies,
        fuel_types=fuel_types,
        car_models=car_models,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)