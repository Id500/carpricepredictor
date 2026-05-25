import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Load dataset
data = pd.read_csv("quikr_car.csv")

# Remove rows with missing values
data = data.dropna()

# Remove "Ask For Price"
data = data[data["Price"] != "Ask For Price"]

# Clean price column
data["Price"] = data["Price"].str.replace(",", "")
data["Price"] = data["Price"].astype(int)

# Clean kms_driven column
data["kms_driven"] = data["kms_driven"].str.replace(" kms", "")
data["kms_driven"] = data["kms_driven"].str.replace(",", "")

# Keep only numeric kms
data = data[data["kms_driven"].str.isnumeric()]

data["kms_driven"] = data["kms_driven"].astype(int)

# Keep valid years only
data = data[data["year"].astype(str).str.isnumeric()]
data["year"] = data["year"].astype(int)

# Features
X = data[["name", "company", "year", "kms_driven", "fuel_type"]]

# Target
y = data["Price"]

# Convert text columns automatically
ohe = OneHotEncoder(handle_unknown="ignore")

column_transformer = ColumnTransformer([
    ("ohe", ohe, ["name", "company", "fuel_type"])
], remainder="passthrough")

# Model pipeline
model = Pipeline([
    ("transformer", column_transformer),
    ("model", LinearRegression())
])

# Train model
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained successfully!")