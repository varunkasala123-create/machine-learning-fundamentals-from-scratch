import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ----- Create synthetic dataset ----- #

np.random.seed(42)

n = 500

size = np.random.normal(1500, 400, n)
bedrooms = np.random.randint(1, 6, n)
age = np.random.randint(0, 50, n)
location = np.random.uniform(1, 10, n)

price = (
    size * 200 +
    bedrooms * 15000 -
    age * 800 +
    location * 30000 +
    np.random.normal(0, 20000, n)
)

df = pd.DataFrame({
    "size": size,
    "bedrooms": bedrooms,
    "age": age,
    "location": location,
    "price": price
})

# ----- Train / Test split ----- #

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----- Train model (Overfitting experiment) ----- #

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

model = make_pipeline(
    PolynomialFeatures(degree=5),
    Ridge(alpha=10.0)
)

model.fit(X_train, y_train)


# ----- Evaluate ----- #

preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)

print("RMSE:", int(rmse))

# ----- Plot errors ----- #

plt.scatter(y_test, preds)
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual")
plt.show()

# ----- Coefficients ----- #

print("\nModel coefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(name, ":", int(coef))
