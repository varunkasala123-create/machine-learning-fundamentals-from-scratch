import numpy as np
import matplotlib.pyplot as plt

# ---------- Generate Data ---------- #

np.random.seed(0)
N = 500

size = np.random.normal(1500, 400, N)
bedrooms = np.random.randint(1, 6, N)
age = np.random.randint(0, 50, N)
location = np.random.uniform(1, 10, N)

X = np.column_stack([size, bedrooms, age, location])

true_w = np.array([200, 15000, -800, 30000])
true_b = 10000

noise = np.random.normal(0, 20000, N)

y = X @ true_w + true_b + noise

# ---------- Normalize Features ---------- #

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

X = (X - X_mean) / X_std

# ---------- Init Weights ---------- #

w = np.zeros(4)
b = 0

lr = 0.5
epochs = 200

losses = []

# ---------- Training Loop ---------- #

for epoch in range(epochs):

    y_pred = X @ w + b

    error = y_pred - y

    loss = np.mean(error ** 2)
    losses.append(loss)

    dw = (2 / N) * X.T @ error
    db = (2 / N) * np.sum(error)

    w -= lr * dw
    b -= lr * db

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.2e}")

# ---------- Results ---------- #

print("\nLearned weights:", w)
print("Bias:", b)

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Curve")
plt.show()
