import numpy as np
import matplotlib.pyplot as plt

from nn import Network, Linear, ReLU, MSE

# ========================================
# NOISY SINE CURVE
# ========================================

# Fix seed for reproducibility
np.random.seed(42)

n = 200

# Original x-values
x = np.linspace(-2 * np.pi, 2 * np.pi, n).reshape(-1, 1)

# Normalise inputs to [-1, 1]
X = x / (2 * np.pi)

# Noisy sine data
y = np.sin(x) + 0.1 * np.random.randn(n, 1)

# The network (2 hidden layers with 64 neurons)
network = Network(
    [
        Linear(1, 32),
        ReLU(),
        Linear(32, 32),
        ReLU(),
        Linear(32, 1),
    ]
)

# The loss function
loss_fn = MSE()

epochs = 20000
learning_rate = 0.01

# The training loop
for i in range(epochs):
    # The forward pass
    y_pred = network.forward(X)

    # Calculate the loss
    loss = loss_fn.forward(y_pred, y)

    # Backpropagate through the loss
    grad = loss_fn.backward()

    # Backpropagate through the network
    network.backward(grad)

    # Update the parameters
    network.update(learning_rate)

    # Keep track of the loss
    if i % 100 == 0:
        print(f"Epoch {i}: Loss = {loss:.6f}")

# ========================================
# PLOT THE DATA
# ========================================

# Get the trained network's predictions
y_pred = network.forward(X)

plt.scatter(x[:, 0], y[:, 0], s=15, alpha=0.6, label="Noisy data")
plt.plot(x[:, 0], np.sin(x[:, 0]), linewidth=2, label="True sine")
plt.plot(x[:, 0], y_pred[:, 0], linewidth=2, label="Neural network")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Neural Network Fitting a Noisy Sine Curve")
plt.legend()
plt.grid(alpha=0.3)

plt.show()
