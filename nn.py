import numpy as np

# The input data is X, which is n by d (i.e. n data points with d features)


# The basic block class
class Block:
    def forward(self, x):
        pass

    def backward(self, grad):
        pass

    def update(self, learning_rate):
        pass


# The linear block
class Linear(Block):
    def __init__(self, input_dim, output_dim):
        # We use He initialisation
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        self.b = np.zeros(output_dim)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        # Save the W and b gradients
        self.dL_dW = self.x.T @ grad
        self.dL_db = np.sum(grad, axis=0)

        # Return the x gradient
        return grad @ self.W.T

    # Update the parameters W and b using gradient descent
    def update(self, learning_rate):
        self.W -= learning_rate * self.dL_dW
        self.b -= learning_rate * self.dL_db


# The ReLU block
class ReLU(Block):
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, grad):
        return grad * (self.x > 0)


# The sigmoid block
class Sigmoid(Block):

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, x):
        self.y = self.sigmoid(x)
        return self.y

    def backward(self, grad):
        return grad * self.y * (1 - self.y)


# Mean Squared Error (our chosen loss function for regression)
class MSE:
    def forward(self, y_pred, y):
        # Save for later use
        self.y_pred = y_pred
        self.y = y

        return np.mean((y_pred - y) ** 2)

    def backward(self):
        return (2 / self.y.size) * (self.y_pred - self.y)


# The network class
class Network:
    def __init__(self, blocks):
        self.blocks = blocks

    def forward(self, x):
        for block in self.blocks:
            x = block.forward(x)

        return x

    def backward(self, grad):
        for block in reversed(self.blocks):
            grad = block.backward(grad)

        return grad

    def update(self, learning_rate):
        for block in self.blocks:
            block.update(learning_rate)
