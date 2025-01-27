import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 1)
        self.b2 = np.zeros(1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.pre_h = np.dot(X, self.W1) + self.b1
        self.post_h = relu(self.pre_h)
        self.pre_out = np.dot(self.post_h, self.W2) + self.b2
        self.post_out = sigmoid(self.pre_out)
        return self.post_out

    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        m = y.shape[0]
        loss_grads = (self.post_out - y) * sigmoid_derivative(self.post_out)
        W2_grads = (1./m) * np.dot(self.post_h.T, loss_grads)
        b2_grads = (1./m) * np.sum(loss_grads, axis=0)
        post_h_grads = np.dot(loss_grads, self.W2.T) * relu_derivative(self.post_h)
        W1_grads = (1./m) * np.dot(X.T, post_h_grads)
        b1_grads = (1./m) * np.sum(post_h_grads, axis=0)
        return W1_grads, b1_grads, W2_grads, b2_grads
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, learning_rate: float = 0.01, printout: bool = True) -> None:
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = binary_cross_entropy(y, y_pred)
            dW1, db1, dW2, db2 = self.backward(X, y)
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            
            if epoch % 1000 == 0:
                if printout:
                    print(f"Epoch {epoch}, Loss {loss}, Accuracy: {np.mean((y_pred > 0.5) == y)}")

if __name__ == "__main__":
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_val = np.load("X_val.npy")
    y_val = np.load("y_val.npy")

    nn = NeuralNetwork(input_size=10, hidden_size=10)
    nn.train(X_train, y_train, epochs=5000)

    y_pred = nn.forward(X_val)
    loss = binary_cross_entropy(y_val, y_pred)
    print(f"validation loss: {loss}")
    print(f"Validation accuracy: {np.mean((y_pred > 0.5) == y_val)}")

import numpy as np

def read(file_path):
    with open(file_path, 'r') as file:
        n, m = map(int, file.readline().split())
    
    features = []
    for i in range(m):
        features.append(list(map(float, file.readline().split())))
    
    labels = list(map(float, file.readline().split()))
    weights = list(map(float, file.readline().split()))
    bias = float(file.readline())
    
    return np.array(features), np.array(labels), np.array(weights), bias, n
      
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def neuron(features, labels, weights, bias, n):
    z = np.dot(features, weights) + bias
    pred = sigmoid(z)
    
    cum_sum = 0
    for i in range(len(labels)):
        cum_sum = cum_sum + ((pred[i] - labels[i]) ** 2)
    
    mse = cum_sum / n 
    
    pred = np.round(pred, 2)
    # mse = round(mse, 2)
    
    return pred, mse


# file_path = "demo.txt"
# features, labels, weights, bias = read(file_path)
features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]]
labels = [0, 1, 0]
weights = [0.7, -0.4]
bias = -0.1
n = 2

print(neuron(features, labels, weights, bias, n))