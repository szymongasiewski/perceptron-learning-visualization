import numpy as np
import time
import os

class Perceptron:
    def __init__(self, learning_rate, iter, function):
        self.lr = learning_rate
        self.iter = iter
        self.weights = None
        self.bias = None
        self.function = function

    def train(self, x, y):
        self.weights = np.random.rand(x.shape[1])
        self.bias = 0

        for iter in range(self.iter):
            for i in range(x.shape[0]):
                y_pred = self.activation_func(np.dot(self.weights, x[i]) + self.bias)

                self.weights = self.weights + self.lr * (y[i] - y_pred) * x[i]
                self.bias = self.bias + self.lr * (y[i] - y_pred)
                
            print("Weights:")
            print(*np.round(self.weights, 2), sep = " | ")
            print(f"BIAS: {self.bias}")
            time.sleep(0.1)
            os.system("cls")

    def activation_func(self, x):
        #binary step
        if self.function == 1:
            if x >= 0:
                return 1
            else:
                return 0

        #sigmoid
        if self.function == 2:
            return 1/(1+np.exp(-x))

        #tanh
        if self.function == 3:
            return np.tanh(x)

        #relu
        if self.function >= 4:
            return np.maximum(0, x)

    def predict(self, x):
        y_pred = []

        for i in range(x.shape[0]):
            y_pred.append(self.activation_func(np.dot(self.weights, x[i]) + self.bias))

        return np.array(y_pred)      
        