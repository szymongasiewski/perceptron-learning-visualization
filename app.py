import pandas as pd
import numpy as np
from perceptron import Perceptron
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys

while True:
    os.system("cls")

    print("MENU")
    print("1. START")
    print("2. EXIT")

    decision = int(input("Your choice: "))

    os.system("cls")

    if decision >= 2: sys.exit()

    print("1. Generate random dataset")
    print("2. Load dataset")

    decision = int(input("Your choice: "))

    os.system("cls")

    if decision == 1:
        samples = int(input("Number of samples: "))
        features = int(input("Number of features: "))
        X, y = datasets.make_blobs(n_samples = samples, n_features = features, centers = 2, cluster_std = 1.05, random_state = 2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 123)

    if decision == 2:
        path = input("Paste path to csv file: ")
        data = pd.read_csv(path)
        X = data.iloc[:,1:].values
        y = data. iloc[:,0]. values
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state = 123)

    os.system("cls")

    learning_rate = float(input("Set learning rate: "))
    iterations = int(input("Set number of iterations: "))

    os.system("cls")

    print("Choose activation function")
    print("1. Binary step")
    print("2. Sigmoid")
    print("3. Tanh")
    print("4. Relu")
    
    activacion_function = int(input("Your choice: "))

    os.system("cls")

    p = Perceptron(learning_rate=learning_rate, iter=iterations, function=activacion_function)

    p.train(X_train,y_train)

    os.system("cls")

    print("Final weights: ")
    print(*np.round(p.weights, 2), sep = " | ")
    print(f"Final bias: {p.bias}")
    print(f"Learning rate: {p.lr}")
    print(f"Iterations: {p.iter}")

    if activacion_function == 1:
        print("Activation function: Binary step")

    if activacion_function == 2:
        print("Activation function: Sigmoid")

    if activacion_function == 3:
        print("Activation function: Tanh")

    if activacion_function == 4:
        print("Activation function: Relu")

    y_pred = p.predict(X_test)

    print(f"Accuracy score: {100 * np.round(accuracy_score(y_test, np.round(y_pred)), 2)}%")

    if X.shape[1] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X_train[:,0], X_train[:,1], marker = 'o', c = y_train)

        x0_1 = np.amin(X_train[:,0])
        x0_2 = np.amax(X_train[:,0])

        x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
        x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

        ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')

        ymin = np.amin(X_train[:,1])
        ymax = np.amax(X_train[:,1])

        ax.set_ylim([ymin - 3, ymax + 3])

        plt.show()
    
    input("Press ENTER and go back to menu")



