import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Calculates the estimated probability of y using the logistic function
def y_probF(x):
    y_prob = 1 / (1 + np.exp(-x))
    return y_prob


# Calculates the cost function
def costF(y_train, X_train, theta):
    cost = -np.mean((y_train * np.log(y_probF(np.dot(X_train, theta)))) + ((1 - y_train) * np.log(1 - y_probF(np.dot(X_train, theta)))))
    return cost


# Minimizes the cost function
def minimize(T, L, y_train, X_train, theta):
    tolerance = 1
    cost = costF(y_train, X_train, theta)
    while tolerance > T:
        oldCost = cost
        theta = theta - L * np.mean(np.dot(y_probF(np.dot(X_train, theta)) - y_train, X_train))
        cost = costF(y_train, X_train, theta)
        tolerance = abs(oldCost - cost)
    return theta


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target

    theta = np.zeros(X.shape[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    theta = minimize(0.001, 0.01, y_train, X_train, theta)
    y_prob = y_probF(np.dot(X_test, theta))
    y_pred = np.round(y_prob)

    print(np.mean(y_pred == y_test))

    # model = LogisticRegression()
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    #
    # print(np.mean(y_pred == y_test))
