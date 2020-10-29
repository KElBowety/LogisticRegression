import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer


# Calculates the estimated probability of y using the logistic function
# Takes parameter x for the exponent in the logistic function
def y_probF(x):
    y_prob = 1 / (1 + np.exp(-x))
    return y_prob


# Calculates the cost function
# Takes parameters y_train and X_train from the dataset
# Takes parameter theta, the features array
def costF(y_train, X_train, theta):
    cost = -np.mean((y_train * np.log(y_probF(np.dot(X_train, theta)))) + ((1 - y_train) * np.log(1 - y_probF(np.dot(X_train, theta)))))
    return cost


# Minimizes the cost function
# Takes parameters tolerance, learning rate, y_train, X_train, and theta (features array)
def minimize(T, L, y_train, X_train, theta):
    tolerance = 1
    cost = costF(y_train, X_train, theta)
    while tolerance > T:
        oldCost = cost
        theta = theta - L * (np.dot(y_probF(np.dot(X_train, theta)) - y_train, X_train))
        cost = costF(y_train, X_train, theta)
        tolerance = abs(oldCost - cost)
    return theta


# Preprocessing for the One vs All multiclass classificati0on
# Sets one class to 1 and all others to 0
# Takes parameters y_train and one, which is the class that will be set to one
def oneVAPP(y_train, one):
    newy = np.array(y_train)
    for y in range(0, len(newy)):
        if newy[y] == one:
            newy[y] = 1
        else:
            newy[y] = 0
    return newy


# Preprocessing for the One vs One multiclass classificati0on
# Compares each two classes together
# Takes parameters X_train, y_train, first class, and second class
def oneVOPP(X_train, y_train, class1, class2):
    a = []
    newX_train = np.array(a)
    newy_train = np.array(a)
    n = 0
    for i in range(0, len(X_train)):
        if y_train[i] == class1 or y_train[i] == class2:
            newX_train = np.append(newX_train, X_train[i])
            newy_train = np.append(newy_train, y_train[i])
            n += 1
    return newX_train, newy_train


# Function for One vs All classification
def oneVAll(X_train, y_train, n, T, L):
    y_prob = []
    for i in range(0, n):
        theta = np.zeros(X.shape[1])
        y_train1 = oneVAPP(y_train, i)
        theta = minimize(T, L, y_train1, X_train, theta)
        y_prob.append(y_probF(np.dot(X_test, theta)))
    return y_prob


# Function for One vs One classification
def oneVOne(X_train, y_train, n, T, L):
    y_prob = dict()
    for i in range(0, n):
        for j in range(i + 1, n):
            newX_train, newy_train = oneVOPP(X_train, y_train, i, j)
            y_prob[str(i)+str(j)] = [newX_train, newy_train]
    return y_prob


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target

# Normalizing the data set to stop division by zero errors
    norm = Normalizer()
    norm.fit(X)
    X = norm.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    oneVOPP(X_train, y_train, 0, 1)

    y_prob = oneVAll(X_train, y_train, 3, 0.00001, 0.01)
    y_pred = []

# This loop finds the probable class and adds to y_pred
    for i in range(0, len(y_prob[0])):
        if (y_prob[0][i] >= y_prob[1][i]) and (y_prob[0][i] >= y_prob[2][i]):
            y_pred.append(0)
        elif (y_prob[1][i] >= y_prob[0][i]) and (y_prob[1][i] >= y_prob[2][i]):
            y_pred.append(1)
        else:
            y_pred.append(2)

    print('OvA Accuracy: ' + str(np.mean(y_pred == y_test)))

    y_prob = oneVOne(X_train, y_train, 3, 0.00001, 0.01)
    y_pred = []

# This loop finds the probable class and adds to y_pred
    for i in range(0, len(y_test)):
        class0 = 0
        class1 = 0
        class2 = 0
        if y_prob.get('01')[0][i] >= 0.5:
            class0 += 1
        if y_prob.get('02')[0][i] >= 0.5:
            class0 += 1
        if (1 - y_prob.get('01')[0][i]) >= 0.5:
            class1 += 1
        if y_prob.get('12')[0][i] >= 0.5:
            class1 += 1
        if (1 - y_prob.get('02')[0][i]) >= 0.5:
            class2 += 1
        if (1 - y_prob.get('12')[0][i]) >= 0.5:
            class2 += 1
        y_pred.append(max(class0, class1, class2))

    print('OvO Accuracy: ' + str(np.mean(y_pred == y_test)))
