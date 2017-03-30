import numpy as np
import random as random
from sklearn import preprocessing
from sklearn import linear_model

def LogisticLearn(X, y):
    classifier = linear_model.LogisticRegression()
    train_size = int(0.8 * len(X))
    classifier.fit(X[:train_size], y[:train_size])
    z = classifier.predict(X[train_size:])
    test_size = len(z)
    comp = [z[i] == y[train_size:][i] for i in range(test_size)]
    agree = sum(comp)
    print (agree, test_size, float(agree) / test_size)    

def getx():
    return 20.0 * random.random() - 10.0

    
num_data = 10000
X_hidden = [[getx() for i in range(6)] for j in range(num_data)]
y_temp = [x[0] + x[1] - x[2] + x[3] - x[5] + random.gauss(0.0, 1.0) for x in X_hidden]
y_base = [1 if v >= 0 else 0 for v in y_temp]

# Convert these to numpy arrays for use with sklearn
X = np.array(X_hidden)
y = np.array(y_base)

# Try as is
print("original data:   ", end="")
LogisticLearn(X, y)

# skew the data
mult = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
offset = [0.0, -5.0, 50.0, -100.0, 1000.0, -50000.0]

X_skew = [[x[j] * mult[j] + offset[j] for j in range(6)] for x in X_hidden]
X = np.array(X_skew)
print("skewed data:     ", end="")
LogisticLearn(X, y)

# normalize the data
print("normalized data: ", end="")
X = preprocessing.scale(X)
LogisticLearn(X, y)


    