import random as random
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

def logistic_train(X, y):
    num_data = len(X)
    t = int(0.8 * num_data)

    clf = linear_model.LogisticRegression()
    clf.fit(X[:t], y[:t])
    z = clf.predict(X[t:])

    tt = len(z)
    comp = [z[i] == y[i+t] for i in range(tt)]
    agree = sum(comp)
    print("%6d %6d %12.4f" % (agree, tt, float(agree)/tt))
    

def getx():
    return 20.0 * random.random() - 10.0

# make up some data 
num_data = 10000
X_base = [[getx() for i in range(6)] for j in range(num_data)]

# y = x1 x4 + x0 x5 + x2 + noise
temp = [x[1] * x[4] + x[1] * x[5] + x[2] + random.gauss(0.0, 1.0) for x in X_base]
y_base = [1 if v >= 0 else 0 for v in temp]

X = np.array(X_base)
y = np.array(y_base)

# train on the original data
logistic_train(X,y)

# add in quadratatic terms
poly = PolynomialFeatures(2)
Xexpand = poly.fit_transform(X)
logistic_train(Xexpand, y)

