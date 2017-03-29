import random as random
import numpy as np
from sklearn import linear_model

# Make up some under determined but sparse data
#    y = x_20 - .5 x_40 + noise
def getx():
    return 20.0 * random.random() - 10.0

num_coords = 100
num_data = 50
X_base = [[getx() for i in range(num_coords)] for j in range(num_data)]
y_base = [x[20] - 0.5 * x[40] + random.gauss(0.0, 1.0) for x in X_base]

X = np.array(X_base)
y = np.array(y_base)

# Now use lasso to find the sparse representation of data
clf = linear_model.Lasso(alpha = 0.1)
clf.fit(X,y)
w = clf.coef_

for i in range(len(w)):
    if abs(w[i]) > .1:
        print (i, w[i])
