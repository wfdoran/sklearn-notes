from sklearn import datasets 
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np

# read in some data
digits = datasets.load_digits()
X = digits.data
y = digits.target

# shuffle the data
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

# pick a training size
t = int(0.8 * len(X))

# pick possible values for gamma and C
params = [{'gamma' : [0.1, 0.01, 0.001, 0.0001], 'C' : [1, 10, 100, 1000]}]

# train a support vector machine, trying each combination of (gamma,C)
clf = GridSearchCV(svm.SVC(), params)
clf.fit(X[:t], y[:t])
print(clf.best_params_)

# how does it perform on test data?
z = clf.predict(X[t:])
tt = len(z)
comp = [z[i] == y[i+t] for i in range(tt)]
agree = sum(comp)
print(agree, tt, float(agree)/tt)