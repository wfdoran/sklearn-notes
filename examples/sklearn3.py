from sklearn import datasets 
from sklearn import svm
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

# train a support vector machine
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(X[:t], y[:t])

# how does it perform on test data?
z = clf.predict(X[t:])
tt = len(z)
comp = [z[i] == y[i+t] for i in range(tt)]
agree = sum(comp)
print(agree, tt, float(agree)/tt)
