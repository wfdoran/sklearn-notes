from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

X,y = make_blobs(n_samples=5000, n_features=2, centers=2, random_state=0)

clf = RandomForestClassifier(n_estimators=3)
train = int(.8 * len(X))
clf = clf.fit(X[:train], y[:train])

z = clf.predict(X[:train])
agree = [y[i] == z[i] for i in range(train)]

print("On training data: %4d %4d %8.4f" % (sum(agree), train, sum(agree)/train))

z = clf.predict(X[train:])

agree = [y[train + i] == z[i] for i in range(len(z))]
print("On testing data:  %4d %4d %8.4f" % (sum(agree), len(z), sum(agree)/len(z)))

icon = ['bo', 'ro', 'rx', 'bx']
for i in range(len(z)):
    if z[i] == y[i+train]:
        plt.plot(X[i+train][0], X[i+train][1], icon[z[i]])
    else:
        plt.plot(X[i+train][0], X[i+train][1], icon[2+z[i]])
plt.show()
