from sklearn import decomposition, datasets
import matplotlib.pyplot as plt

# load some sample data
digits = datasets.load_digits()
X = digits.data
y = digits.target

# reduce the data to 2 components
pca = decomposition.PCA(n_components=2)
X_features = pca.fit(X,None).transform(X)

# print the features correpsonding to 0
icon = ['bo', 'go', 'bs', 'co', 'mo', 'ro', 'gs', 'rs', 'cs', 'ms']
for i in range(len(X_features)):
    plt.plot(X_features[i,0], X_features[i,1], icon[y[i]])
plt.show()
