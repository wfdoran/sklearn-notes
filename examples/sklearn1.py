from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt

# load some sample data
boston = datasets.load_boston()
X = boston.data
y = boston.target

# linear fit
model = linear_model.LinearRegression()
model.fit(X,y)
z = model.predict(X)

# plot 
plt.scatter(y,z, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='black')
plt.xlabel('Actual')
plt.ylabel('Model')
plt.show()

# train on half of the data, test on the other half
n = len(X) // 2
model.fit(X[:n], y[:n])
z = model.predict(X[n:])

plt.scatter(y[n:],z, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='black')
plt.xlabel('Actual')
plt.ylabel('Model')
plt.show()
