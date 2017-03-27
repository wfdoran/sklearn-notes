from sklearn import datasets
from sklearn.cluster import KMeans

# read in some data
iris = datasets.load_iris()
X = iris.data
y = iris.target

cluster = KMeans(n_clusters=3)
z = cluster.fit_predict(X) 

count0 = 0
count1 = 0
count2 = 0
count3 = 0

for i in range(len(y)):
    for j in range(i+1, len(y)):
        if y[i] == y[j]:
            count0 += 1
            if z[i] == z[j]:
                count1 += 1
        else:
            count2 += 1
            if z[i] != z[j]:
                count3 += 1

print "%4d %4d %8.4f" % (count0, count1, float(count1) / count0) 
print "%4d %4d %8.4f" % (count2, count3, float(count3) / count2) 

print cluster.cluster_centers_
