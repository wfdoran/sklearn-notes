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

print ("Should be same cluster:  %4d of %4d (%8.4f)" % (count1, count0, float(count1) / count0)) 
print ("Should be different:     %4d of %4d (%8.4f)" % (count3, count2, float(count3) / count2))
print()
print ("cluster centers:")
print (cluster.cluster_centers_)
