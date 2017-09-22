from sklearn.datasets import make_blobs
from matplotlib import pyplot
from sklearn.cluster import KMeans
import numpy as np

data, label = make_blobs(n_samples=100, n_features=2, centers=5)
# 绘制样本显示
#print(data)
#print(label)
pyplot.scatter(data[:, 0], data[:, 1], c=label)
pyplot.show()
kmeans = KMeans(n_clusters=5, random_state=0).fit(data)
centers=kmeans.cluster_centers_
label2=kmeans.predict(data)
pyplot.scatter(data[:, 0], data[:, 1], c=label2)

print(centers)

pyplot.show()
