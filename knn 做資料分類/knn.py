#顯示利用knn分類後的indices與sparse graph
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import numpy as np
X = np.random.randint(0,10,size=[10,2])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
y = nbrs.kneighbors_graph(X).toarray()
print("indices\n",indices)
print("distances\n",distances)
print("sparse graph\n",y)