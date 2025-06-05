from sklearn.neighbors import kneighbors_graph
import numpy as np

for name in ['mushroom', 'zoo', 'covertype']:
    X = np.load(f"data/Paper_datasets/{name}.npy")
    for k in [10, 20, 30]:
        A = kneighbors_graph(
            X, n_neighbors=k, mode='connectivity', include_self=False)
        A = A.toarray()
        np.save(f"data/Paper_datasets/{name}_knn_adj_k{k}.npy", A)


# import os
# print(os.getcwd())
