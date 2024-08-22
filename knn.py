import numpy as np


class KNN:
    def __init__(self, n_neighbors = 5):
        self.n_neighbors = n_neighbors
        self.X = None
        self.y = None
    
    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, x):
        dists = []
        neighbors = []
        
        # Calcular distância euclidiana
        for X in self.X:
            dists.append(np.linalg.norm(X - x))
        
        # Selecionar os K vizinhos
        for i in range(self.n_neighbors):
            m = dists.index(min(dists))
            neighbors.append(self.y[m])
            dists[m] = max(dists) + 1
        
        # Calcular moda
        vals, counts = np.unique(neighbors, return_counts=True)
        index = np.argmax(counts)
        return vals[index]
