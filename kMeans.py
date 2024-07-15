import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = []

    def initialise(self, X_train):
        """
        Initialize the self.centroids class variable, using the "k-means++" method, 
        Pick a random data point as the first centroid,
        Pick the next centroids with probability directly proportional to their distance from the closest centroid
        Function returns self.centroids as an np.array
        USE np.random for any random number generation that you may require 
        (Generate no more than K random numbers). 
        Do NOT use the random module at ALL!
        """
        # Initialize centroids using k-means++ initialization
        self.centroids = [X_train[np.random.choice(len(X_train))]]
        while len(self.centroids) < self.n_clusters:
            dist_sq = np.array([min([np.linalg.norm(x - c)**2 for c in self.centroids]) for x in X_train])
            probs = dist_sq / dist_sq.sum()
            cum_probs = probs.cumsum()
            r = np.random.rand()
            for j, p in enumerate(cum_probs):
                if r < p:
                    i = j
                    break
            self.centroids.append(X_train[i])
        return np.array(self.centroids)

    def fit(self, X_train):
        """
        Updates the self.centroids class variable using the two-step iterative algorithm on the X_train dataset.
        X_train has dimensions (N,d) where N is the number of samples and each point belongs to d dimensions
        Ensure that the total number of iterations does not exceed self.max_iter
        Function returns self.centroids as an np array
        """
        # TODO
        for _ in range(self.max_iter):
            # Assign each point to the nearest centroid
            classifications = []
            for point in X_train:
                distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
                classifications.append(np.argmin(distances))
            
            # Update centroids
            new_centroids = []
            for i in range(self.n_clusters):
                cluster_points = [X_train[j] for j in range(len(X_train)) if classifications[j] == i]
                new_centroid = np.mean(cluster_points, axis=0)
                new_centroids.append(new_centroid)
            
            if np.array_equal(self.centroids, new_centroids):
                break
            
            self.centroids = np.array(new_centroids)
        
        return self.centroids
        # END TODO

    def evaluate(self, X):
        """
        Given N data samples in X, find the cluster that each point belongs to 
        using the self.centroids class variable as the centroids.
        Return two np arrays, the first being self.centroids 
        and the second is an array having length equal to the number of data points 
        and each entry being between 0 and K-1 (both inclusive) where K is number of clusters.
        """
        classification = np.array([np.argmin([np.linalg.norm(x - c) for c in self.centroids]) for x in X])
        return self.centroids, classification

def evaluate_loss(X, centroids, classification):
    loss = 0
    for idx, point in enumerate(X):
        loss += np.linalg.norm(point - centroids[classification[idx]])
    return loss