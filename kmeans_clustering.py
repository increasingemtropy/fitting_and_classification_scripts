from sklearn.cluster import KMeans
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

def plot_clusters(X,idx1=0,idx2=1,y=None):
    
    plt.scatter(X[:,idx1],X[:,idx2],c=y)
    plt.show()
    
def plot_centroids(centroids,idx1=0,idx2=1,weights=None,circ_col='w',mark_col='k'):
    
    if weights is not None:
        centroids = centroids[weights > weights.max()/10]
        
    plt.scatter(centroids[:,idx1],centroids[:,idx2],
                marker='o',s=30,linewidths=8,
                color=circ_col, zorder=10, alpha=0.9)
    plt.scatter(centroids[:,idx1],centroids[:,idx2],
                marker='x',s=50,linewidths=50,
                color=mark_col, zorder=11, alpha=1)
    
def plot_clusters_labels(X,model,idx1=0,idx2=1,y=None):
    plt.scatter(X[:,idx1],X[:,idx2],c=y)
    plot_centroids(model.cluster_centers_,idx1,idx2)
    plt.show()
    
   
iris = datasets.load_iris()
X = iris.data
y = iris.target
k=3

model=KMeans(n_clusters=k, random_state=7)

y_pred = model.fit_predict(X)

plot_clusters(X,0,1,y_pred)
plot_clusters(X,0,1,y)
plot_clusters_labels(X,model,0,1,y_pred)