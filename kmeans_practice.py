from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

 
# Generate 1000 pairs of random numbers between 0 and 1 for the 'inputs'

test_data = np.random.random((1000,2))

# Generate output labels for the data, 
# Here we do XOR
# XOR   0  1
#     0 F  T
#     1 T  F

test_labels = np.random.random((1000,1))
for i in range(0,1000):
    test_labels[i,0]=(test_data[i,0]>0.5)^(test_data[i,1]>0.5)
    
    
# Since there are 2 outputs (0 and 1), let's try to identify two clusters

# Instance the model
model=KMeans(n_clusters=2, random_state=7)

# Fit the model then generate predictions

y_pred = model.fit_predict(test_data)

# Recast test labels so plt.scatter will accept them
test_labels = np.ndarray.flatten(test_labels.astype(int))

plt.scatter(test_data[:,0],test_data[:,1],c=test_labels)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Ground Truth')
plt.show()

plt.scatter(test_data[:,0],test_data[:,1],c=y_pred)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Clusters for k=2')
plt.show()

# Probably doesn't work... try 4 clusters?

model2 = KMeans(n_clusters=4,random_state = 7)

y_pred2 = model2.fit_predict(test_data)

plt.scatter(test_data[:,0],test_data[:,1],c=y_pred2)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Clusters for k=4')
plt.show()

#