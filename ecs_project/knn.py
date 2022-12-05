import numpy as np
import pandas as pd
from numba import njit, prange
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
X_train = pd.read_pickle('../../ecsTest/ecs_project/X_train.pkl')
y_train = pd.read_pickle('../../ecsTest/ecs_project/y_train.pkl')
X_test = pd.read_pickle('../../ecsTest/ecs_project/X_test.pkl')
y_test = pd.read_pickle('../../ecsTest/ecs_project/y_test.pkl')
def getfeatures(data):
    feature1 = data.sum(axis=1).reshape(data.shape[0], 10000)
    feature2 = (data[:,0,:,:] - data[:,1,:,:]).reshape(data.shape[0], 10000)
    feature = np.hstack([feature1,feature2])
    return feature

X_train_flat = getfeatures(X_train)
X_test_flat = getfeatures(X_test)
#randomly permute data points
np.random.seed(1)
inds = np.random.permutation(X_train_flat.shape[0])
X_train_flat = X_train_flat[inds]
y_train = y_train[inds]
# Scale the data
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat = scaler.transform(X_test_flat)
## https://medium.com/analytics-vidhya/speed-up-cosine-similarity-computations-in-python-using-numba-c04bc0741750
@njit(fastmath=True, parallel=True)
def cosine_similarity(X_train, X_test):
    ntrain, dim = X_train.shape
    ntest, _ = X_test.shape
    dists = np.zeros((ntest, ntrain))
    # for each item in testing data
    for i in prange(ntest):
        # for each item in training data
        for j in prange(ntrain):
            ab = 0
            aa = 0
            bb = 0
            # for each feature
            for c in prange(dim):
                ab += X_test[i,c] * X_train[j, c]
                aa = X_test[i,c] * X_test[i,c]
                bb = X_train[j, c] * X_train[j, c]
            if aa != 0 and bb != 0:
                dists[i, j] = ab / np.sqrt(aa * bb)
    return dists
def KNN_predict(X_train, y_train, X_test, k):
    # get the distances
    distances = cosine_similarity(X_train, X_test)
    n, _ = X_test.shape
    predicted_labels = np.zeros(n, dtype=int)
    for i in range(n):
        # sort the distance, get the indices
        ind = np.argsort(distances[i,:])[:k]
        # get the label of k nearest neighbors
        klabels = y_train[ind].reshape((k,)).tolist()
        # majority vote and change label to integer
        predicted_labels[i] = round(max(klabels, key=klabels.count))
    return predicted_labels
y_pred = KNN_predict(X_train_flat, y_train, X_test_flat, 241)
acc = accuracy_score(y_test, y_pred)
print(acc)
pd.DataFrame(y_pred).to_csv('../../ecsTest/ecs_project/knn_y_pred.csv')