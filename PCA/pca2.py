from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
import pandas as pd

# define a matrix
df = pd.read_csv("dataset.csv", skiprows=1, names=['Sample', 'Type', 'PW', 'PL', 'SW', 'SL'])
# Dropping the sample number from dataframe
df.drop(['Sample'], axis=1)
features = ['Type', 'PW', 'PL', 'SW', 'SL']
A = df
# calculate the mean of each column
M = mean(A.T, axis=1)
# print(M)
# center columns by subtracting column means
C = A - M
# print(C)
# calculate covariance matrix of centered matrix
V = cov(C.T)
# print(V)
# eigen decomposition of covariance matrix
values, vectors = eig(V)
# print(vectors)
# print(values)
# project data
P = vectors.T.dot(C.T)
# print(P.T)
from sklearn.decomposition import PCA

# define a matrix
# print(A)
# create the PCA instance
pca = PCA(1)
# fit on data
pca.fit(A)
# access values and vectors
# print(pca.components_)
# print(pca.explained_variance_)
# transform data
B = pca.transform(A)
print(B)
