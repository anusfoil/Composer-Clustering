import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import SpectralClustering, MeanShift, KMeans, AgglomerativeClustering
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale

import sklearn.metrics as sm
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report


# read the csv file 
all_compositions = pd.read_csv("Feature/bartok_mendelssohn_feature_values.csv")

# number of cluters (composers)
n = 2

print(all_compositions.shape)

# # We have more "Yes" votes than "No" votes overall
# print(pd.value_counts(all_compositions.iloc[:,3:].values.ravel()))

# Standardize
clmns = all_compositions.columns.values.tolist()[1:]
# print(clmns)


# standarize the data
all_compositions_std = stats.zscore(all_compositions[clmns])
all_compositions_std = np.nan_to_num(all_compositions_std)



#Cluster the data
spectral = SpectralClustering(n_clusters=n, assign_labels="discretize", random_state=0).fit(all_compositions_std)
meanshift = MeanShift(bandwidth=30).fit(all_compositions_std)
kmeans = KMeans(n_clusters=n, random_state=0).fit(all_compositions_std)
agglo = AgglomerativeClustering(n_clusters=n, affinity='euclidean',linkage='ward').fit(all_compositions_std)

# labels = spectral.labels_
# labels = meanshift.labels_
# labels = kmeans.labels_
labels = agglo.labels_

#Glue back to originaal data
all_compositions['clusters'] = labels

#Add the column into our list
clmns.extend(['clusters'])

#Lets analyze the clusters
print(all_compositions[clmns].groupby(['clusters']).mean())

print(all_compositions.iloc[:, :])




