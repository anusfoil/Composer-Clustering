import sys
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

composers = ["dandrieu", "soler",
			 "dvorak", "schumann",
			 "butxehude", "faure",
			 "scriabin", "byrd",
			 "shostakovich", "brahms",
			 "chopin", "debussy",
			 "schubert", "alkan",
			 "handel", "mozart",
			 "haydn", "beethoven",
			 "scarlatti", "bach"]

in_dir = "0229_Experiment/41_processed_csv/"
out_dir = "0229_Experiment/41_result/"

# files: an array of csv file names. We will combine the files 
# csv format: [index] [file name] [y-column] [feature] [feature] ... [feature]
# in y-column, we identify each composer by an index. 
# The index column and file name column will be ignored. 
def run_clustering(algo, files):

	all_compositions = 0
	# read the csv file 
	for idx, file in enumerate([in_dir + f for f in files]): 
		# first term is experiment name
		if idx == 0:
			continue
		if type(all_compositions) == int:
			all_compositions = pd.read_csv(file)
		else:
			all_compositions = all_compositions.append(pd.read_csv(file) )


	# number of cluters (composers)
	n = len(files) - 1

	# get rid of the labels and names, this is the data to run
	comp_data = all_compositions.iloc[:, 3:]

	# print(comp_data)

	# sys.exit()

	# Standardize
	clmns = comp_data.columns.values.tolist()[1:]
	# print(clmns)

	# standarize the data
	comp_data_std = stats.zscore(comp_data[clmns])
	comp_data_std = np.nan_to_num(comp_data_std)

	#Cluster the data
	
	
	if algo == "kmeans":
		kmeans = KMeans(n_clusters=n, random_state=0).fit(comp_data_std)
		labels = kmeans.labels_
	if algo == "spectral":
		spectral = SpectralClustering(n_clusters=n, affinity="rbf",
			assign_labels="kmeans", random_state=0).fit(comp_data_std)
		labels = spectral.labels_
	if algo == "meanshift":
		meanshift = MeanShift(bandwidth=30).fit(comp_data_std)
		labels = meanshift.labels_
	if algo == "agglo":
		agglo = AgglomerativeClustering(n_clusters=n, affinity='euclidean',linkage='ward').fit(comp_data_std)
		labels = agglo.labels_
	

	#Glue back to originaal data
	all_compositions.insert(3, 'clusters', labels)

	#Add the column into our list
	clmns.extend(['clusters'])

	#Lets analyze the clusters
	# print(all_compositions[clmns].groupby(['clusters']).mean())

	# print(all_compositions.iloc[:, :])


	all_compositions.to_csv(out_dir + files[0] + "_" + algo + ".csv")

	# calculate the statistics 


'''
Format 1: 2 composers, each 50 pieces, all form 
10 experiments with pairs 
Format 2: 4 composers in (baroque, classical, romantic, late-romantic), each 50 pieces, all form 
5 experiments 
Format 3: For each style, we find 4 composers to differentiate, each 50 pieces, all form 
Format 4: 10 top composers with all their pieces 
Handel, Bach, Scarlatti, Beethoven, Mozart, Schubert, Brahms, Debussy, Dvorak, Shostakovich 
Format 5: 20 composers with equal number of pieces (60 - 100)
Format 6: All music 

'''
f1_e1 = ["f1_e1", "dandrieu_all.csv", "soler_all.csv"]
f1_e2 = ["f1_e2", "dvorak_all.csv", "schumann_all.csv"]
f1_e3 = ["f1_e3", "buxtehude_all.csv", "faure_all.csv"]
f1_e4 = ["f1_e4", "scriabin_all.csv", "byrd_all.csv"]
f1_e5 = ["f1_e5", "shostakovich_all.csv", "brahms_all.csv"]
f1_e6 = ["f1_e6", "chopin_all.csv", "debussy_all.csv"]
f1_e7 = ["f1_e7", "schubert_all.csv", "alkan_all.csv"]
f1_e8 = ["f1_e8", "handel_all.csv", "mozart_all.csv"]
f1_e9 = ["f1_e9", "haydn_all.csv", "beethoven_all.csv"]
f1_e10 = ["f1_e10", "scarlatti_all.csv", "bach_all.csv"]

f1s = [f1_e1, f1_e2, f1_e3, f1_e4, f1_e5, f1_e6,
		f1_e7, f1_e8, f1_e9, f1_e10]

# f1s = [f1_e1]

f4 = ["f4", "handel_all.csv", "bach_all.csv", 
		"scarlatti_all.csv", "beethoven_all.csv", 
		"mozart_all.csv", "schubert_all.csv",
		"brahms_all.csv", "debussy_all.csv",
		"dvorak_all.csv", "shostakovich_all.csv"]

f6 = ["dandrieu_all.csv", "soler_all.csv",
	"dvorak_all.csv", "schumann_all.csv",
	"buxtehude_all.csv", "faure_all.csv",
	"scriabin_all.csv", "byrd_all.csv",
	"shostakovich_all.csv", "brahms_all.csv",
	"chopin_all.csv", "debussy_all.csv",
	"schubert_all.csv", "alkan_all.csv",
	"handel_all.csv", "mozart_all.csv",
	"haydn_all.csv", "beethoven_all.csv",
	"scarlatti_all.csv", "bach_all.csv"]

# for x in f1s:
# 	run_clustering("kmeans", x)
	# run_clustering("spectral", x)


run_clustering("kmeans", f1_e3)



