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

from metrics import *

in_dir = "0229_Experiment/processed_csv/"
out_dir = "0229_Experiment/result/"

# files: an array of csv file names. We will combine the files 
# csv format: [index] [file name] [y-column] [feature] [feature] ... [feature]
# in y-column, we identify each composer by an index. 
# The index column and file name column will be ignored. 
def run_clustering_features(algo, files, save=False):

	all_compositions = 0
	# read the csv file 
	for idx, file in enumerate([in_dir + f for f in files]): 
		if type(all_compositions) == int:
			all_compositions = pd.read_csv(file)
		else:
			all_compositions = all_compositions.append(pd.read_csv(file) )

	# get rid of the labels and names, this is the data to run
	comp_data = all_compositions.iloc[:, 3:]

	# print(comp_data)

	# sys.exit()

	# Standardize
	clmns = comp_data.columns.values.tolist()[1:]
	# print(clmns)

	# number of cluters (composers)
	n = len(np.unique((np.array(all_compositions.iloc[:, 2:3])).flatten()))
	print(n)

	# standarize the data
	comp_data_std = stats.zscore(comp_data[clmns])
	comp_data_std = np.nan_to_num(comp_data_std)

	#Cluster the data
	if algo == "kmeans":
		kmeans = KMeans(n_clusters=n, random_state=0).fit(comp_data_std)
		print(comp_data_std.shape)
		labels = kmeans.labels_
		sys.exit()
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
	
	y_pred = labels
	y = np.array(all_compositions.iloc[:, 2:3]).reshape(y_pred.shape)


	# print(np.array(y).reshape((y.shape[0])))
	# print(y_pred)

	# evaluating 
	if save:
		name = "_".join([f[:4] for f in files])
		with open("results/raw_feature/f1/{}.txt".format(name), "w") as f:

			f.write(", ".join([f[:-4] for f in files]) + "\n")
			f.write("acc: {}\n".format(acc(y, y_pred)))
			f.write("vms: {}\n".format(vms(y, y_pred)))
			f.write("nmi: {}\n".format(nmi(y, y_pred)))
			f.write("ari: {}\n".format(ari(y, y_pred)))


	return acc(y, y_pred)

	#Glue back to originaal data
	all_compositions.insert(3, 'clusters', labels)

	#Add the column into our list
	clmns.extend(['clusters'])

	#Lets analyze the clusters
	# print(all_compositions[clmns].groupby(['clusters']).mean())

	# print(all_compositions.iloc[:, :])


	all_compositions.to_csv(out_dir + files[0] + "_" + algo + ".csv")

	# calculate the statistics 



# files: an array of directories
# load the compositions dataset
# the piano rolls are pre saved in the directory
def load_comps_pianoroll(files):

    import os
    from PIL import Image
    data_dir = "pianoroll/"

    composer_label = np.empty((0,))

    x, y = np.empty((0, 128, 1000)), np.empty((0,))
    for idx, comp in enumerate(files):
        composer_data = np.empty((0, 128, 1000))
        path = data_dir + comp
        for filename in os.listdir(path):
            img = np.asarray(Image.open("{}/{}".format(path, filename)).convert("L"))
            try:
                composer_data = np.append(composer_data, np.expand_dims(img, axis=0), axis=0)

            except:
              pass
        x = np.append(x, composer_data, axis=0)
        y = np.append(y, np.ones((composer_data.shape[0],)) * idx, axis=0)

    # x = np.vstack((data1, data2))
    # y = np.hstack((np.zeros(data1.shape[0]), np.ones(data2.shape[0])))

    x = x.reshape((x.shape[0], -1))
    assert x.shape[0] == y.shape[0]

    return x, y



# files: an array of csv file names. We will combine the files 
def run_clustering_pr(algo, files, save=False):

	x, y = load_comps_pianoroll(files)
	comp_data = x


	# number of cluters (composers)
	n = len(np.unique(y))

	# standarize the data
	comp_data_std = stats.zscore(comp_data)
	comp_data_std = np.nan_to_num(comp_data_std)

	print(comp_data_std.shape)

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
	
	y_pred = labels



	# print(np.array(y).reshape((y.shape[0])))
	# print(y_pred)

	# evaluating 
	if save:
		name = "_".join([f[:4] for f in files])
		with open("results/raw_pr/f1/{}.txt".format(name), "w") as f:

			f.write(", ".join([f[:-4] for f in files]) + "\n")
			f.write("acc: {}\n".format(acc(y, y_pred)))
			f.write("vms: {}\n".format(vms(y, y_pred)))
			f.write("nmi: {}\n".format(nmi(y, y_pred)))
			f.write("ari: {}\n".format(ari(y, y_pred)))


	return acc(y, y_pred)


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

UPDATE: we should do every pair; do a confusion matrix (heatmap)! 

'''

# 直接的midi数据
alkan_dir = "alkan_(c)contributors-kunstderfuge/"
bach_dir = "bach_all/"
dandrieu_dir = "dandrieu_(c)contributors-kunstderfuge/"
dvorak_dir = "dvorak_(c)contributors-kunstderfuge/"
scriabin_dir = "scriabin_(c)contributors-kunstderfuge/"
byrd_dir = "byrd_(c)contributors-kunstderfuge/"
faure_dir = "faure_(c)contributors-kunstderfuge/"
buxtehude_dir = "buxtehude_(c)contributors-kunstderfuge/"
beethoven_dir = "beethoven_all/"
brahms_dir = "brahms_all/"
schubert_dir = "schubert_all/"
chopin_dir = "chopin_all/"
debussy_dir = "debussy_all/"
handel_dir = "handel_all/"
haydn_dir = "haydn_all/"
scarlatti_dir = "scarlatti_all/"
mozart_dir = "mozart_all/"
schumann_dir = "schumann_(c)contributors-kunstderfuge/"
scriabin_dir = "scriabin_(c)contributors-kunstderfuge/"
shostakovich_dir = "shostakovich_(c)contributors-kunstderfuge/"
soler_dir = "soler_(c)contributors-kunstderfuge/"


alkan_csv = "alkan_all.csv"
bach_csv = "bach_all.csv"
beethoven_csv = "beethoven_all.csv"
brahms_csv = "brahms_all.csv"
buxtehude_csv = "buxtehude_all.csv"
byrd_csv = "byrd_all.csv"
chopin_csv = "chopin_all.csv"
dandrieu_csv = "dandrieu_all.csv"
dvorak_csv = "dvorak_all.csv"
debussy_csv = "debussy_all.csv"
faure_csv = "faure_all.csv"
handel_csv = "handel_all.csv"
haydn_csv = "haydn_all.csv"
mozart_csv = "mozart_all.csv"
scarlatti_csv = "scarlatti_all.csv"
schubert_csv = "schubert_all.csv"
schumann_csv = "schumann_all.csv"
scriabin_csv = "scriabin_all.csv"
shostakovich_csv = "shostakovich_all.csv"
soler_csv = "soler_all.csv"

csv_comps = [alkan_csv, bach_csv, beethoven_csv,
               brahms_csv, buxtehude_csv, byrd_csv, 
               chopin_csv, dandrieu_csv, dvorak_csv, debussy_csv,
               faure_csv, handel_csv, haydn_csv,
               mozart_csv, scarlatti_csv, schubert_csv,
              schumann_csv, scriabin_csv, shostakovich_csv, soler_csv]

pr_comps = [alkan_dir, bach_dir, beethoven_dir,
               brahms_dir, buxtehude_dir, byrd_dir, 
               chopin_dir, dandrieu_dir, dvorak_dir, debussy_dir,
               faure_dir, handel_dir, haydn_dir,
               mozart_dir, scarlatti_dir, schubert_dir,
              schumann_dir, scriabin_dir, shostakovich_dir, soler_dir]

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

f2_e1 = ["f2_e1", "byrd_all.csv", "mozart_all.csv", "schumann_all.csv", "shostakovich_all.csv"]
f2_e2 = ["f2_e2", "bach_all.csv", "haydn_all.csv", "schubert_all.csv", "debussy_all.csv"]
f2_e3 = ["f2_e3", "buxtehude_all.csv", "beethoven_all.csv", "brahms_all.csv", "scriabin_all.csv"]
f2_e4 = ["f2_e4", "dandrieu_all.csv", "scarlatti_all.csv", "chopin_all.csv", "alkan_all.csv"]


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
	"haydn_all.csv", "beethoven_all.csv",
	"scarlatti_all.csv", "bach_all.csv"]

# for x in f1s:
# 	run_clustering("kmeans", x)
	# run_clustering("spectral", x)

if __name__ == '__main__':

	# all_pairs = [[a, b] for a in csv_comps for b in csv_comps if (a != b and a > b)]
	# for p in list(all_pairs):
	# 	print(p)
	# 	run_clustering("kmeans", p)
	# 	print("done!")
	# data = []
	# for a in pr_comps:
	# 	tmp = []
	# 	for b in pr_comps:
	# 		tmp.append(run_clustering_pr("kmeans", [a, b], save=True))
	# 	data.append(tmp)
	# data = np.array(data)

	# print(run_clustering_features("kmeans", [dandrieu_csv, soler_csv], save=True))
	# np.savetxt("results/pr_accuracy.txt", data)
	data = np.around(np.loadtxt("results/ae_represented_feature/feature_rep_accuracy.txt"), decimals=2)
	print("here 1")


	for i in range(len(data)):
		data[i, i] = 1

	print(data)
	# sys.exit()

	fig, ax = plt.subplots()
	im = ax.imshow(data, cmap="Wistia")

	comps_names = [c[:-8] for c in csv_comps]

	# We want to show all ticks...
	ax.set_xticks(np.arange(len(comps_names)))
	ax.set_yticks(np.arange(len(comps_names)))
	# ... and label them with the respective list entries
	ax.set_xticklabels(comps_names)
	ax.set_yticklabels(comps_names)

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	         rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	for i in range(len(comps_names)):
	    for j in range(len(comps_names)):
	        text = ax.text(j, i, data[i, j],
	                       ha="center", va="center", color="black", size="6")

	ax.set_title("heatmap of pairwise composer clustering acc - feature representation")
	fig.tight_layout()
	plt.show()









