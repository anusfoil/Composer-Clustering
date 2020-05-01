"""
==========================
tSNE to visualize digits
==========================

Here we use :class:`sklearn.manifold.TSNE` to visualize the digits
datasets. Indeed, the digits are vectors in a 8*8 = 64 dimensional space.
We want to project them in 2D for visualization. tSNE is often a good
solution, as it groups and separates data points based on their local
relationship.

"""
import pandas as pd
import numpy as np
from scipy import stats
from raw_clustering import load_comps_pianoroll

def plot_repre(files):

	############################################################
	# Load the data
	# files = ["dandrieu_all.csv", "soler_all.csv",
	# "dvorak_all.csv", "schumann_all.csv",
	# "handel_all.csv", "mozart_all.csv",
	# "buxtehude_all.csv", "faure_all.csv",
	# "scriabin_all.csv", "byrd_all.csv",
	# "shostakovich_all.csv", "brahms_all.csv",
	# "chopin_all.csv", "debussy_all.csv",
	# "schubert_all.csv", "alkan_all.csv",
	# "haydn_all.csv", "beethoven_all.csv",
	# "scarlatti_all.csv", "bach_all.csv"]

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

	files = [alkan_dir, bach_dir, beethoven_dir,
               brahms_dir, buxtehude_dir, byrd_dir, 
               chopin_dir, dandrieu_dir, dvorak_dir, debussy_dir,
               faure_dir, handel_dir, haydn_dir,
               mozart_dir, scarlatti_dir, schubert_dir,
              schumann_dir, scriabin_dir, shostakovich_dir, soler_dir]

	X, y = load_comps_pianoroll(files)


	# in_dir = "0229_Experiment/processed_csv/"

	# all_compositions = 0
	# # read the csv file 
	# for idx, file in enumerate([in_dir + f for f in files]): 
	# 	if type(all_compositions) == int:
	# 		all_compositions = pd.read_csv(file)
	# 	else:
	# 		all_compositions = all_compositions.append(pd.read_csv(file) )

	# # get rid of the labels and names, this is the data to run
	# comp_data = all_compositions.iloc[:, 3:]

	# # print(comp_data)

	# # sys.exit()

	# # Standardize
	# clmns = comp_data.columns.values.tolist()[1:]
	# # print(clmns)

	# # number of cluters (composers)
	# n = len(np.unique((np.array(all_compositions.iloc[:, 2:3])).flatten()))
	# print(n)

	# # standarize the data
	# comp_data_std = stats.zscore(comp_data[clmns])
	# X = np.nan_to_num(comp_data_std)

	# y = np.array(all_compositions.iloc[:, 2:3]).reshape(X.shape[0])

	############################################################
	# Fit and transform with a TSNE
	from sklearn.manifold import TSNE
	tsne = TSNE(n_components=2, random_state=0)

	print("here 1")
	############################################################
	# Project the data in 2D
	X_2d = tsne.fit_transform(X)

	print("here 2")
	############################################################
	# Visualize the data
	target_names = [f[:5] for f in files]
	target_ids = range(len(target_names))

	from matplotlib import pyplot as plt
	plt.figure(figsize=(6, 5))
	colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'yellow', 'orange', 'purple',
			'pink', 'indigo', 'olive', 'darkred', 'lime', 
			'royalblue', 'grey', 'blueviolet', 'palevioletred', 'yellowgreen']

	# colors = 'r', 'g'
	for i, c, label in zip(target_ids, colors, target_names):
	    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
	plt.legend()
	plt.show()



# plot_repre([])