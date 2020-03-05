import sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.cluster import SpectralClustering, MeanShift, KMeans, AgglomerativeClustering
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale

import sklearn.metrics as sm
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report


def print_4_metrics(result, verbose=False):

	labels_true = result.iloc[:, 3:4].transpose().values[0]
	labels_pred = result.iloc[:, 4:5].transpose().values[0]

	# print(labels_pred)

	a = sm.adjusted_mutual_info_score(labels_true, labels_pred)
	b = sm.adjusted_rand_score(labels_true, labels_pred)

	c = sm.homogeneity_score(labels_true, labels_pred)
	d = sm.completeness_score(labels_true, labels_pred)

	if verbose: 
		print("adjusted_mutual_info_score:")
		print(a) 
		print("adjusted_rand_score:")
		print(b) 
		print("homogeneity_score:")
		print(c) 
		print("completeness_score:")
		print(d) 


	return (a, b, c, d)

aa = []
bb = []
cc = []
dd = []
for i in range(1, 11):
	result = pd.read_csv("0229_Experiment/result/f1_e" + str(i) + "_kmeans.csv")
	a, b, c, d = print_4_metrics(result)
	aa.append(a)
	bb.append(b)
	cc.append(c)
	dd.append(d)

print(aa)
print(bb)
print(cc)
print(dd)

# 1, 5, 7 

def draw_pic(result):

	# result = pd.read_csv("0229_Experiment/result/f1_e1_kmeans_sel.csv")

	# Scatter Plot with Hue for visualizing data in 3-D
	cols = ['clusters', 'Range', 'Pitch_Variability', 'Chromatic_Motion', 'Standard_Triads']
	print("here 1")
	pp = sns.pairplot(result.iloc[:, 4:9], hue='clusters', size=1.8, aspect=1.8, 
	                  palette={0: "#FF9999", 1: "#FFE888"},
	                  plot_kws=dict(edgecolor="black", linewidth=0.5))
	print("here 2")
	fig = pp.fig 
	fig.subplots_adjust(top=0.93, wspace=0.3)
	print("here 3")
	t = fig.suptitle('Wine Attributes Pairwise Plots', fontsize=14)
	fig.savefig("output.png")


