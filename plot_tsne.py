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
from raw_clustering import load_comps_pianoroll

def plot_repre(files):

	############################################################
	# Load the data
	X, y = load_comps_pianoroll(files)

	############################################################
	# Fit and transform with a TSNE
	from sklearn.manifold import TSNE
	tsne = TSNE(n_components=2, random_state=0)

	############################################################
	# Project the data in 2D
	X_2d = tsne.fit_transform(X)

	############################################################
	# Visualize the data
	target_names = [f[:5] for f in files]
	target_ids = range(len(target_names))

	from matplotlib import pyplot as plt
	plt.figure(figsize=(6, 5))
	# colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
	colors = 'r', 'g'
	for i, c, label in zip(target_ids, colors, target_names):
	    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
	plt.legend()
	plt.show()

