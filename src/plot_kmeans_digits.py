"""
===========================================================
A demo of K-Means clustering on the handwritten digits data
===========================================================

In this example we compare the various initialization strategies for
K-means in terms of runtime and quality of the results.

As the ground truth is known here, we also apply different cluster
quality metrics to judge the goodness of fit of the cluster labels to the
ground truth.

Cluster quality metrics evaluated (see :ref:`clustering_evaluation` for
definitions and discussions of the metrics):

=========== ========================================================
Shorthand    full name
=========== ========================================================
homo         homogeneity score
compl        completeness score
v-meas       V measure
ARI          adjusted Rand index
AMI          adjusted mutual information
silhouette   silhouette coefficient
=========== ========================================================

"""
# print(__doc__)

import os, sys
from time import time
import numpy as np
import matplotlib.pyplot as plt

import pretty_midi
import librosa, librosa.display

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


# =========== ========================================================


np.set_printoptions(threshold=sys.maxsize)

data_dir = "New_Data_Selection/"
bach_dir = "bach_lute_(c)contributors-kunstderfuge/"
beethoven_dir = "beethoven_iii_(c)contributors-kunstderfuge/"
file_name = "bwv997_1_(c)grossman.mid"

bach_file = data_dir + bach_dir + file_name

bach_data, bach_label = [], []
beethoven_data, beethoven_label = [], []

def get_piano_roll_matrix(midi_data, start_pitch, end_pitch, fs=50, draw=False):
    # roll = midi_data.get_piano_roll(fs)[start_pitch:end_pitch]
    matrix = midi_data.get_piano_roll(fs)[:, :10000]
    # print(matrix[:, 30:40])
    # print(matrix.shape)

    if draw: 
      librosa.display.specshow(matrix,
            hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
            fmin=pretty_midi.note_number_to_hz(start_pitch))

    return np.array(matrix).flatten()



for filename in os.listdir(data_dir + beethoven_dir):
    if ".mid" in filename:
        print(filename)
        midi_data = pretty_midi.PrettyMIDI(data_dir + beethoven_dir + filename)
        l = midi_data.get_end_time()
        # scale the sampling frequency by the length of data, so the picture is 
        # of the same size
        fs = 100 * (10000/(l * 50 - 1))
        # beethoven_data.append(get_piano_roll_matrix(midi_data, 48, 96, fs=fs, draw=True))
        plt.figure(figsize=(8,6))
        beethoven_data.append(get_piano_roll_matrix(midi_data,36,108,fs=fs,draw=False))
        plt.title('piano roll plot of file: {}'.format(file_name))
        # plt.show()
        beethoven_label.append(1)

for filename in os.listdir(data_dir + bach_dir):
    if ".mid" in filename:
        print(filename)
        midi_data = pretty_midi.PrettyMIDI(data_dir + bach_dir + filename)
        l = midi_data.get_end_time()
        # scale the sampling frequency by the length of data, so the picture is 
        # of the same size
        fs = 50 * (10000/(l * 50 - 1))
        # bach_data.append(get_piano_roll_matrix(midi_data, 48, 96, fs=fs))
        plt.figure(figsize=(8,6))
        bach_data.append(get_piano_roll_matrix(midi_data,36,108,fs=fs,draw=False))
        plt.title('piano roll plot of file: {}'.format(file_name))
        # plt.show()
        bach_label.append(0)


data = np.array(bach_data + beethoven_data)
labels = np.array(bach_label + beethoven_label)

print(data.shape)


# =========== ========================================================




np.random.seed(42)

# X_digits, y_digits = load_digits(return_X_y=True)
# print("X_digits shape: {}".format(X_digits.shape))
# print("y_digits shape: {}".format(y_digits.shape))
# data = scale(X_digits)
print("data shape: {}".format(data.shape))

n_samples, n_features = data.shape
n_digits = len(np.unique(labels))
# n_digits = len(np.unique(y_digits))
# labels = y_digits

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print(labels)
    print(estimator.labels_)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based",
              data=data)
print(82 * '_')

# #############################################################################
# Visualize the results on PCA-reduced data

# reduced_data = PCA(n_components=2).fit_transform(data)
# kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
# kmeans.fit(reduced_data)

# # Step size of the mesh. Decrease to increase the quality of the VQ.
# h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation='nearest',
#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Paired,
#            aspect='auto', origin='lower')

# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)
# plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
#           'Centroids are marked with white cross')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()
