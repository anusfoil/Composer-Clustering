import os, sys
import pretty_midi
import librosa, librosa.display
import matplotlib.pyplot as plt 

import numpy as np 
import pandas as pd 

import pypianoroll


np.set_printoptions(threshold=sys.maxsize)

data_dir = "New_Data_Selection/"
bach_dir = "bach_all/"
file_name = data_dir + bach_dir + "BWV847b_WTKI_(c)bachovich.mid"

mt = pypianoroll.load(file_name)

pypianoroll.save("try.npz", mt)

# for t in mt.tracks:
    # print(t.pianoroll.shape)
    # pypianoroll.plot(t, filename="try_{}.png".format(t.name))

mpr = mt.get_merged_pianoroll()
print(mpr.shape)

# with np.load('try.npz') as data:
#     print(data['pianoroll_2_csc_data'].shape)

# print(data)

# pypianoroll.plot_multitrack(mt, filename="try.png", 
#     mode='separate', track_label='name', 
#     preset='default', cmaps=None, xtick='auto', ytick='octave', 
#     xticklabel=True, yticklabel='auto', tick_loc=None, 
#     tick_direction='in', label='both', grid='both', 
#     grid_linestyle=':', grid_linewidth=0.5)




