import os, sys
import pretty_midi
import librosa, librosa.display
import matplotlib.pyplot as plt 

import numpy as np 
import pandas as pd 


np.set_printoptions(threshold=sys.maxsize)

data_dir = "New_Data_Selection/"
bach_dir = "bach_lute_(c)contributors-kunstderfuge/"
file_name = "bwv997_1_(c)grossman.mid"

bach_file = data_dir + bach_dir + file_name

def plot_piano_roll(midi_data, start_pitch, end_pitch, fs=100):
    # roll = midi_data.get_piano_roll(fs)[start_pitch:end_pitch]
    roll = midi_data.get_piano_roll(fs)
    # print(roll[:, 30:40])
    print(roll.shape)
    librosa.display.specshow(roll,
                            hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                            fmin=pretty_midi.note_number_to_hz(start_pitch))

# plt.figure(figsize=(8,6))
# midi_data = pretty_midi.PrettyMIDI(bach_file)
# plot_piano_roll(midi_data,60,96)
# plt.title('piano roll plot of file: {}'.format(file_name))
# plt.show()


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


for filename in os.listdir(data_dir + bach_dir):
    if ".mid" in filename:
        midi_data = pretty_midi.PrettyMIDI(data_dir + bach_dir + filename)
        l = midi_data.get_end_time()
        # scale the sampling frequency by the length of data, so the picture is 
        # of the same size
        fs = 50 * (10000/(l * 50 - 1))
        get_piano_roll_matrix(midi_data, 60, 96, fs=fs)

def get_test_bach_beethoven():
    
    data_dir = "New_Data_Selection/"
    bach_dir = "bach_lute_(c)contributors-kunstderfuge/"
    beethoven_dir = "beethoven_iii_(c)contributors-kunstderfuge/"
    file_name = "bwv997_1_(c)grossman.mid"

    bach_file = data_dir + bach_dir + file_name

    bach_data, bach_label = [], []
    beethoven_data, beethoven_label = [], []

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

    return data, labels


def get_bach_beethoven():
    
    data_dir = "New_Data_Selection/"
    bach_dir = "bach_concertos_(c)contributors-kunstderfuge/"
    beethoven_dir = "beethoven_i_(c)contributors-kunstderfuge/"

    bach_data, bach_label = [], []
    beethoven_data, beethoven_label = [], []

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

    return data, labels


