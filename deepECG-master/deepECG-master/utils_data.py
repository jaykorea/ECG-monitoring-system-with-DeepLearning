import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
# import seaborn as sns


def value_of_mat(mat_filename):
    """
    load the mat file and return the data.
    sio.loadmat() returns a dict and 'val' means value.
    """

    return sio.loadmat(mat_filename)["val"][0, :]


def len_of_mat(mat_filename):
    return len(value_of_mat(mat_filename))


def numbers2onehots(a_list):
    def num2onehot(number, length):
        x = np.zeros(length)
        x[number] = 1
        return x

    length = max(a_list) + 1
    return np.array([num2onehot(number, length) for number in a_list])


def onehots2numbers(onehot_array):
    return [list(onehot).index(1) for onehot in onehot_array]


def load_cinc_data(data_path, lb_len, all_labels):
    """
    load 2017 PhysioNet/CinC Challenge dataset

    :param data_path: String
    :param lb_len: int: lower bound of the strength of signal
    :param all_labels: List[String], all the labels, determine the index of the annotation
    :return: X, Y
        X: list of 1-dim np.array, each element is a signal.
        Y: 2-dim np.array, each sample is a one-hot class label.
    """
    
    labels_path = "{}/REFERENCE.csv".format(data_path)

    # get .mat files list
    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    mat_files = [f for f in files if f.startswith("A") and f.endswith('.mat')]

    # filter out short mat_files
    mat_files = [f for f in mat_files if len_of_mat(os.path.join(data_path, f)) >= lb_len]

    # get X
    # X is signals, which is a list of 1-dim np.array
    X = [value_of_mat(os.path.join(data_path, f)) for f in mat_files]

    n_sample = len(X)
    print('Total training size is ', n_sample)

    # get Y
    # signal_IDs is a list of string
    signal_IDs = [f.split(".")[0] for f in mat_files]

    df_label = pd.read_csv(labels_path, sep=',', header=None, names=None)
    df_label.columns = ["sigID", "label"]
    df_label = df_label.set_index("sigID")

    labels = [df_label.loc[sigID, "label"] for sigID in signal_IDs]
    label_ids = [all_labels.index(l) if l in all_labels else 3 for l in labels]

    Y = numbers2onehots(label_ids)

    return X, Y


def plot_ecg(mat_filename, time_interval=1000):
    ecg_signal = list(value_of_mat(mat_filename))
    plt.plot(ecg_signal[:time_interval])
    return ecg_signal

