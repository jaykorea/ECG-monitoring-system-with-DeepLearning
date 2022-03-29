import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gc

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
# import seaborn as sns

from models.Conv1d import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold

from utils_ml import *
from utils_data import *


def train_and_evaluate(model, X_train, Y_train, X_test, Y_test, model_name):

    model_path = "./trained_models/{}/".format(model_name)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    checkpointer = ModelCheckpoint(filepath='{}/best_model.h5'.format(model_path),
                                   monitor='val_acc',
                                   verbose=1,
                                   save_best_only=True)

    # early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=1, mode='auto')

    # print("x shape", X_train.shape)
    # print("y shape", Y_train.shape)

    hist = model.fit(X_train, Y_train,
                     validation_data=(X_test, Y_test),
                     batch_size=275,
                     epochs=3,
                     verbose=2,
                     shuffle=True,
                     callbacks=[checkpointer])

    # save history
    pd.DataFrame(hist.history).to_csv('{}/history.csv'.format(model_path))

    pd.DataFrame(hist.history['acc']).to_csv('{}/train_acc.csv'.format(model_path))
    pd.DataFrame(hist.history['loss']).to_csv('{}/loss.csv'.format(model_path))
    pd.DataFrame(hist.history['val_acc']).to_csv('{}/val_acc.csv'.format(model_path))
    pd.DataFrame(hist.history['val_loss']).to_csv('{}/val_loss.csv'.format(model_path))

    # evaluation
    predictions = model.predict(X_test)

    score = accuracy_score(onehots2numbers(Y_test), predictions.argmax(axis=1))
    print('Last epoch\'s validation score is ', score)

    df = pd.DataFrame(predictions.argmax(axis=1))
    df.to_csv('{}/preds_{.4f}.csv'.format(model_path, score), index=None, header=None)

    cm = confusion_matrix(onehots2numbers(Y_test), predictions.argmax(axis=1))
    df = pd.DataFrame(cm)
    df.to_csv('{}/confusion_matrix_{.4f}.csv'.format(model_path, score), index=None, header=None)

    del model
    K.clear_session()
    gc.collect()

    return score


def cross_validation(model, X, Y, n_fold=10):
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True)
    y = Y.reshape(X.shape[0],)

    scores = []
    for i, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train = X[train_index, :]
        Y_train = Y[train_index, :]
        X_val = X[val_index, :]
        Y_val = Y[val_index, :]
        score = train_and_evaluate(model, X_train, Y_train, X_val, Y_val, i)
        scores.append(score)

    return scores


# project parameters
DATA_PATH = 'data/training2017/'
LABELS_PATH = DATA_PATH + 'REFERENCE.csv'

# lower bound of the length of the signal
LB_LEN_MAT = 100

# upper bound of the length of the signal
UB_LEN_MAT = 10100

LABELS = ["N", "A", "O"]
n_classes = len(LABELS) + 1

np.random.seed(7)

if __name__ == "__main__":

    # this helps a lot when debugging
    print(os.getcwd())

    X, Y = load_cinc_data(DATA_PATH, LB_LEN_MAT, LABELS)

    # data preprocessing
    X = duplicate_padding(X, UB_LEN_MAT)
    X = (X - X.mean()) / (X.std())
    X = np.expand_dims(X, axis=2)

    # shuffle the data
    values = [i for i in range(len(X))]
    permutations = np.random.permutation(values)
    X = X[permutations, :]
    Y = Y[permutations, :]

    # train test split
    train_test_ratio = 0.9
    n_sample = X.shape[0]

    X_train = X[:int(train_test_ratio * n_sample), :]
    Y_train = Y[:int(train_test_ratio * n_sample), :]
    X_test = X[int(train_test_ratio * n_sample):, :]
    Y_test = Y[int(train_test_ratio * n_sample):, :]
    
    # load the model and train it
    model = conv1d(UB_LEN_MAT)

    cross_validation(model, X, Y)
    # train_and_evaluate(model, X_train, Y_train, X_test, Y_test, "conv_model")

