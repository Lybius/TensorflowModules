import numpy as np
import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers.advanced_activations import ReLU
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import h5py
import os
from matplotlib import pyplot as plt
import unittest
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, confusion_matrix, roc_auc_score, accuracy_score, precision_score, accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.regularizers import l1_l2

from .VariationalAutoencoder import VAE
from .helper import window_stack


class ShiftVAE(VAE):

    def __init__(self,pattern_length,shift=1,channels=1,random_state=None):
        self.pattern_length=pattern_length
        self.shift=shift
        self.channels=channels
        self.scaler=[[MinMaxScaler() for i in range(channels)] for j in range(pattern_length)]
        super().__init__((pattern_length,channels),random_state)

    def train(self, series, epochs=10, batch_size=50, X_target=None, **kwargs):
        # normalize series to rate of change
        # series=series[1:]/series[:-1]
        # rearrage series to stacked matrix
        ## INPUT pattern
        X=window_stack(series[:-self.shift],stepsize=1,window_length=self.pattern_length)
        X=X[...,np.newaxis]
        X=X.swapaxes(0,1)
        ## OUTPUT pattern
        Y=window_stack(series[self.shift:],stepsize=1,window_length=self.pattern_length)
        Y=Y[...,np.newaxis]
        Y=Y.swapaxes(0,1)
        # drop inf training data
        # drop=np.isinf(X).any(axis=1).any(axis=1) | np.isinf(Y).any(axis=1).any(axis=1)
        # X,Y=X[~drop],Y[~drop]
        # train network on shifted data
        
        # train scalers for each position in shape X.shape=(samples,timesteps,channels)
        # for p in range(self.pattern_length):
        #     for c in range(self.channels):
        #         X[:,p,c]=self.scaler[p][c].fit_transform(X[:,p,c].reshape((-1,1))).ravel()
        #         Y[:,p,c]=self.scaler[p][c].transform(Y[:,p,c].reshape((-1,1))).ravel()
        super().train(X,epochs,batch_size,Y)

    def predict(self,series,estimators=50):
        # o=series[-1] # last value used for rescaling
        # normalize series to rate of change
        X=series[-self.pattern_length:].copy()
        # series=np.nan_to_num(series[1:]/series[:-1]) # convert inf to very large values
        # scale input
        X=X.reshape((-1,)+self.shape) # reshape to trained shape
        # for p in range(self.pattern_length):
        #     for c in range(self.channels):
        #         X[:,p,c]=self.scaler[p][c].transform(X[:,p,c].reshape((-1,1))).ravel()

        # predict
        latent=self.encode(X).sample() # sample latent variables
        samples=self.decode(latent).sample(estimators) # sample observation
        # samples=samples[:,:,-self.shift:,:]  # drop unnecessary values
        m=tf.math.reduce_mean(samples,axis=0).numpy() # expected value
        # rescale output
        # for p in range(self.pattern_length):
        #     for c in range(self.channels):
        #         m[:,p,c]=self.scaler[p][c].inverse_transform(m[:,p,c].reshape((-1,1))).ravel()
        # return np.array([o*m[:i].prod() for i in range(1,len(m)+1)])
        return m[0,-self.shift:,:]