import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from VariationalAutoencoder import VAE
from helper import window_stack


class ShiftVAE(VAE):

    def __init__(self,pattern_length,shift=1,channels=1,random_state=None):
        self.pattern_length=pattern_length
        assert shift>0 # shift must be positive
        self.shift=shift
        self.channels=channels
        self.scaler=[[MinMaxScaler() for i in range(channels)] for j in range(pattern_length)]
        super().__init__((pattern_length,channels),random_state)

    def train(self, series, epochs=10, batch_size=50):
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

    def predict(self,series,estimators=1000):
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



if __name__=="__main__":
    t=np.linspace(0,100,1000)
    series=np.sin(t)/2+0.5
    series2=np.cos(t)/2+0.5


    vae=ShiftVAE(100,shift=1)
    vae.build_LSTM(encoded_size=10,n_lstm=100,l1=0.01,l2=0.1,dropout_rate=0.5,mode="bernoulli")
    vae.train(series,epochs=200)

    
    for i in range(20):
        x_new=vae.predict(series2)
        series2=np.append(series2,x_new)
    plt.plot(series2)
    plt.savefig("plot.png")  