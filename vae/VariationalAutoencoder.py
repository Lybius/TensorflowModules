import numpy as np
import tensorflow as tf
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

class VAE:
    """Variational Autoencoder based on Keras Sequential Model."""

    def __init__(self, shape=None, random_state=None):
        """Train VAE with data.

        Parameters
        ----------
        shape : tuple
            Shape of input.
        random_state : int
            Initialize numpy and tensorflow randomizers with seed.
        """
        self.shape = shape
        # set random state
        if random_state is not None:
            np.random.seed(random_state)
            tf.random.set_seed(random_state)

    def encode(self, X):
        """Encodes the input.

        Note
        ----
        X is a 4-dimensional ndarray (number_samples,x_size,y_size,channels) and represents a set of images of the same size.
        The layers need to be initialized and trained before using this method.

        Parameters
        ----------
        X : 'obj':np.ndarray
            The 4-dimensional image data which is to be encoded

        """
        return self.__encoder(X)

    def decode(self, code):
        """Decodes sampled code.

        Note
        ----
        The layers need to be initialized and trained before using this method.

        Parameters
        ----------
        code : 'obj':np.ndarray
            Code which was sampled from the encoded layer. SIze depends on encoded_size

        """
        return self.__decoder(code)


    def train(self, X, epochs=10, batch_size=50, X_target=None, **kwargs):
        """Train VAE with data.

        Parameters
        ----------
        X : :obj:`numpy.ndarray`
            Dataset which is trained.
            First dimension is the number of samples, last dimension is the number of channels.
        epochs : int
            Number of epochs of optimization
        batch_size : int
            Size of each batch during optimization
        kwargs : dict
            various parameters for the Keras optimizer, e.g. learning_rate, clipvalue
        """
        ## Encoder / Decoder
        encoder = self.__encoder
        decoder = self.__decoder
        # Model
        vae = tf.keras.Model(inputs=encoder.inputs,
                             outputs=decoder(encoder.outputs[0]))
        # optimize model by minimizing the expected reconstruction error
        def negative_log_likelihood(x, rv_x): return -rv_x.log_prob(x)
        vae.compile(optimizer=tf.optimizers.Adam(**kwargs),
                    loss=negative_log_likelihood)
        # fit data to reconstruct itself
        if X_target is None: # else train alternative target representation
            X_target=X
        vae.fit(X, X_target, batch_size=batch_size, epochs=epochs)



    def build_lenet5_images(self, encoded_size, k1=20, k2=50, d1=50):
        """Build encoder/decoder based on LeNet5.

        The encoder is based on LeNet5-architecture.

        Parameters
        ----------
        encoded_size
            Size of the code layer. Determines the size of the output of the encoder.
        k1
            Number of kernels in the first convolutional layer.
        k2
            Number of kernels in the second convolutional layer.
        d1
            Number of neurons in the dense layer.

        """

        shape = self.shape

        if shape[0] != shape[1]:  # check if input is quadratic
            raise ValueError("Input shape must represent quadratic image.")
        if not ((shape[0]-12)/4).is_integer() or shape[0] < 16:
            raise ValueError(
                "Input must have shape equal to 4*k+12, where k is integer greater than zero.")

        # number of neurons in second dense layer of decoder
        d2 = k2*((shape[0]-12)//4)**2
        q = (shape[0]-12)//4  # size of second feature layer

        # Prior
        prior = tfd.Independent(
            tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
            reinterpreted_batch_ndims=1)  # Bayesian prior for the latent encoding variables
        # Encoder
        self.__encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(
                input_shape=shape, name="encoder_input"),
            tf.keras.layers.Conv2D(filters=k1, kernel_size=5, strides=(
                1, 1), activation='relu', name="encoder_convolutional_1"),
            tf.keras.layers.MaxPool2D(pool_size=(
                2, 2), padding='valid', name="encoder_max_pooling2d_1"),
            tf.keras.layers.Conv2D(filters=k2, kernel_size=5, strides=(
                1, 1), activation='relu', name="encoder_convolutional_2"),
            tf.keras.layers.MaxPool2D(pool_size=(
                2, 2), padding='valid', name="encoder_max_pooling2d_2"),
            tf.keras.layers.Flatten(name="encoder_flatten"),
            tf.keras.layers.Dense(d1, activation='relu',
                                  name="encoder_dense_1"),
            tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(
                encoded_size), activation='relu', name="encoder_dense_2"),
            tfp.layers.MultivariateNormalTriL(encoded_size, activity_regularizer=tfp.layers.KLDivergenceRegularizer(
                prior, weight=1.0), name="encoder_multivariate_normal")
        ])
        # Decoder
        self.__decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(
                input_shape=[encoded_size], name="decoder_input"),
            tf.keras.layers.Dense(d1, activation='relu',
                                  name="decoder_dense_1"),
            tf.keras.layers.Dense(d2, activation='relu',
                                  name="decoder_dense_2"),
            tf.keras.layers.Reshape([q, q, k2], name="decoder_reshape"),
            tf.keras.layers.UpSampling2D(
                size=(2, 2), name="decoder_upsampling_1"),
            tf.keras.layers.Conv2DTranspose(k1, 5, strides=(
                1, 1), padding='valid', name="decoder_convolutional_transpose_1"),
            tf.keras.layers.UpSampling2D(
                size=(2, 2), name="decoder_upsampling_2"),
            # make one set of kernels for mu and one for sigma. (Hence times 2!)
            tf.keras.layers.Conv2DTranspose(
                2*shape[2], 5, strides=(1, 1), padding='valid', name="decoder_convolutional_transpose_2"),
            tf.keras.layers.Flatten(name="decoder_flatten"),
            tfp.layers.IndependentNormal(shape, name="decoder_normal")
            # tfp.layers.
            # (shape,convert_to_tensor_fn=tfd.Bernoulli.logits,name="decoder_bernoulli")
        ])
        self.__initialized_layers = True
    
    def build_LSTM(self, encoded_size, n_lstm=100,l1=0.0,l2=0.0):
        """Build encoder/decoder based on LSTMs.

        This VAE is based on a pair of LSTMs.

        Parameters
        ----------
        encoded_size
            Size of the code layer. Determines the size of the output of the encoder.
        n_lstm
            Number of LSTM Cells used as encoder/decoder

        """
        shape = self.shape
        timesteps = shape[0]  # shape=(timesteps,channel)
        channels = shape[1]  # shape=(timesteps,channel)
        # Prior
        prior = tfd.Independent(
            tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
            reinterpreted_batch_ndims=1)  # Bayesian prior for the latent encoding variables
        # Encoder
        self.__encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(
                input_shape=shape, name="encoder_input"),
            tf.keras.layers.LSTM(n_lstm, 
                                activation='relu',
                                name="encoder_lstm",
                                kernel_regularizer=l1_l2(l1=l1, l2=l2),
                                recurrent_regularizer=l1_l2(l1=l1, l2=l2),
                                activity_regularizer=l1_l2(l1=l1, l2=l2)),
            tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(
                encoded_size), activation='relu', name="encoder_dense"),
            tfp.layers.MultivariateNormalTriL(encoded_size, activity_regularizer=tfp.layers.KLDivergenceRegularizer(
                prior, weight=1.0), name="encoder_multivariate_normal")
        ])
        # Decoder
        self.__decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(
                input_shape=[encoded_size], name="decoder_input"),
            tf.keras.layers.RepeatVector(timesteps, name="repeat"),
            tf.keras.layers.LSTM(n_lstm, 
                                activation='relu',
                                return_sequences=True, 
                                name="decoder_lstm",
                                kernel_regularizer=l1_l2(l1=l1, l2=l2),
                                recurrent_regularizer=l1_l2(l1=l1, l2=l2),
                                activity_regularizer=l1_l2(l1=l1, l2=l2)),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(tfp.layers.IndependentBernoulli.params_size(
                    channels), name="decoder_dense"),
                name="time_distributor"),
            tfp.layers.IndependentBernoulli(channels, name="decoder_bernoulli")
            # tf.keras.layers.TimeDistributed(
            #     tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(
            #         channels), name="decoder_dense"),
            #     name="time_distributor"),
            # tfp.layers.IndependentNormal(channels, name="decoder_normal")
        ])
        self.__initialized_layers = True

    def save_weights(self, filename="weights.h5"):
        """Save weights locally.

        Parameters
        ----------
        filename : 'obj':string
            The filename of the h5 save file.
        """
        file = h5py.File(filename, 'w')
        print("Saving weights...")
        # Encoder
        encoder_weights = self.__encoder.get_weights()
        encoder_grp = file.create_group("encoder")
        for i in tqdm(range(len(encoder_weights))):
            encoder_grp.create_dataset(
                'encoder_weight' + str(i), data=encoder_weights[i])
        # Decoder
        decoder_weights = self.__decoder.get_weights()
        decoder_grp = file.create_group("decoder")
        for i in tqdm(range(len(decoder_weights))):
            decoder_grp.create_dataset(
                'decoder_weight' + str(i), data=decoder_weights[i])
        print("Finished.")
        file.close()

    def load_weights(self, filename="weights.h5"):
        """Load locally saved weights.

        Note
        ----
        This method only loads the weights. The hierarchy must be constructed by building the enocder and decoder.

        Parameters
        ----------
        filename : 'obj':string
            The filename of the h5 save file.

        """
        file = h5py.File(filename, 'r')
        print("Loading weights...")
        # Encoder
        encoder_weights = []
        for i in tqdm(range(len(file["encoder"].keys()))):
            encoder_weights.append(
                file["encoder"]['encoder_weight' + str(i)][:])
        self.__encoder.set_weights(encoder_weights)
        # Decoder
        decoder_weights = []
        for i in tqdm(range(len(file["decoder"].keys()))):
            decoder_weights.append(
                file["decoder"]['decoder_weight' + str(i)][:])
        print("Finished.")
        self.__decoder.set_weights(decoder_weights)

        file.close()



class ReconstructionScoreVAE(VAE):

    def __init__(self,shape=None,random_state=None):
        super().__init__(shape,random_state)


    def score(self, X, n_samples=10):
        """Compute anomaly score by means of the reconstruction probability.

        This anomaly score is based on the paper "Variational Autoencoder based Anomaly Detection using Reconstruction Probability".
        The score represents the negative log-likelihood of a reconstruction by the trained network.
        The higher the score gets, the less likely would be a reconstruction of the input, hence more likely to be an anomaly.

        Parameters
        ----------
        X : 'obj':np.ndarray
            The 4-dimensional image data which is to be scored
        n_samples : int
            Number of samples used for evaluation of the Monte-Carlo estimate

        Returns
        -------
        'obj':np.ndarray
            Score for each sample in X by means of the estimated reconstruction error. 

        """
        code = self.encode(X).sample(n_samples)
        print("Scoring...")
        scores = [-self.decode(code[k]).log_prob(X)
                  for k in tqdm(range(n_samples))]
        return tf.reduce_mean(scores, axis=0).numpy()

    def predict(self, X, threshold, n_samples=10):
        """Predict whether X is anomolous.

        Classify based on the score and a threshold whether a sample is anomolous or not.
        As the score is a recognition score, samples with smaller score are recognized less likely.

        Parameters
        ----------
        X : 'obj':np.ndarray
            The 4-dimensional image data which is to be classified.
        threshold : float
            Threshold used to classify the samples.
        n_samples : int
            Number of samples used for evaluation of the Monte-Carlo estimate.

        Returns
        -------
        'obj':np.ndarray
            Boolean classification. Iff True the sample is anomolous.

        """
        score = self.score(X, n_samples)
        return score > threshold

    def threshold_max_acc(self, X_val, y_val):
        """Threshold with maximal accuracy

        Find threshold, which maximizes the accuracy on the validation set.

        Parameters
        ----------
        X_val : 'obj':np.ndarray
            Subset used for validation.
        y_val : 'obj':np.ndarray
            Validation set labels

        Returns
        -------
        float
            Threshold, which can be used to determine anomalies. 
            if score(x)<threshold then x is an anomaly.
        float
            Accuracy at the threshold point
        """
        y_val = (y_val != 0)  # set everything but zero to 1
        pos = np.mean(y_val)  # positive samples
        neg = 1-pos
        # calculate score
        score_val = self.score(X_val, n_samples=10)
        # compute roc curve
        false_pos, true_pos, threshold = roc_curve(y_val, score_val)
        accuracy = (true_pos-false_pos)*pos+neg  # (TP+TN)/(P+N)
        idx = np.argmax(accuracy)
        return threshold[idx], accuracy[idx]

    def threshold_max_f(self, X_val, y_val, beta=1):
        """Threshold with f-score

        Find threshold, which maximizes the f-score on the validation set.

        Parameters
        ----------
        X_val : 'obj':np.ndarray
            Subset used for validation.
        y_val : 'obj':np.ndarray
            Validation set labels
        beta : float
            determines the F_beta function to use. 

        Returns
        -------
        float
            Threshold, which can be used to determine anomalies. 
            if score(x)<threshold then x is an anomaly.
        float
            Value of the f-score at th threshold point

        """
        y_val = (y_val != 0)  # set everything but zero to 1
        # calculate score
        score_val = self.score(X_val, n_samples=20)
        # compute precision, recall curve
        precision, recall, threshold = precision_recall_curve(y_val, score_val)
        precision = precision[:-1]
        recall = recall[:-1]
        f = (1+beta**2)*precision*recall/((precision*beta**2)+recall)
        idx = np.argmax(f)
        return threshold[idx], f[idx]

    def stats(self, X_test, y_test, threshold, print_results=False):
        """Return common statistics.

        Compute common statistics to test the network and a given threshold.

        Parameters
        ----------
        X_test : 'obj':np.ndarray
            Subset used for testing.
        y_test : 'obj':np.ndarray
            Test set labels
        threshold : float
            Evaluated scoring threshold
        print_results : boolean
            If true print results to stdout.

        Returns
        -------
        dict
            Dictionary containing common test statistics liek accuracy, precision, recall, f1-score and a confusion matrix

        """
        # data
        y_pred = self.predict(X_test, threshold)
        y_test = (y_test != 0)  # set everything but zero to 1
        # stats
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        return_vals = {"accuracy": accuracy,
                       "precision": precision,
                       "recall": recall,
                       "f1": f1,
                       "confusion": confusion}
        if print_results:
            confusion_string = f"[[\t%i \t%i\t]\n [\t%i \t%i\t]]" % (
                confusion[0][0], confusion[0][1], confusion[1][0], confusion[1][1])
            print("Stats\n"
                  "----\n"
                  "threshold: %0.4f \n"
                  "accuracy: %0.2f\n"
                  "f1-score: %0.2f\n"
                  "precision: %0.2f\n"
                  "recall: %0.2f\n"
                  "confusion matrix:\n%s" % (threshold, accuracy, f1, precision, recall, confusion_string))
        return return_vals

    def plot_roc(self, X_test, y_test, filepath="roc.png"):
        """Plot the Receiver operating characteristic

        Calculates the receiver operating characteristic and saves a plot to disk.

        Parameters
        ----------
        X_test : 'obj':np.ndarray
            Subset used for testing.
        y_test : 'obj':np.ndarray
            Test set labels
        filepath : 'obj':str
            Filepath to be used for saving the plot.
        """
        y_test = (y_test != 0)  # set everything but zero to 1
        pos = np.mean(y_test)  # positive samples
        neg = 1-pos
        # calculate score
        score_test = self.score(X_test, n_samples=20)
        # compute roc curve
        false_pos, true_pos, threshold = roc_curve(y_test, score_test)
        y_pred = score_test[np.newaxis].transpose() > threshold
        accuracy = np.mean(y_pred == y_test[np.newaxis].transpose(), axis=0)
        # accuracy = (true_pos-false_pos)*pos+neg  # (TP+TN)/(P+N)
        roc_auc = auc(false_pos, true_pos)
        # max accuracy threshold
        idx = np.argmax(accuracy)
        fig, ax = plt.subplots()
        plt.scatter(false_pos[idx], true_pos[idx], c="b",
                    s=70, label="maximal ACC= %0.2f" % accuracy[idx])
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_pos, true_pos, 'b',
                 label='ROC curve (AUC= %0.2f)' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(filepath)

    def plot_prc(self, X_test, y_test, beta=1, filepath="prc.png"):
        """Plot the Precision and Recall Statistics
        Calculates the precision and recall statistics and saves a plot to disk.

        Parameters
        ----------
        X_test : 'obj':np.ndarray
            Subset used for testing.
        y_test : 'obj':np.ndarray
            Test set labels
        beta : float
            determines the F_beta function to use. 
        filepath : 'obj':str
            Filepath to be used for saving the plot.
        """
        y_test = (y_test != 0)  # set everything but zero to 1
        pos = np.mean(y_test)  # positive samples
        # calculate score
        score_val = self.score(X_test, n_samples=20)
        # compute precision, recall curve
        precision, recall, _ = precision_recall_curve(y_test, score_val)
        precision = precision[:-1]
        recall = recall[:-1]
        f = (1+beta**2)*precision*recall/((precision*beta**2)+recall)
        prc_auc = auc(recall, precision)
        # max f1 threshold
        idx = np.argmax(f)
        fig, ax = plt.subplots()
        plt.scatter(recall[idx], precision[idx], c="b", s=70,
                    label="maximal F%0.1f= %0.2f" % (beta, f[idx]))
        plt.title('Precision-Recall Characteristic')
        plt.plot(recall, precision, 'b',
                 label='PRC curve (AUC= %0.2f)' % prc_auc)
        plt.legend(loc='lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        # plt.plot([0, 1], [pos, pos], 'r--')
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.savefig(filepath)

def window_stack(a, stepsize=1, window_length=3):
    H=tuple(a[i:1+i-window_length or None:stepsize] for i in range(0,window_length))
    return np.vstack(H)

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
        for p in range(self.pattern_length):
            for c in range(self.channels):
                X[:,p,c]=self.scaler[p][c].fit_transform(X[:,p,c].reshape((-1,1))).ravel()
                Y[:,p,c]=self.scaler[p][c].transform(Y[:,p,c].reshape((-1,1))).ravel()
        super().train(X,epochs,batch_size,Y)

    def predict(self,series,estimators=50):
        # o=series[-1] # last value used for rescaling
        # normalize series to rate of change
        X=series[-self.pattern_length:].copy()
        # series=np.nan_to_num(series[1:]/series[:-1]) # convert inf to very large values
        # scale input
        X=X.reshape((-1,)+self.shape) # reshape to trained shape
        for p in range(self.pattern_length):
            for c in range(self.channels):
                X[:,p,c]=self.scaler[p][c].transform(X[:,p,c].reshape((-1,1))).ravel()

        # predict
        latent=self.encode(X).sample() # sample latent variables
        samples=self.decode(latent).sample(estimators) # sample observation
        # samples=samples[:,:,-self.shift:,:]  # drop unnecessary values
        m=tf.math.reduce_mean(samples,axis=0).numpy() # expected value
        # rescale output
        for p in range(self.pattern_length):
            for c in range(self.channels):
                m[:,p,c]=self.scaler[p][c].inverse_transform(m[:,p,c].reshape((-1,1))).ravel()
        # return np.array([o*m[:i].prod() for i in range(1,len(m)+1)])
        return m[0,-self.shift:,:]
        



if __name__ == "__main__":
    series=np.ones(10000)
    vae=ShiftVAE(20,shift=3,channels=1)
    vae.build_LSTM(2,20)
    vae.train(series,epochs=1)
    x=np.ones(20)
    print(vae.predict(x))