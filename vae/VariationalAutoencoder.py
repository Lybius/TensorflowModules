import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from tqdm import tqdm
import h5py
import os
from matplotlib import pyplot as plt
import unittest
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, confusion_matrix, roc_auc_score, accuracy_score, precision_score, accuracy_score, recall_score, f1_score, confusion_matrix


class CVAE:
    """Convolutional Variational Autoencoder based on Keras Sequential Model."""

    def __init__(self, shape=None, random_state=None):
        """Train VAE with data.

        Parameters
        ----------
        shape : tuple
            Shape of input images.
        random_state : int
            Initialize numpy and tensorflow randomizers with seed.
        """
        if len(shape) == 2:
            # add channel dimension to monochrome images.
            shape = (shape[0], shape[1], 1)
        if len(shape) != 3:
            raise ValueError("shape has to be a tuple of length 3!")
        self.__input_shape = shape
        # layers have to be initialized before using.
        self.__initialized_layers = False
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
        self.check_initialized()
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
        self.check_initialized()
        return self.__decoder(code)

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
        self.check_initialized()
        X = self.resize(X)  # resize images to input shape
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

    def train(self, X, epochs=10, batch_size=50, learning_rate=1e-3):
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
        learning_rate : float
            Learning rate for the optimizer
        """
        self.check_initialized()
        ## Encoder / Decoder
        encoder = self.__encoder
        decoder = self.__decoder
        # Data
        X = self.resize(X)  # Resize Images

        # Model
        vae = tf.keras.Model(inputs=encoder.inputs,
                             outputs=decoder(encoder.outputs[0]))
        # optimize model by minimizing the expected reconstruction error
        def negative_log_likelihood(x, rv_x): return -rv_x.log_prob(x)
        vae.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                    loss=negative_log_likelihood)
        # fit data to reconstruct itself
        vae.fit(X, X, batch_size=batch_size, epochs=epochs)

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
        self.check_initialized()
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
        self.check_initialized()
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

    def stats(self, X_test, y_test, threshold,print_results=False):
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
        self.check_initialized()
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
                       "confusion": confusion }
        if print_results:
                confusion_string=f"[[\t%i \t%i\t]\n [\t%i \t%i\t]]" % (confusion[0][0],confusion[0][1],confusion[1][0],confusion[1][1])
                print("Stats\n"\
                "----\n"\
                "threshold: %0.4f \n"\
                "accuracy: %0.2f\n"\
                "f1-score: %0.2f\n"\
                "precision: %0.2f\n"\
                "recall: %0.2f\n"\
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
        self.check_initialized()
        y_test = (y_test != 0)  # set everything but zero to 1
        pos = np.mean(y_test)  # positive samples
        neg = 1-pos
        # calculate score
        score_test = self.score(X_test, n_samples=20)
        # compute roc curve
        false_pos, true_pos, threshold = roc_curve(y_test, score_test)
        y_pred=score_test[np.newaxis].transpose()>threshold
        accuracy=np.mean(y_pred==y_test[np.newaxis].transpose(),axis=0)
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
        self.check_initialized()
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

    def build_lenet5(self, encoded_size, k1=20, k2=50, d1=50):
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

        shape = self.__input_shape

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

    def resize(self, X):
        """Resize image to correct dimensions.

        Resizes single image by interpolation.

        Parameters
        ----------
        X
            Either a list of 3d numpy ND-arrays or 4d ND-array.

        Returns
        -------
        np.ndarray
            ND-array of resized images

        """
        print("Resize Images...")
        # change datatype
        X = np.array(X, dtype=np.float32)
        # Scale to (0,1)
        X = (X-X.min())/(X.max()-X.min())
        # add new axis for channels
        if X[0].ndim == 2:
            X = X[..., np.newaxis]
        # repeat image for each channel
        if X.shape[1] == self.__input_shape[1] and X.shape[2] == self.__input_shape[2] and X.shape[3] == 1:
            X = np.repeat(X, self.__input_shape[3], axis=-1)
        # resize image
        if X[0].shape != self.__input_shape:
            return np.array([resize(x, output_shape=self.__input_shape) for x in tqdm(X)], dtype=np.float32)
        else:
            return X

    def save_weights(self, filename="weights.h5"):
        """Save weights locally.

        Parameters
        ----------
        filename : 'obj':string
            The filename of the h5 save file.
        """
        self.check_initialized()
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
        self.check_initialized()
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

    def check_initialized(self):
        """Checks whether layers have been initialized."""
        if not self.__initialized_layers:
            raise RuntimeError("The encode/decode layers have not yet been initialized."
                               "(This method only loads the weights. The hierarchy must be constructed by building the enocder and decoder.)")
        return True


def experiment1():
    """Test training and scoring"""
    X = np.random.rand(1000, 100, 100, 3)
    vae = CVAE(shape=(64, 64, 3))
    vae.build_lenet5(10)
    vae.train(X, epochs=1)
    vae.save_weights()

    vae = CVAE(shape=(64, 64, 3))
    vae.build_lenet5(10)
    vae.load_weights()
    Y = np.random.rand(10, 100, 100, 3)
    print(vae.score(Y))


def experiment2():
    """Train classification of zeros and eights"""
    filename="experiment2.h5"
    if not os.path.isfile(filename):  # train
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = [X_train[i] for i in range(len(X_train)) if y_train[i] == 0]
        vae = CVAE(shape=(28, 28))
        vae.build_lenet5(10)
        vae.train(X_train, epochs=20)
        vae.save_weights(filename)
    else:  # load from file
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        vae = CVAE(shape=(28, 28))
        vae.build_lenet5(10)
        vae.load_weights(filename)

    X_test_zero = [X_test[i] for i in range(len(X_test)) if y_test[i] == 0]
    X_test_nonzero = [X_test[i] for i in range(len(X_test)) if y_test[i] == 8]
    score_zero = vae.score(X_test_zero)
    score_nonzero = vae.score(X_test_nonzero)
    print(f"Zero class:\n mean={np.mean(score_zero)}, median={np.median(score_zero)}, std={np.std(score_zero)}, min={np.min(score_zero)}, max={np.max(score_zero)}")
    print(f"Non-Zero class:\n mean={np.mean(score_nonzero)}, median={np.median(score_nonzero)}, std={np.std(score_nonzero)}, min={np.min(score_nonzero)}, max={np.max(score_nonzero)}")

    both = np.concatenate([score_zero, score_nonzero])
    bins = np.linspace(np.quantile(both, 0.1), np.quantile(both, 0.9), 20)
    plt.title("Reconstruction Log-Probability")
    plt.hist(score_zero, bins, alpha=0.5, label='zero')
    plt.hist(score_nonzero, bins, alpha=0.5, label='eight')
    plt.legend(loc='upper right')
    plt.savefig("experiment2.png")


def experiment3():
    """Train betwork on zeros. Treat all other numbers as anomalies."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    # take 10% for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1)
    # Network
    filename="experiment3.h5"
    if not os.path.isfile(filename):  # train
        X_train = [X_train[i] for i in range(len(X_train)) if y_train[i] == 0]
        vae = CVAE(shape=(28, 28))
        vae.build_lenet5(10)
        vae.train(X_train, epochs=20)
        vae.save_weights(filename)
    else:  # load from file
        vae = CVAE(shape=(28, 28))
        vae.build_lenet5(10)
        vae.load_weights(filename)

    # ROC
    vae.plot_roc(X_test, y_test)
    # PRC
    vae.plot_prc(X_test, y_test)

    # use validation set to determine threshold
    th_f1, _ = vae.threshold_max_f(X_val, y_val)
    th_acc, _ = vae.threshold_max_acc(X_val, y_val)
    print("Thresholding using accuracy:")
    vae.stats(X_test,y_test,th_acc,print_results=True)
    print("Thresholding using f1-score:")
    vae.stats(X_test,y_test,th_f1,print_results=True)
    


class CVAE_Testing(unittest.TestCase):
    def test_resize(self):
        """Test that images are correctly resized"""
        n_samples = 2

        for k in [1, 2]:
            input_size = 4*k+12
            for size in [1, 50]:
                X = np.random.rand(n_samples, size, size)
                vae = CVAE(shape=(input_size, input_size))
                vae.build_lenet5(encoded_size=5)
                # Test whether X has correct dimensions for network
                vae.train(X, batch_size=1, epochs=1)
                # Test whether input shape is correct
                self.assertTrue(vae._CVAE__encoder.inputs[0].shape.as_list() == [
                                None, input_size, input_size, 1])
                # Test whether output shape is correct
                self.assertTrue(vae._CVAE__decoder.outputs[0].shape.as_list() == [
                                None, input_size, input_size, 1])

    def test_different_sizes(self):
        """Test that LeNEt-5 works with different layer sizes."""
        n_samples = 3
        for k in [1, 2]:
            size = np.random.randint(5, 300)
            X = np.random.rand(n_samples, size, size)
            input_size = 4*k+12
            vae = CVAE(shape=(input_size, input_size))
            vae.build_lenet5(encoded_size=np.random.randint(1, 2),
                             k1=np.random.randint(1, 2),
                             k2=np.random.randint(1, 2),
                             d1=np.random.randint(1, 2))
            vae.train(X, batch_size=np.random.randint(1, 10), epochs=1)

    def test_save_load(self):
        """Test that weights are correctly saved to disk and restored."""
        n_samples = 5
        size = 28
        X = np.random.rand(n_samples, size, size)
        vae1 = CVAE(shape=(28, 28))
        vae1.build_lenet5(10)
        vae1.train(X, epochs=1)
        filename = "unittest.h5"
        encoder1 = vae1._CVAE__encoder
        decoder1 = vae1._CVAE__decoder
        vae1.save_weights(filename)

        vae2 = CVAE(shape=(28, 28))
        vae2.build_lenet5(10)
        vae2.load_weights(filename)
        encoder2 = vae2._CVAE__encoder
        decoder2 = vae2._CVAE__decoder

        # Assert that The weights have been saved correctly and are the same as before
        self.assertTrue([(w1 == w2).all() for (w1, w2) in zip(
            encoder1.get_weights(), encoder2.get_weights())])
        self.assertTrue([(w1 == w2).all() for (w1, w2) in zip(
            decoder1.get_weights(), decoder2.get_weights())])
        os.remove(filename)

    def test_channel1(self):
        """Test that a nchannels=1 input gets treated the same as no channels"""
        n_samples = 1000
        size = 64
        X1 = np.random.rand(n_samples, size, size)
        X2 = X1[..., np.newaxis]
        X_test = np.random.rand(100, size, size)

        vae1 = CVAE(shape=(28, 28), random_state=0)
        vae1.build_lenet5(10)
        vae1.train(X1, epochs=1)
        score1 = vae1.score(X_test)

        vae2 = CVAE(shape=(28, 28), random_state=0)
        vae2.build_lenet5(10)
        vae2.train(X2, epochs=1)
        score2 = vae2.score(X_test)
        # test if all scores are the same
        self.assertTrue([(s1 == s2).all() for (s1, s2) in zip(score1, score2)])


if __name__ == "__main__":
    print("Run unittesting...")
    suite = unittest.TestSuite()
    suite.addTest(CVAE_Testing("test_resize"))
    suite.addTest(CVAE_Testing("test_different_sizes"))
    suite.addTest(CVAE_Testing("test_save_load"))
    suite.addTest(CVAE_Testing("test_channel1"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
    print("Run actual experiment...")
    experiment3()
