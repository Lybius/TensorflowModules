import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tqdm import tqdm
import h5py
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import TerminateOnNaN, ModelCheckpoint, EarlyStopping

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


    def train(self, X, epochs=10, batch_size=50, X_target=None, validation_split=0.1, **kwargs):
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
        vae.compile(optimizer=tf.optimizers.Adam(clipvalue=1.0,**kwargs),
                    loss=negative_log_likelihood)
        # Define Callbacks for training
        filepath="weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
        early_stopping=EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
        callbacks_list = [
                            TerminateOnNaN(),
                            checkpoint,
                            early_stopping
                        ]
        # fit data to reconstruct itself
        if X_target is None: # else train alternative target representation
            X_target=X
        vae.fit(X, X_target, 
            batch_size=batch_size, 
            epochs=epochs, 
            validation_split=validation_split,
            callbacks = callbacks_list)



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
    
    def build_LSTM(self, encoded_size, n_lstm=100,l1=0.0,l2=0.0,dropout_rate=0.5,mode="normal"):
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
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.LSTM(n_lstm, 
                                activation='relu',
                                name="encoder_lstm",
                                kernel_regularizer=l1_l2(l1=l1, l2=l2),
                                recurrent_regularizer=l1_l2(l1=l1, l2=l2),
                                activity_regularizer=l1_l2(l1=l1, l2=l2)),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(
                encoded_size), activation=None, name="encoder_dense"),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tfp.layers.MultivariateNormalTriL(encoded_size, activity_regularizer=tfp.layers.KLDivergenceRegularizer(
                prior, weight=1.0), name="encoder_multivariate_normal")
        ])
        # Decoder
        self.__decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(
                input_shape=[encoded_size], name="decoder_input"),
            tf.keras.layers.RepeatVector(timesteps, name="repeat"),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.LSTM(n_lstm, 
                                activation='relu',
                                return_sequences=True, 
                                name="decoder_lstm",
                                kernel_regularizer=l1_l2(l1=l1, l2=l2),
                                recurrent_regularizer=l1_l2(l1=l1, l2=l2),
                                activity_regularizer=l1_l2(l1=l1, l2=l2)),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        if mode=="normal":
            self.__decoder.add(
                tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(
                    channels), name="decoder_dense"),
                name="time_distributor")
            )  
            self.__decoder.add(
                tfp.layers.IndependentNormal(channels, name="decoder_normal")
            )
        elif mode=="bernoulli":
            self.__decoder.add(
                    tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(tfp.layers.IndependentBernoulli.params_size(
                        channels), name="decoder_dense"),
                    name="time_distributor")
                )
            self.__decoder.add(
                tf.keras.layers.Dropout(dropout_rate)
            )
            self.__decoder.add(
                tf.keras.layers.BatchNormalization()
            )
            self.__decoder.add(
                tfp.layers.IndependentBernoulli(channels, name="decoder_bernoulli")
            )          
            
        
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


if __name__ == "__main__":
    pass
    # series=np.sin(np.linspace(-30,30,1000))
    # vae=ShiftVAE(20,shift=3,channels=1)
    # vae.build_LSTM(2,20)
    # vae.train(series,epochs=1)
    # x=np.cos(np.linspace(-30,30,1000))
    # print(vae.predict(x))