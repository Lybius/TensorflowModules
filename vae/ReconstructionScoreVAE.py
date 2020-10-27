import numpy as np
import tensorflow as tf
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, accuracy_score, auc, confusion_matrix, confusion_matrix, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_curve
from VariationalAutoencoder import VAE

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