import numpy as np
import tensorflow as tf
from math import floor
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.models import load_model


def calc_euclidian_dists(x, y):
    """
    Calculate euclidian distance between two 3D tensors.

    Args:
        x (tf.Tensor):
        y (tf.Tensor):

    Returns (tf.Tensor): 2-dim tensor with distances.

    """
    n = x.shape[0]
    m = y.shape[0]
    x = tf.tile(tf.expand_dims(x, 1), [1, m, 1])
    y = tf.tile(tf.expand_dims(y, 0), [n, 1, 1])
    return tf.reduce_mean(tf.math.pow(x - y, 2), 2)


class Prototypical(Model):
    """
    Implemenation of Prototypical Network.
    """
    def __init__(self, n_support, n_query, n_test_class, n_train_class, w, h, c):
        """
        Args:
            n_support (int): number of support examples.
            n_query (int): number of query examples.
            n_test_class (int): number of test classes.
            n_train_class (int): number of training classes.
            w (int): image width .
            h (int): image height.
            c (int): number of channels.
        """
        super(Prototypical, self).__init__()
        self.w, self.h, self.c = w, h, c
        self.n_support = n_support
        self.n_query = n_query
        self.n_test_class = n_test_class
        self.n_train_class = n_train_class

        initializer = tf.keras.initializers.GlorotNormal(seed=2025)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.w, self.h, self.c))
        ])
        lb = min(self.w, self.h)
        while floor(lb / 2) >= 1:
            self.encoder.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', kernel_initializer=initializer))
            self.encoder.add(tf.keras.layers.BatchNormalization())
            self.encoder.add(tf.keras.layers.ReLU())
            self.encoder.add(tf.keras.layers.MaxPool2D((2, 2)))
            lb = lb / 2
        
    def metrics(self, log_p_y, y, n_class):
        metrics_dict = {}

        # accuracy computation
        y_pred = tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32)
        y = tf.cast(y, tf.int32)
        eq = tf.cast(tf.equal(y_pred, y), tf.float32)
        metrics_dict['accuracy'] = tf.reduce_mean(eq)

        # precision and recall per class
        prec_list = []
        recall_list = []
        tp_list = []
        fp_list = []
        fn_list = []
        for c in range(n_class):
            y_pred_c = tf.cast(tf.equal(y_pred, c), tf.int32)
            y_c = tf.cast(tf.equal(y, c), tf.int32)
            tp = tf.reduce_sum(tf.cast(tf.equal(y_pred_c, y_c), tf.float32))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(y_pred_c, y_c),
                                    tf.not_equal(y_pred_c, -1)), tf.float32))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(y_pred_c, y_c),
                                    tf.not_equal(y_c, -1)), tf.float32))
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            prec_list.append(precision)
            recall_list.append(recall)
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)

        # precision macro averaged
        metrics_dict['prec_macro'] = tf.reduce_mean(tf.stack(prec_list))

        # precision micro averaged
        metrics_dict['prec_micro'] = tf.reduce_sum(tf.stack(tp_list)) / (tf.reduce_sum(tf.stack(tp_list)) +
                                                          tf.reduce_sum(tf.stack(fp_list)) + 1e-8)
        
        # recall macro averaged
        metrics_dict['recall_macro'] = tf.reduce_mean(tf.stack(recall_list))

        # recall micro averaged
        metrics_dict['recall_micro'] = tf.reduce_sum(tf.stack(tp_list)) / (tf.reduce_sum(tf.stack(tp_list)) +
                                                            tf.reduce_sum(tf.stack(fn_list)) + 1e-8)
        return metrics_dict
    
    def call(self, s_files, s_label, q_files, q_label):
        print(f'support shape: {s_files.shape}, query shape: {q_files.shape}')
 
        y = q_label
        y_onehot = tf.one_hot(y, depth=self.n_test_class, dtype=tf.float32)
        
        z_prototypes = self.encoder(s_files)
        z_prototypes = tf.reshape(z_prototypes, (self.n_train_class, self.n_support, -1))
        z_prototypes = tf.math.reduce_mean(z_prototypes, axis=1)

        z_query = self.encoder(q_files)

        # Calculate distances between query and prototypes
        dists = calc_euclidian_dists(z_query, z_prototypes)

        # log softmax of calculated distances
        log_p_y = tf.nn.log_softmax(-dists, axis=-1)
        log_p_y = tf.reshape(log_p_y, [self.n_test_class, self.n_query, -1])
        
        loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_onehot, log_p_y), axis=-1), [-1]))
        metrics = self.metrics(log_p_y, y, self.n_test_class)
        return loss, metrics

    def save(self, model_path):
        """
        Save encoder to the file.

        Args:
            model_path (str): path to the .h5 file.

        Returns: None

        """
        self.encoder.save(model_path)

    def load(self, model_path):
        """
        Load encoder from the file.

        Args:
            model_path (str): path to the .h5 file.

        Returns: None

        """
        self.encoder(tf.zeros([1, self.w, self.h, self.c]))
        self.encoder.load_weights(model_path)
