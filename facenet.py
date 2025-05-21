import tensorflow as tf
import numpy as np
import os
import cv2

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def load_model(model):
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print('Loading model from file:', model_exp)
        with tf.io.gfile.GFile(model_exp, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.compat.v1.import_graph_def(graph_def, name='')
    else:
        raise ValueError('Model file not found at path: {}'.format(model_exp))
