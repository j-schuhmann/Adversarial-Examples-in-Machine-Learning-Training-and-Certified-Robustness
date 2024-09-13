import tensorflow as tf
import numpy as np
import tensorflow as tf


def custom_sign(x):
        return tf.sign(x)

def model_with_sign(model):
    model_with_sign = tf.keras.models.Sequential([
        model,                       
        tf.keras.layers.Lambda(custom_sign)          
    ])
    return model_with_sign



def print_accuracy(model,x_image,y_image,adversarial=False,robustness_experiment=None):
    'robustness_experiment corresponds to the experiments with certfied robustness'
    if robustness_experiment:
        predictions = model_with_sign(model).predict(x_image)
        accuracy = np.mean(np.equal(y_image, predictions.flatten()))
    else:
        predictions = model(x_image)
        predictions=np.argmax(predictions, axis=1)
        y_image=np.argmax(y_image, axis=1)
        accuracy = np.mean(np.equal(y_image, predictions))
    
        
    if adversarial:
        print(f"Accuracy on adversarial test set: {accuracy}")
    else:
        print(f"Accuracy on test set: {accuracy}")
    return accuracy


def positive_negative_parts(A):

    A_abs = tf.abs(A)

    A_positive = 0.5 * (A + A_abs)

    A_negative = 0.5 * (A_abs - A)

    return A_positive, A_negative



def mnist_data_and_preprocessing():

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    

    x_train=x_train / 255.0  # normalize
    x_test=x_test/255.0

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
    y_test = tf.one_hot(y_test.astype(np.int32), depth=10)
    
    return x_train, y_train, x_test, y_test

def mnist_binary_data_and_preprocessing():

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_filter = np.where((y_train == 1) | (y_train == 2))
    test_filter = np.where((y_test == 1) | (y_test == 2))

    x_train, y_train = x_train[train_filter], y_train[train_filter] # only want 1,2 because binary classification
    x_test, y_test = x_test[test_filter], y_test[test_filter]

    x_train=x_train / 255.0  # normalize
    x_test=x_test/255.0

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    y_train[y_train == 2] = -1 # want to predict {-1,1} instead of {1,2}
    y_test[y_test == 2] = -1
    
    return x_train, y_train, x_test, y_test


def mnist_binary_all_data_and_preprocessing():

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    y_train = np.where(y_train < 5, 1, -1)
    y_test = np.where(y_test < 5, 1, -1)

    x_train=x_train / 255.0  # normalize
    x_test=x_test/255.0

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    return x_train, y_train, x_test, y_test

    
    



def custom_sign_metric_for_training(y_true, y_pred):
    y_pred_sign = tf.sign(y_pred)
    
    y_pred_sign = tf.squeeze(y_pred_sign)  
   
    y_true = tf.cast(y_true, tf.int32)
    
    y_pred_sign = tf.cast(y_pred_sign, tf.int32)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred_sign), tf.float32))

    return accuracy
