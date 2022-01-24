import tensorflow as tf

def get_data(valid_dataset_size):
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    X_vaild, X_train = X_train_full[:valid_dataset_size] / 255., X_train_full[valid_dataset_size: ] / 255.
    y_vaild, y_train = y_train_full[:valid_dataset_size], y_train_full[valid_dataset_size:]
    X_test = X_test / 255.

    return (X_train, y_train), (X_vaild, y_vaild), (X_test, y_test)
