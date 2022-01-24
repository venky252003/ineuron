import tensorflow as tf
import time
import os

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUMBER_CLASSES):
    LAYERS = [
        tf.keras.layers.Flatten(input_shape=[28,28], name='Input_Layer'),
        tf.keras.layers.Dense(300, activation='relu', name='hiddenLayer1'),
        tf.keras.layers.Dense(100, activation='relu', name='hiddenLayer2'),
        tf.keras.layers.Dense(NUMBER_CLASSES, activation='softmax', name='outputLayer')
    ]

    model_CLF = tf.keras.models.Sequential(LAYERS)
    model_CLF.summary()
    model_CLF.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)

    return model_CLF

def save_model(model, save_path, model_dir):
    unique_name = get_unique_path(save_path)
    path_to_model = os.path.join(model_dir, unique_name)
    model.save(path_to_model)

def get_unique_path(file_name):
    unique_file_name = time.strftime(f"%Y%m%d_%H%M%S_{file_name}")
    return unique_file_name
