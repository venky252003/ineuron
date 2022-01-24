from utils import read_config, get_data, create_model, save_model, save_plot
import argparse
import os

def training(config_path):
    config = read_config(config_path)
    validation_size = config['params']['validation_dataset']
    (X_train, y_train), (X_vaild, y_vaild), (X_test, y_test) = get_data(validation_size)

    loss_function = config['params']['loss_function']
    metric = config['params']['metric']
    optimizer = config['params']['optimizer']
    num_classes = config['params']['num_classes']

    model = create_model(loss_function, optimizer, metric, num_classes)

    EPOCHS = config['params']['epocs']
    VALIDATIONS_SET = (X_vaild, y_vaild)

    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATIONS_SET)

    artifacts_dir = config['artifacts']['artifacts_dir']
    model_dir = config['artifacts']['model_dir']

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    model_name = config['artifacts']['model_name']

    save_model(model, model_name, model_dir_path)
    plots_dir = config['artifacts']['plots_dir']
    plot_name = config['artifacts']['plot_name']

    plot_dir_path = os.path.join(artifacts_dir, plots_dir)
    os.makedirs(plot_dir_path, exist_ok=True)

    save_plot(history, plot_name, plot_dir_path)


    return model

if __name__ == '__main__':
    training("config.yaml")