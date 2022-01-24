import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from model import *


def save_plot(data, save_path, plot_dir):    
    unique_file_name = get_unique_path(save_path)
    plot_dir_path = os.path.join(plot_dir, unique_file_name)
    pd.DataFrame(data.history).plot(figsize=(10, 7))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.savefig(plot_dir_path)
    plt.show()


    