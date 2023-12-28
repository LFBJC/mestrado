import os
import tensorflow as tf
import pandas as pd
import numpy as np
from utils import MMRE, images_and_targets_from_data_series, plot_multiple_box_plot_series
from matplotlib import pyplot as plt

base_path = "E:/mestrado/Pesquisa/Dados simulados/Saída da otimização de hiperparâmetros v7"
data_path = "E:/mestrado/Pesquisa/Dados simulados/Dados"
for steps_ahead in os.listdir(base_path):
    for data_index in os.listdir(f"{base_path}/{steps_ahead}"):
        data = pd.read_csv(f"{data_path}/{data_index}.csv").to_dict('records')
        model = tf.keras.models.load_model(
            f'{base_path}/{steps_ahead}/{data_index}/best_model.h5',
            custom_objects={'MMRE': MMRE}
        )
        opt_hist = pd.read_csv(f'{base_path}/{steps_ahead}/{data_index}/opt_hist.csv')
        win_size = opt_hist[opt_hist['score'] == opt_hist['score'].min()]['win_size'].iloc[0]
        X, Y, inverse_normalizations, normalizations = images_and_targets_from_data_series(
            data, input_win_size=win_size, steps_ahead=int(steps_ahead.replace(' steps ahead', ''))
        )
        # Y_ = np.array([f(y) for f, y in zip(normalizations, Y)])
        for dim in range(1, Y.shape[1]):
            Y[:, dim] += Y[:, dim - 1]
        T = model.predict(X, verbose=0)
        for dim in range(1, T.shape[1]):
            T[:, dim] += T[:, dim - 1]
        T = np.array([f(t) for f, t in zip(inverse_normalizations, T)])
        assert (all([b_plot[0] <= b_plot[1] <= b_plot[2] <= b_plot[3] <= b_plot[4] for b_plot in T]))
        plot_multiple_box_plot_series([
            [{'whislo': row[0], 'q1': row[1], 'med': row[2], 'q3': row[3], 'whishi': row[4]} for row in Y[:, -1, :].squeeze()],
            [{'whislo': row[0], 'q1': row[1], 'med': row[2], 'q3': row[3], 'whishi': row[4]} for row in T]
        ])