import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, Input, Model
from keras import backend as K
from typing import Literal


def random_walk(n_samples: int = 10000, begin_value: float=None):
    if begin_value is None:
        begin_value = np.random.rand()
    out = [begin_value]
    for step in range(n_samples-1):
        out.append(out[-1] + (np.random.rand()*2 - 1))
    return out


def aggregate_data_by_chunks(data, chunk_size: int = 10):
    box_plots = []
    for chunk in range(int(np.ceil(len(data)/chunk_size))):
        chunk_data = data[chunk*chunk_size:chunk*chunk_size + chunk_size]
        box_plots.append({
            'whislo': min(chunk_data),
            'q1': np.quantile(chunk_data, 0.25),
            'med': np.quantile(chunk_data, 0.5),
            'q3': np.quantile(chunk_data, 0.75),
            'whishi': max(chunk_data)
        })
    return box_plots


def plot_single_box_plot_series(box_plot_series, splitters=[]):
    fig, ax = plt.subplots()
    ax.bxp(box_plot_series, showfliers=False)
    if splitters:
        for splitter in splitters:
            ax.axvline(splitter)
    plt.show()


def plot_multiple_box_plot_series(box_plot_series):
    if len(box_plot_series) > 1:
        assert all([len(x) == len(box_plot_series[0]) for x in box_plot_series[1:]])
        # positions = list(range(len(box_plot_series[0])))*len(box_plot_series)
        colors = ['green', 'red', 'blue']
        if len(box_plot_series) > 3:
            import random
            for _ in range(len(box_plot_series) - 3):
                r = lambda: random.randint(0, 255)
                colors.append('#%02X%02X%02X' % (r(), r(), r()))
        fig, ax = plt.subplots()
        for i, b_series in enumerate(box_plot_series):
            ax.bxp(b_series, boxprops={'color': colors[i]}, showfliers=False)
        plt.show()


def images_and_targets_from_data_series(data, input_win_size=20):
    for i in range(len(data)-1-input_win_size):
        image_data = data[i:input_win_size+i]
        image = np.array([[[v] for v in list(bbox.values())] for bbox in image_data])
        inverse_normalization = lambda x: (np.max(image) - np.min(image))*x + np.min(image)
        image = (image - np.min(image))/(np.max(image) - np.min(image))
        targets = np.array(list(data[input_win_size+i].values()))
        targets = (targets - np.min(image))/(np.max(image) - np.min(image))
        yield image, targets, inverse_normalization


def create_model(
    input_shape=(20, 5, 1), filters_conv_1=32, kernel_size_conv_1=(4, 1), activation_conv_1='relu',
    pool_size_1=(2, 2), pool_type_1: Literal["max", "average"] = "max",
    filters_conv_2=16, kernel_size_conv_2=(1, 2), activation_conv_2='relu',
    dense_neurons=16, dense_activation='relu'
):
    input = Input(shape=input_shape)
    conv_1 = layers.Conv2D(
        filters_conv_1, kernel_size_conv_1, activation=activation_conv_1, input_shape=input_shape
    )(input)
    if pool_type_1 == 'max':
        pooling_1 = layers.MaxPooling2D(pool_size_1)(conv_1)
    else:
        pooling_1 = layers.AveragePooling2D(pool_size_1)(conv_1)
    conv_2 = layers.Conv2D(filters_conv_2, kernel_size_conv_2, activation=activation_conv_2)(pooling_1)
    flatten = layers.Flatten()(conv_2)
    hidden_dense = layers.Dense(dense_neurons, activation=dense_activation)(flatten)
    out_min = layers.Dense(1)(hidden_dense)
    out_ranges = layers.Dense(input_shape[1]-1, activation='relu')(hidden_dense)
    out = layers.concatenate([out_min, out_ranges])
    return Model(inputs=[input], outputs=[out])


def MMRE(y_true, y_pred):
    return K.mean(K.abs((y_true-y_pred)/(y_true + K.epsilon)))


if __name__ == "__main__":
    plot_single_box_plot_series(aggregate_data_by_chunks(random_walk(), 100))
    plot_multiple_box_plot_series([aggregate_data_by_chunks(random_walk(), 100), aggregate_data_by_chunks(random_walk(), 100)])
