import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, Input, Model
from keras import backend as K


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


def create_model(input_shape=(20, 5, 1)):
    input = Input(shape=input_shape)
    conv_1 = layers.Conv2D(32, (input_shape[0]//5, 1), activation='relu', input_shape=input_shape)(input)
    max_pooling_1 = layers.MaxPooling2D((2, 2))(conv_1)
    conv_2 = layers.Conv2D(16, (1, 2), activation='relu')(max_pooling_1)
    flatten = layers.Flatten()(conv_2)
    hidden_dense = layers.Dense(16, activation='relu')(flatten)
    out_min = layers.Dense(1)(hidden_dense)
    out_ranges = layers.Dense(input_shape[1]-1, activation='relu')(hidden_dense)
    out = layers.concatenate([out_min, out_ranges])
    return Model(inputs=[input], outputs=[out])


def MMRE(y_true, y_pred):
    return K.mean(K.abs(y_true-y_pred)/y_true)


if __name__ == "__main__":
    plot_single_box_plot_series(aggregate_data_by_chunks(random_walk(), 100))
    plot_multiple_box_plot_series([aggregate_data_by_chunks(random_walk(), 100), aggregate_data_by_chunks(random_walk(), 100)])
