import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import losses
from utils import aggregate_data_by_chunks, random_walk, plot_multiple_box_plot_series, \
    plot_single_box_plot_series, images_and_targets_from_data_series, create_model, MMRE

data_set_index = 0
test_data = pd.read_csv(f'data/{data_set_index}/test.csv').to_dict('records')
model = tf.keras.models.load_model(
    f'best_models/{data_set_index}/best_model.h5',
    custom_objects={"MMRE": MMRE}
)
input_win_size = model.input_shape[1]
X = np.zeros((1, input_win_size, 5, 1))
predictions = []
test_losses = []
ground_truth = []
for image, target_bbox, inverse_normalization in images_and_targets_from_data_series(
        test_data, input_win_size=input_win_size
):
    X[0, :, :, :] = image
    pred = model.predict(X, verbose=0)
    # create a 2D target tensor with shape (batch_size, output_dim)
    ground_truth.append({
        'whislo': target_bbox[0],
        'q1': target_bbox[1],
        'med': target_bbox[2],
        'q3': target_bbox[3],
        'whishi': target_bbox[4]
    })
    y = np.array(list(target_bbox)).reshape((1, -1))
    # predict ranges instead of bbox values
    y[:, 1:] -= y[:, :-1]
    test_losses.append(MMRE(y, pred))
    # convert range to actual bounding box
    pred[:, 1] += pred[:, 0]
    pred[:, 2] += pred[:, 1]
    pred[:, 3] += pred[:, 2]
    pred[:, 4] += pred[:, 3]
    pred = inverse_normalization(pred)
    predictions.append(pred)
predictions = [
    {
        'whislo': pred[0, 0],
        'q1': pred[0, 1],
        'med': pred[0, 2],
        'q3': pred[0, 3],
        'whishi': pred[0, 4]
    } for pred in predictions
]
result = np.mean(test_losses)
with open('results_log.txt', 'a+') as results_log:
    results_log.write(f'{data_set_index}: {result}\n')
os.makedirs('test_result_plots', exist_ok=True)
plot_multiple_box_plot_series([ground_truth, predictions], save_path=f'test_result_plots/{data_set_index}.png', show=False)
