import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import losses
from utils import aggregate_data_by_chunks, random_walk, plot_multiple_box_plot_series, \
    images_and_targets_from_data_series, create_model

data = aggregate_data_by_chunks(random_walk(n_samples=1000), chunk_size=10)
input_win_size = 20
targets_size=5
model = create_model()
model.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['mean_squared_error'])
for epoch in range(2):
    predictions = []
    for image, target_bboxes, inverse_normalization in images_and_targets_from_data_series(
            data, input_win_size=input_win_size, targets_size=targets_size
    ):
        # create a 3D input tensor with shape (batch_size, time_steps, input_dim)
        X = np.zeros((1, image.shape[0], image.shape[1], 1))
        X[0, :, :, :] = image

        # create a 2D target tensor with shape (batch_size, output_dim)
        y = np.array(list(target_bboxes[0])).reshape((1, -1))
        # predict ranges instead of bbox values
        y[:, 1:] -= y[:, :-1]

        # train the model on the input-output pair for one epoch
        print(X.shape, y.shape)
        model.fit(X, y, batch_size=1, epochs=1, verbose=0)

        pred = model.predict(X)
        # convert range to actual bounding box
        pred[:, 1] += pred[:, 0]
        pred[:, 2] += pred[:, 1]
        pred[:, 3] += pred[:, 2]
        pred[:, 4] += pred[:, 3]
        pred = inverse_normalization(pred)
        predictions.append(pred)

        # generate predictions for the remaining bounding boxes
        for bbox in target_bboxes[1:]:
            # create a new input tensor with shape (1, time_steps, input_dim)
            X = np.zeros((1, image.shape[0], image.shape[1], 1))
            pred = model.predict(X)
            # convert range to actual bounding box
            pred[:, 1] += pred[:, 0]
            pred[:, 2] += pred[:, 1]
            pred[:, 3] += pred[:, 2]
            pred[:, 4] += pred[:, 3]
            X[0, :-1, :, :] = X[0, 1:, :, :]
            X[0, -1, :, :] = tf.transpose(pred)

            # create a new target tensor with shape (1, output_dim)
            y = np.array(list(bbox)).reshape((1, -1))
            # predict ranges instead of bbox values
            y[:, 1:] -= y[:, :-1]

            # train the model on the new input-output pair for one epoch
            model.fit(X, y, batch_size=1, epochs=1, verbose=0)

            # update the input image with the predicted bounding box
            print(image.shape)
            print(X.shape)
            image = np.concatenate((image[1:, :], X[0, -1:, :, :]), axis=0)
            pred = model.predict(X)
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
ground_truth = [bbox for i in range(len(data)-targets_size-input_win_size) for bbox in data[input_win_size+i:input_win_size+i+targets_size]]
print(predictions, len(predictions))
print(ground_truth, len(ground_truth))
plot_multiple_box_plot_series([ground_truth, predictions])
