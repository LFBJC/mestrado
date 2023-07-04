import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import metrics
from utils import aggregate_data_by_chunks, random_walk, plot_multiple_box_plot_series, \
    plot_single_box_plot_series, images_and_targets_from_data_series, create_model, MMRE

np.seterr(all='raise')
np.random.seed(19091996)
# TODO split into training and test
data = aggregate_data_by_chunks(random_walk(n_samples=2000), chunk_size=10)
plot_single_box_plot_series(data, splitters=[len(data)//2, 3*len(data)//4])
train_data, val_data, test_data = data[:len(data)//2], data[len(data)//2:3*len(data)//4], data[3*len(data)//4:]
input_win_size = 20
N_EPOCHS = 50
model = create_model()
model.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=[MMRE])
X = np.zeros((1, input_win_size, 5, 1))
train_losses = []
val_losses = []
epochs_no_improve = 0
for epoch in tqdm(range(N_EPOCHS)):
    if epochs_no_improve < 10:
        train_losses_for_this_epoch = []
        for image, target_bbox, inverse_normalization in images_and_targets_from_data_series(
                train_data, input_win_size=input_win_size
        ):
            # plt.imshow(image)
            X[0, :, :, :] = image

            # create a 2D target tensor with shape (batch_size, output_dim)
            y = np.array(list(target_bbox)).reshape((1, -1))
            # predict ranges instead of bbox values
            y[:, 1:] -= y[:, :-1]

            # train the model on the input-output pair for one epoch
            model.fit(X, y, batch_size=1, epochs=1, verbose=0)
            train_losses_for_this_epoch.append(model.evaluate(x=X, y=y, verbose=0))

            pred = model.predict(X, verbose=0)
            # convert range to actual bounding box
            pred[:, 1] += pred[:, 0]
            pred[:, 2] += pred[:, 1]
            pred[:, 3] += pred[:, 2]
            pred[:, 4] += pred[:, 3]
            pred = inverse_normalization(pred)

        train_losses.append(np.mean(train_losses_for_this_epoch))
        del train_losses_for_this_epoch

        val_losses_for_this_epoch = []
        for image, target_bbox, inverse_normalization in images_and_targets_from_data_series(
                train_data, input_win_size=input_win_size
        ):
            X[0, :, :, :] = image
            pred = model.predict(X, verbose=0)
            # create a 2D target tensor with shape (batch_size, output_dim)
            y = np.array(list(target_bbox)).reshape((1, -1))
            # predict ranges instead of bbox values
            y[:, 1:] -= y[:, :-1]
            val_losses_for_this_epoch.append(metrics.mean_squared_error(y, pred))
        val_losses.append(np.mean(val_losses_for_this_epoch))
        del val_losses_for_this_epoch
        if len(val_losses) > 2 and val_losses[-1] >= val_losses[-2]:
            epochs_no_improve += 1
plt.figure()
plt.plot(train_losses, label='train')
plt.plot(val_losses, label='val')
plt.legend()
plt.show()
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
    test_losses.append(metrics.mean_squared_error(y, pred))
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
print(np.mean(test_losses))
plot_multiple_box_plot_series([ground_truth, predictions])
