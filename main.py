import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers, losses

# Load the data from Yahoo Finance
original_data = yf.download('AAPL')
print(original_data.columns)
print(original_data.index.name)


def calculate_stats(x):
    stats = {}
    for c in ["Open", "Close", "Adj Close"]:
        stats[f"max_{c}"] = x[c].max()
        stats[f"3rd_quartile_{c}"] = x[c].quantile(0.75)
        stats[f"median_{c}"] = x[c].median()
        stats[f"1st_quartile_{c}"] = x[c].quantile(0.25)
        stats[f"min_{c}"] = x[c].min()
    return stats


data = original_data.groupby(
    by=[original_data.index.month, original_data.index.year]
).apply(calculate_stats)
data.index = pd.to_datetime(dict(year=data.index.get_level_values(1), month=data.index.get_level_values(0), day=1))
data = pd.DataFrame.from_records(data)


def normalization(x, batch_index):
    global data
    batch_data = data.iloc[batch_index*10:batch_index*10+9]
    delta = (batch_data.max()-batch_data.min())
    x_minus_min = (x - batch_data.min())
    return x_minus_min/delta


def inverse_normalization(x, batch_index):
    global data
    batch_data = data.iloc[batch_index*10:batch_index*10+9]
    delta = (batch_data.max()-batch_data.min())
    return x*delta + batch_data.min()


images = []
labels = []
for batch_index in range(data.shape[0]//10-1):
    img_data = data.iloc[batch_index*10:batch_index*10+9]
    label_data = data.iloc[(batch_index+1)*10]
    images.append(
        img_data.apply(
            func=(lambda x: normalization(x, batch_index)),
            axis=1
        )
    )
    labels.append(
        normalization(label_data, batch_index)
    )

images = np.array(images)
labels = np.array(labels)
train_images, val_images, test_images = images[:int(0.5*images.shape[0])], images[int(0.5*images.shape[0]):int(0.75*images.shape[0])], images[int(0.75*images.shape[0]):]
train_labels, val_labels, test_labels = labels[:int(0.5*labels.shape[0])], labels[int(0.5*labels.shape[0]):int(0.75*images.shape[0])], labels[int(0.75*images.shape[0]):]
train_images = tf.convert_to_tensor(train_images[:, :, :, np.newaxis], dtype=tf.float32)
train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)
val_images = tf.convert_to_tensor(val_images[:, :, :, np.newaxis], dtype=tf.float32)
val_labels = tf.convert_to_tensor(val_labels, dtype=tf.float32)
test_images = tf.convert_to_tensor(test_images[:, :, :, np.newaxis], dtype=tf.float32)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(9, len(data.columns), 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 2), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(data.columns)))
model.summary()
model.compile(optimizer='adam',
              loss=losses.MeanSquaredError(),
              metrics=['mean_squared_error'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(val_images, val_labels))
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# Predicted vs actual
test_labels = [inverse_normalization(x, batch_index) for batch_index, x in enumerate(test_labels)]
# test_labels = [batch_data.iloc[i] for batch_data in test_labels for i in range(batch_data.shape[0])]
predicted_labels = [inverse_normalization(pred, batch_index) for batch_index, pred in enumerate(model.predict(test_images))]
# predicted_labels = [batch_data.iloc[i] for batch_data in predicted_labels for i in range(batch_data.shape[0])]
print(test_labels)
print(predicted_labels)
plot_labels = []
original_column_names = set([
    col_name.replace("max_", "").replace("3rd_quartile_", "").replace("median_", "")\
        .replace("1st_quartile_", "").replace("min_", "")
    for col_name in data.columns
])
stats_dict = {"Actual": {c: {} for c in original_column_names}, "Predicted": {c: {} for c in original_column_names}}
for i, c in enumerate(data.columns):
    if c.startswith("max_"):
        original_column_name = c.replace("max_", "")
        stats_dict["Actual"][original_column_name]["whishi"] = [e[c] for e in test_labels]
        stats_dict["Predicted"][original_column_name]["whishi"] = [e[i] for e in predicted_labels]
    elif c.startswith("3rd_quartile_"):
        original_column_name = c.replace("3rd_quartile_", "")
        stats_dict["Actual"][original_column_name]["q3"] = [e[c] for e in test_labels]
        stats_dict["Predicted"][original_column_name]["q3"] = [e[i] for e in predicted_labels]
    elif c.startswith("median_"):
        original_column_name = c.replace("median_", "")
        stats_dict["Actual"][original_column_name]["med"] = [e[c] for e in test_labels]
        stats_dict["Predicted"][original_column_name]["med"] = [e[i] for e in predicted_labels]
    elif c.startswith("1st_quartile_"):
        original_column_name = c.replace("1st_quartile_", "")
        stats_dict["Actual"][original_column_name]["q1"] = [e[c] for e in test_labels]
        stats_dict["Predicted"][original_column_name]["q1"] = [e[i] for e in predicted_labels]
    elif c.startswith("min_"):
        original_column_name = c.replace("min_", "")
        stats_dict["Actual"][original_column_name]["whislo"] = [e[c] for e in test_labels]
        stats_dict["Predicted"][original_column_name]["whislo"] = [e[i] for e in predicted_labels]
    else:
        continue
    plot_labels.extend([f"Actual {original_column_name}", f" Predicted {original_column_name}"])
print(stats_dict)
for original_column_name in stats_dict["Actual"].keys():
    stats = []
    for i in range(len(test_labels)):
        stats.append({k: v[i] for k, v in stats_dict["Actual"][original_column_name].items()})
        stats.append({k: v[i] for k, v in stats_dict["Predicted"][original_column_name].items()})
    positions = []
    for pos in range(len(stats)//2):
        positions.extend([pos, pos])
    fig, ax = plt.subplots()
    ax.bxp(stats, positions=positions, showfliers=False)
    plt.title(f'actual vs predicted {original_column_name}')
    plt.show()
