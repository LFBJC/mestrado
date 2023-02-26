import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, losses

# Load the data from Yahoo Finance
data = yf.download('AAPL')
print(data)
data.reset_index(inplace=True)


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
train_images, val_images = images[:int(0.8*images.shape[0])], images[int(0.8*images.shape[0]):]
train_labels, val_labels = labels[:int(0.8*labels.shape[0])], labels[int(0.8*labels.shape[0]):]
train_images = tf.convert_to_tensor(train_images[:, :, :, np.newaxis], dtype=tf.float32)
train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)
val_images = tf.convert_to_tensor(val_images[:, :, :, np.newaxis], dtype=tf.float32)
val_labels = tf.convert_to_tensor(val_labels, dtype=tf.float32)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(9, 7, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 2), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(7))
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
