import os

import matplotlib.pyplot as plt
import optuna
import numpy as np
import pandas as pd
from utils import create_model, MMRE, images_and_targets_from_data_series
# from tqdm import tqdm


def objective(trial, study, data_set_index):
    train_data = pd.read_csv(f'data/{data_set_index}/train.csv').to_dict('records')
    val_data = pd.read_csv(f'data/{data_set_index}/val.csv').to_dict('records')
    os.makedirs(f'hyperparameter_optimization_output/{data_set_index}', exist_ok=True)
    win_size = trial.suggest_int('win_size', 10, 40)
    filters_conv_1 = trial.suggest_int('filters_conv_1', 10, 30)
    kernel_size_conv_1 = (
        trial.suggest_int('kernel_size_conv_1[0]', 2, win_size//3),
        trial.suggest_int('kernel_size_conv_1[1]', 1, 5)
    )
    activation_conv_1 = trial.suggest_categorical('activation_conv_1', ['relu', 'elu', 'sigmoid', 'linear', 'tanh', 'swish'])
    pool_size_1 = (
        trial.suggest_int('pool_size_1[0]', 1, (win_size - kernel_size_conv_1[0] + 1)//2),
        1
    )
    pool_type_1 = trial.suggest_categorical('pool_type_1', ['max', 'average'])
    filters_conv_2 = trial.suggest_int('filters_conv_2', 5, filters_conv_1 // 2)
    w2 = (win_size - kernel_size_conv_1[0]) + 1
    w3 = (w2 - pool_size_1[0])//pool_size_1[0] + 1
    del w2
    kernel_size_conv_2 = (
        trial.suggest_int('kernel_size_conv_2[0]', 1, w3),
        1
    )
    w4 = (w3 - kernel_size_conv_2[0]) + 1
    h4 = (7 - kernel_size_conv_1[1] - pool_size_1[1])//pool_size_1[1] - kernel_size_conv_2[1] + 1
    activation_conv_2 = trial.suggest_categorical('activation_conv_2',
                                                  ['relu', 'elu', 'sigmoid', 'linear', 'tanh', 'swish'])
    dense_neurons = trial.suggest_int('dense_neurons', 5, w4*h4*filters_conv_2)
    dense_activation = trial.suggest_categorical(
        'dense_activation', ['relu', 'elu', 'sigmoid', 'linear', 'tanh', 'swish']
    )
    model = create_model(
        input_shape=(win_size, 5, 1),
        filters_conv_1=filters_conv_1, kernel_size_conv_1=kernel_size_conv_1,
        activation_conv_1=activation_conv_1,
        pool_size_1=pool_size_1, pool_type_1=pool_type_1,
        filters_conv_2=filters_conv_2, kernel_size_conv_2=kernel_size_conv_2,
        activation_conv_2=activation_conv_2,
        dense_neurons=dense_neurons, dense_activation=dense_activation
    )
    model.compile(
        optimizer=trial.suggest_categorical('optimizer', ['adam', 'adadelta', 'adagrad', 'rmsprop', 'sgd']),
        loss=trial.suggest_categorical('loss', [
            'mean_squared_error', 'mean_squared_logarithmic_error', 'mean_absolute_percentage_error',
            'mean_absolute_error'
        ]),
        metrics=[MMRE]
    )
    N_EPOCHS = 50
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    X = np.zeros((1, win_size, 5, 1))
    for epoch in range(N_EPOCHS):
        if epochs_no_improve < 10:
            train_losses_for_this_epoch = []
            for image, target_bbox, inverse_normalization in images_and_targets_from_data_series(
                    train_data, input_win_size=win_size
            ):
                # plt.imshow(image)
                X[0, :, :, :] = image

                # create a 2D target tensor with shape (batch_size, output_dim)
                y = np.array(list(target_bbox)).reshape((1, -1))
                # predict ranges instead of bbox values
                y[:, 1:] -= y[:, :-1]

                # train the model on the input-output pair for one epoch
                model.fit(X, y, batch_size=1, epochs=1, verbose=0)
                train_losses_for_this_epoch.append(MMRE(y, model.predict(X, verbose=0)))
            train_losses.append(np.mean(train_losses_for_this_epoch))

            val_losses_for_this_epoch = []
            for image, target_bbox, inverse_normalization in images_and_targets_from_data_series(
                    val_data, input_win_size=win_size
            ):
                X[0, :, :, :] = image
                pred = model.predict(X, verbose=0)
                # create a 2D target tensor with shape (batch_size, output_dim)
                y = np.array(list(target_bbox)).reshape((1, -1))
                # predict ranges instead of bbox values
                y[:, 1:] -= y[:, :-1]
                val_losses_for_this_epoch.append(MMRE(y, pred))
            val_losses.append(np.mean(val_losses_for_this_epoch))
            del val_losses_for_this_epoch
            if len(val_losses) > 2 and val_losses[-1] >= val_losses[-2]:
                epochs_no_improve += 1
    error = val_losses[-1]
    try:
        if error < study.best_value:
            model.save(f'hyperparameter_optimization_output/{data_set_index}/best_model.h5')
            plt.figure()
            plt.plot(train_losses, label='train')
            plt.plot(val_losses, label='val')
            plt.legend()
            plt.savefig(f'hyperparameter_optimization_output/{data_set_index}/best_model_train_hist.png')
    except ValueError:
        # if no trials are completed yet save the first trial
        model.save(f'hyperparameter_optimization_output/{data_set_index}/best_model.h5')
        plt.figure()
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.legend()
        plt.savefig(f'hyperparameter_optimization_output/{data_set_index}/best_model_train_hist.png')
    opt_hist_df = pd.DataFrame.from_records([trial.params])
    opt_hist_df['score'] = error
    opt_hist_df.to_csv(f'hyperparameter_optimization_output/{data_set_index}/opt_hist.csv', mode='a')
    return error


data_set_index = 3
study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=30),
    study_name=f'hyperparameter_opt_{data_set_index}'
)
study.optimize(lambda trial: objective(trial, study, data_set_index), n_trials=100, n_jobs=2)
