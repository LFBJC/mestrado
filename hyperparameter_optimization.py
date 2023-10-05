import os

import matplotlib.pyplot as plt
import optuna
import numpy as np
import pandas as pd
from utils import create_model, MMRE, images_and_targets_from_data_series
from tqdm import tqdm
import tensorflow as tf

steps_ahead = [1, 5, 20]

def objective(trial, study, data_set_index):
    train_data = pd.read_csv(f'data/{data_set_index}.csv').to_dict('records')
    os.makedirs(f'hyperparameter_optimization_output/{data_set_index}', exist_ok=True)
    win_size = trial.suggest_int('win_size', 100, 800)
    filters_conv_1 = trial.suggest_int('filters_conv_1', 12, 50)
    kernel_size_conv_1 = (
        trial.suggest_int('kernel_size_conv_1[0]', win_size//5, win_size//3),
        trial.suggest_int('kernel_size_conv_1[1]', 1, 5)
    )
    activation_conv_1 = trial.suggest_categorical('activation_conv_1', ['relu', 'elu', 'sigmoid', 'linear', 'tanh', 'swish'])
    pool_size_1 = (
        trial.suggest_int('pool_size_1[0]', 1, (win_size - kernel_size_conv_1[0] + 1)//2),
        1
    )
    pool_type_1 = trial.suggest_categorical('pool_type_1', ['max', 'average'])
    filters_conv_2 = trial.suggest_int('filters_conv_2', 6, filters_conv_1 // 2)
    w2 = (win_size - kernel_size_conv_1[0]) + 1
    w3 = (w2 - pool_size_1[0])//pool_size_1[0] + 1
    kernel_size_conv_2 = (
        trial.suggest_int('kernel_size_conv_2[0]', 1, w3 - 1),
        1
    )
    print(f'w3: {w3}\nkernel conv 2: {kernel_size_conv_2}')
    w4 = (w3 - kernel_size_conv_2[0]) + 1
    h4 = (7 - kernel_size_conv_1[1] - pool_size_1[1])//pool_size_1[1] - kernel_size_conv_2[1] + 1
    activation_conv_2 = trial.suggest_categorical('activation_conv_2',
                                                  ['relu', 'elu', 'sigmoid', 'linear', 'tanh', 'swish'])
    dense_neurons = trial.suggest_int('dense_neurons', 5, w4*h4*filters_conv_2 - 1)
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
    train_losses_by_steps_ahead = []
    X, Y, inverse_normalizations = images_and_targets_from_data_series(
        train_data, input_win_size=win_size, steps_ahead=max(steps_ahead)
    )
    Y_ = Y
    Y_[:, :, 1:] -= Y_[:, :, :-1]
    model.fit(X, Y_[:, 0, :], batch_size=X.shape[0], epochs=N_EPOCHS, verbose=0)
    T = model.predict(X, verbose=0)
    if 1 in steps_ahead:
        train_losses_by_steps_ahead.append(tf.keras.backend.eval(MMRE(Y_[:, 0, :], T)))
    for s in range(2, max(steps_ahead)+1):
        T = model.predict(X, verbose=0).reshape(T.shape[0], 1, -1, 1)
        X_ = np.concatenate([X[:, 1:, :, :], T], axis=1)
        if s in steps_ahead:
            train_losses_by_steps_ahead.append(
                tf.keras.backend.eval(MMRE(Y_[:, s-1, :], model.predict(X_)))
            )
    error = np.mean(train_losses_by_steps_ahead)
    if np.isnan(error):
        error = -1
    opt_hist_df = pd.DataFrame.from_records([trial.params])
    opt_hist_df['w2'] = w2
    opt_hist_df['w3'] = w3
    opt_hist_df['w4'] = w4
    opt_hist_df['h4'] = h4
    opt_hist_df['score'] = error
    opt_hist_df.to_csv(f'hyperparameter_optimization_output/opt_hist.csv', mode='a', index=False, header=(trial.number == 0))
    return error


data_set_index = 0
study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=30),
    study_name=f'hyperparameter_opt_{data_set_index}'
)
study.optimize(lambda trial: objective(trial, study, data_set_index), n_trials=200)
