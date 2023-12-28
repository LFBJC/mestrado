import os

import matplotlib.pyplot as plt
import optuna
import numpy as np
import pandas as pd
from utils import create_model, MMRE, MMRE_Loss, images_and_targets_from_data_series, plot_multiple_box_plot_series, plot_single_box_plot_series
from tqdm import tqdm
import tensorflow as tf
caminho_de_saida = "E:/mestrado/Pesquisa/Dados simulados/Saída da otimização de hiperparâmetros v10"

def objective(trial, study, data_set_index, steps_ahead):
    train_data = pd.read_csv(f'E:/mestrado/Pesquisa/Dados simulados/Dados/{data_set_index}.csv').to_dict('records')
    # plot_single_box_plot_series(train_data)
    os.makedirs(f'{caminho_de_saida}/{steps_ahead} steps ahead/{data_set_index}', exist_ok=True)
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
    model = create_model(
        input_shape=(win_size, 5, 1),
        filters_conv_1=filters_conv_1, kernel_size_conv_1=kernel_size_conv_1,
        activation_conv_1=activation_conv_1,
        pool_size_1=pool_size_1, pool_type_1=pool_type_1,
        filters_conv_2=filters_conv_2, kernel_size_conv_2=kernel_size_conv_2,
        activation_conv_2=activation_conv_2,
        dense_neurons=dense_neurons, dense_activation='sigmoid'
    )
    N_EPOCHS = 50
    X, Y, inverse_normalizations, normalizations = images_and_targets_from_data_series(
        train_data, input_win_size=win_size, steps_ahead=steps_ahead
    )
    model.compile(
        optimizer=trial.suggest_categorical('optimizer', ['adam', 'adadelta', 'adagrad', 'rmsprop', 'sgd']),
        loss=trial.suggest_categorical('loss', [
            'mean_squared_error', 'mean_squared_logarithmic_error', 'mean_absolute_percentage_error',
            'mean_absolute_error' # , MMRE_Loss(inverse_normalizations)
        ]),
        # loss=MMRE_Loss(inverse_normalizations),
        metrics=[MMRE]
    )
    Y_ = np.array([f(y) for f, y in zip(normalizations, Y)])
    Y_[:, :, 1:] -= Y_[:, :, :-1]
    model.fit(X, Y_[:, 0, :], batch_size=X.shape[0], epochs=N_EPOCHS, verbose=0)
    T = model.predict(X, verbose=0)
    for dim in range(1, T.shape[1]):
        T[:, dim] += T[:, dim - 1]
    if not (all([b_plot[0] <= b_plot[1] <= b_plot[2] <= b_plot[3] <= b_plot[4] for b_plot in T])):
        raise ValueError(f"DEU ERRO ANTES DE INVERTER A NORMALIZAÇÃO {b_plot}")
    T = np.array([f(t) for f, t in zip(inverse_normalizations, T)])
    if not (all([b_plot[0] <= b_plot[1] <= b_plot[2] <= b_plot[3] <= b_plot[4] for b_plot in T])):
        raise ValueError(f"DEU ERRO APÓS DE INVERTER A NORMALIZAÇÃO {b_plot}")
    # plot_multiple_box_plot_series([
    #     [{'whislo': row[0], 'q1': row[1], 'med': row[2], 'q3': row[3], 'whishi': row[4]} for row in Y_.squeeze()],
    #     [{'whislo': row[0], 'q1': row[1], 'med': row[2], 'q3': row[3], 'whishi': row[4]} for row in T]
    # ])
    error = 0
    if steps_ahead == 1:
        error = tf.keras.backend.eval(MMRE(Y[:, 0, :], T))
    else:
        for s in range(2, steps_ahead+1):
            T = model.predict(X, verbose=0)
            for dim in range(1, T.shape[1]):
                T[:, dim] += T[:, dim - 1]
            T = T.reshape(T.shape[0], 1, -1, 1)
            T = np.array([f(t) for f, t in zip(inverse_normalizations, T)])
            X_ = np.concatenate([X[:, 1:, :, :], T], axis=1)
            if s == steps_ahead:
                predicted = model.predict(X_)
                for dim in range(1, predicted.shape[1]):
                    predicted[:, dim] += predicted[:, dim - 1]
                error = tf.keras.backend.eval(MMRE(Y[:, s-1, :], np.array([f(t) for f, t in zip(inverse_normalizations, predicted)])))
    if np.isnan(error):
        print(debug)
    try:
        if error < study.best_value:
            model.save(f'{caminho_de_saida}/{steps_ahead} steps ahead/{data_set_index}/best_model.h5')
    except ValueError:
        # if no trials are completed yet save the first trial
        model.save(f'{caminho_de_saida}/{steps_ahead} steps ahead/{data_set_index}/best_model.h5')
    opt_hist_df = pd.DataFrame.from_records([trial.params])
    opt_hist_df['w2'] = w2
    opt_hist_df['w3'] = w3
    opt_hist_df['w4'] = w4
    opt_hist_df['h4'] = h4
    opt_hist_df['score'] = error
    hist_path = f'{caminho_de_saida}/{steps_ahead} steps ahead/{data_set_index}/opt_hist.csv'
    append_condition = os.path.exists(hist_path)
    print('append_condition:', append_condition)
    opt_hist_df.to_csv(hist_path, mode='a' if append_condition else 'w', index=False, header=(not append_condition))
    return error


steps_ahead = 20 # [1, 5, 20]
n_trials = 100
for data_set_index in range(0, 10, 2):
    os.makedirs(f'{caminho_de_saida}/{steps_ahead} steps ahead/{data_set_index}', exist_ok=True)
    if not os.path.exists(f'{caminho_de_saida}/{steps_ahead} steps ahead/{data_set_index}/opt_hist.csv'):
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=30),
            study_name=f'hyperparameter_opt_{data_set_index}'
        )
        study.optimize(lambda trial: objective(trial, study, data_set_index, steps_ahead), n_trials=n_trials)
    else:
        df_temp = pd.read_csv(f'{caminho_de_saida}/{steps_ahead} steps ahead/{data_set_index}/opt_hist.csv')
        if df_temp.shape[0] == 0:
            study = optuna.create_study(
                direction='minimize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=30),
                study_name=f'hyperparameter_opt_{data_set_index}'
            )
            study.optimize(lambda trial: objective(trial, study, data_set_index, steps_ahead), n_trials=n_trials)
        elif df_temp.shape[0] < n_trials:
            study = optuna.create_study(
                direction='minimize',
                study_name=f'hyperparameter_opt_{data_set_index}'
            )
            for _, row in df_temp.iterrows():
                study.add_trial(
                    optuna.trial.create_trial(
                        params={c: v for c, v in row.to_dict().items() if c not in ['w2', 'w3', 'w4', 'h4', 'score']},
                        distributions={
                            'win_size': optuna.distributions.IntDistribution(100, 800),
                            'filters_conv_1': optuna.distributions.IntDistribution(12, 50),
                            'kernel_size_conv_1[0]': optuna.distributions.IntDistribution(row['win_size']//5, row['win_size']//3),
                            'kernel_size_conv_1[1]': optuna.distributions.IntDistribution(1, 5),
                            'activation_conv_1': optuna.distributions.CategoricalDistribution(['relu', 'elu', 'sigmoid', 'linear', 'tanh', 'swish']),
                            'pool_size_1[0]': optuna.distributions.IntDistribution(1, (row['win_size'] - row['kernel_size_conv_1[0]'] + 1)//2),
                            'pool_type_1': optuna.distributions.CategoricalDistribution(['max', 'average']),
                            'filters_conv_2': optuna.distributions.IntDistribution(6, row['filters_conv_1'] // 2),
                            'kernel_size_conv_2[0]': optuna.distributions.IntDistribution(1, row['w3'] - 1),
                            'activation_conv_2': optuna.distributions.CategoricalDistribution(['relu', 'elu', 'sigmoid', 'linear', 'tanh', 'swish']),
                            'dense_neurons': optuna.distributions.IntDistribution(5, row['w4']*row['h4']*row['filters_conv_2'] - 1),
                            'optimizer': optuna.distributions.CategoricalDistribution(['adam', 'adadelta', 'adagrad', 'rmsprop', 'sgd']),
                            'loss': optuna.distributions.CategoricalDistribution([
                                'mean_squared_error', 'mean_squared_logarithmic_error', 'mean_absolute_percentage_error',
                                'mean_absolute_error'
                            ])
                        },
                        value=row['score']
                    )
                )
            study.optimize(lambda trial: objective(trial, study, data_set_index, steps_ahead), n_trials=n_trials-df_temp.shape[0])
            del df_temp
        else:
            del df_temp
os.system('shutdown -s')