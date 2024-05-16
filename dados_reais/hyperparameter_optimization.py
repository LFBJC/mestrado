import os
import sys
import matplotlib.pyplot as plt
import optuna
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import create_model, create_lstm_model, MMRE, MMRE_Loss, images_and_targets_from_data_series,\
    plot_multiple_box_plot_series, plot_single_box_plot_series, normalize_data, from_ranges, to_ranges, denormalize_data
pasta_de_entrada = f"E:/mestrado/Pesquisa/Dados reais/Dados tratados/demanda energética - kaggle"
conjunto_de_dados = pasta_de_entrada.split('/')[-1]

def objective_cnn(trial, study, steps_ahead):
    train_data = pd.read_csv(f'{pasta_de_entrada}/train.csv')
    train_data = train_data[[c for c in train_data.columns if c != "SETTLEMENTDATE"]].to_dict('records')
    val_data = pd.read_csv(f'{pasta_de_entrada}/val.csv')
    val_data = val_data[[c for c in val_data.columns if c != "SETTLEMENTDATE"]].to_dict('records')
    # plot_single_box_plot_series(train_data)
    os.makedirs(f'{caminho_de_saida}/{steps_ahead} steps ahead', exist_ok=True)
    if isinstance(train_data[0], dict):
        out_size = len(train_data[0].keys())
        available_kernel_sizes = [(2, 2), (3, 2)]
    else:
        available_kernel_sizes = [(2, 1), (3, 1)]
        out_size = 1
    print(f'OUT SIZE: {out_size}')
    win_size = trial.suggest_int('win_size', 10, len(train_data)//10)
    filters_conv_1 = trial.suggest_int('filters_conv_1', 2, 10)
    kernel_size_conv_1 = trial.suggest_categorical('kernel_size_conv_1', available_kernel_sizes)
    activation_conv_1 = trial.suggest_categorical('activation_conv_1', ['relu', 'elu', 'sigmoid', 'linear', 'tanh', 'swish'])
    pool_size_1 = trial.suggest_categorical('pool_size_1', [(2, 1), (3, 1)])
    pool_type_1 = trial.suggest_categorical('pool_type_1', ['max', 'average'])
    # filters_conv_2 = trial.suggest_int('filters_conv_2', 6, filters_conv_1 // 2)
    w2 = (win_size - kernel_size_conv_1[0]) - pool_size_1[0] + 2
    h2 = (5 - kernel_size_conv_1[0]) - pool_size_1[1] + 2
    # w3 = (w2 - pool_size_1[0])//pool_size_1[0] + 1
    # kernel_size_conv_2 = trial.suggest_categorical('kernel_size_conv_2', available_kernel_sizes)
    # print(f'w3: {w3}\nkernel conv 2: {kernel_size_conv_2}')
    # w4 = (w3 - kernel_size_conv_2[0]) + 1
    # h4 = (7 - kernel_size_conv_1[1] - pool_size_1[1])//pool_size_1[1] - kernel_size_conv_2[1] + 1
    # activation_conv_2 = trial.suggest_categorical('activation_conv_2',
    #                                               ['relu', 'elu', 'sigmoid', 'linear', 'tanh', 'swish'])
    # dense_neurons = trial.suggest_int('dense_neurons', 5, w4*h4*filters_conv_2 - 1)
    dense_neurons = trial.suggest_int('dense_neurons', 5, w2 * h2 * filters_conv_1 - 1)
    model = create_model(
        input_shape=(win_size, out_size, 1),
        filters_conv_1=filters_conv_1, kernel_size_conv_1=kernel_size_conv_1,
        activation_conv_1=activation_conv_1,
        pool_size_1=pool_size_1, pool_type_1=pool_type_1,
        # filters_conv_2=filters_conv_2, kernel_size_conv_2=kernel_size_conv_2,
        # activation_conv_2=activation_conv_2,
        dense_neurons=dense_neurons, dense_activation='sigmoid'
    )
    N_EPOCHS = 1000
    X, Y = images_and_targets_from_data_series(
        train_data, input_win_size=win_size, steps_ahead=steps_ahead
    )
    if out_size > 1:
        train_data_array = np.array([list(x.values()) for x in train_data])
    else:
        train_data_array = np.array(train_data)
    min_, max_ = np.min(train_data_array), np.max(train_data_array)
    X, Y = normalize_data(X, Y, min_, max_)
    if out_size > 1:
        X, Y = to_ranges(X), to_ranges(Y)
    model.compile(
        optimizer=trial.suggest_categorical('optimizer', ['adam', 'adadelta', 'adagrad', 'rmsprop', 'sgd']),
        loss=trial.suggest_categorical('loss', [
            'mean_squared_error', 'mean_squared_logarithmic_error', 'mean_absolute_percentage_error',
            'mean_absolute_error' # , MMRE_Loss(inverse_normalizations)
        ]),
        # loss=MMRE_Loss(inverse_normalizations),
        metrics=[MMRE]
    )
    Y_ignoring_steps_ahead = Y[:, 0, :]
    history = model.fit(
        X, Y_ignoring_steps_ahead,
        batch_size=X.shape[0],
        epochs=N_EPOCHS,
        verbose=0,
        validation_split=1/3,
        callbacks=tf.keras.callbacks.EarlyStopping(patience=10)
    )
    X_val, Y_val = images_and_targets_from_data_series(
        val_data, input_win_size=win_size, steps_ahead=steps_ahead
    )
    X_val, Y_val = normalize_data(X_val, Y_val, min_, max_)
    if out_size > 1:
        X_val, Y_val = to_ranges(X_val), to_ranges(Y_val)
        T = from_ranges(model.predict(X_val, verbose=0), axis=1)
    else:
        T = model.predict(X_val, verbose=0)
    # if not (all([b_plot[0] <= b_plot[1] <= b_plot[2] <= b_plot[3] <= b_plot[4] for b_plot in T])):
    #     raise ValueError(f"DEU ERRO ANTES DE INVERTER A NORMALIZAÇÃO {b_plot}")
    T = denormalize_data(T, min_, max_)
    # if not (all([b_plot[0] <= b_plot[1] <= b_plot[2] <= b_plot[3] <= b_plot[4] for b_plot in T])):
    #     raise ValueError(f"DEU ERRO APÓS DE INVERTER A NORMALIZAÇÃO {b_plot}")
    # plot_multiple_box_plot_series([
    #     [{'whislo': row[0], 'q1': row[1], 'med': row[2], 'q3': row[3], 'whishi': row[4]} for row in Y_.squeeze()],
    #     [{'whislo': row[0], 'q1': row[1], 'med': row[2], 'q3': row[3], 'whishi': row[4]} for row in T]
    # ])
    error = 0
    if steps_ahead == 1:
        error = tf.keras.backend.eval(MMRE(Y_val[:, 0, :], T))
    else:
        for s in range(2, steps_ahead+1):
            T = model.predict(X_val, verbose=0)
            if out_size > 1:
                T = from_ranges(T, axis=1)
            T = T.reshape(T.shape[0], 1, -1, 1)
            T = denormalize_data(T, min_, max_)
            X_ = np.concatenate([X_val[:, 1:, :, :], T], axis=1)
            if s == steps_ahead:
                predicted = model.predict(X_)
                if out_size > 1:
                    predicted = from_ranges(predicted, axis=1)
                error = tf.keras.backend.eval(MMRE(Y_val[:, s-1, :], denormalize_data(predicted, min_, max_)))
    if np.isnan(error):
        print(debug)
    try:
        if error < study.best_value:
            model.save(f'{caminho_de_saida}/{steps_ahead} steps ahead/best_model.h5')
            pickle.dump(history.history, open(f'{caminho_de_saida}/{steps_ahead} steps ahead/best_model_history.pkl', 'wb'))
    except ValueError:
        # if no trials are completed yet save the first trial
        model.save(f'{caminho_de_saida}/{steps_ahead} steps ahead/best_model.h5')
        pickle.dump(history.history, open(f'{caminho_de_saida}/{steps_ahead} steps ahead/best_model_history.pkl', 'wb'))
    opt_hist_df = pd.DataFrame.from_records([trial.params])
    # opt_hist_df['w2'] = w2
    # opt_hist_df['w3'] = w3
    # opt_hist_df['w4'] = w4
    # opt_hist_df['h4'] = h4
    opt_hist_df['score'] = error
    hist_path = f'{caminho_de_saida}/{steps_ahead} steps ahead/opt_hist.csv'
    append_condition = os.path.exists(hist_path)
    print('append_condition:', append_condition)
    opt_hist_df.to_csv(hist_path, mode='a' if append_condition else 'w', index=False, header=(not append_condition))
    return error


def objective_lstm(trial, study, steps_ahead):
    train_data = pd.read_csv(f'{pasta_de_entrada}/train.csv')
    train_data = train_data[[c for c in train_data.columns if c != "SETTLEMENTDATE"]].to_dict('records')
    val_data = pd.read_csv(f'{pasta_de_entrada}/val.csv')
    val_data = val_data[[c for c in val_data.columns if c != "SETTLEMENTDATE"]].to_dict('records')
    win_size = trial.suggest_int('win_size', 10, len(train_data) // 10)
    data = np.array([np.array(list(x.values())) for x in train_data])
    min_, max_ = np.min(data), np.max(data)
    X = np.array([data[i:i + win_size] for i, j in
                  zip(range(0, win_size * ((len(data) - steps_ahead) // win_size), win_size),
                      range((len(data) - win_size - steps_ahead) // win_size))])
    Y = np.array([data[i + win_size + steps_ahead] for i, j in
                  zip(range(0, win_size * ((len(data) - steps_ahead) // win_size), win_size),
                      range((len(data) - win_size - steps_ahead) // win_size))])
    X, Y = normalize_data(X, Y, min_, max_)
    if isinstance(train_data[0], dict):
        X, Y = to_ranges(X, axis=1), to_ranges(Y, axis=1)
    # plot_single_box_plot_series(train_data)
    os.makedirs(f'{caminho_de_saida}/{steps_ahead} steps ahead', exist_ok=True)
    num_layers = trial.suggest_int("Número de camadas", 1, 2)
    units = []
    activations = []
    recurrent_activations = []
    dropouts = []
    recurrent_dropouts = []
    for camada in range(num_layers):
        if camada == 0:
            units.append(trial.suggest_int(f"Unidades na camada {camada}", 4, 64))
        else:
            units.append(trial.suggest_int(f"Unidades na camada {camada}", 3, units[-1]))
        activations.append("sigmoid")
        recurrent_activations.append("sigmoid")
        dropouts.append(trial.suggest_float(f"Dropout da camada {camada}", 0.1, 0.5))
        recurrent_dropouts.append(trial.suggest_float(f"Dropout recorrente da camada {camada}", 0.1, 0.5))
    model = create_lstm_model(X.shape[1:], units, activations, recurrent_activations, dropouts, recurrent_dropouts)
    N_EPOCHS = 1000
    model.compile(
        optimizer=trial.suggest_categorical('optimizer', ['adam', 'adadelta', 'adagrad', 'rmsprop', 'sgd']),
        loss=trial.suggest_categorical('loss', [
            'mean_squared_error', 'mean_squared_logarithmic_error', 'mean_absolute_percentage_error',
            'mean_absolute_error'
        ]),
        metrics=[MMRE]
    )
    history = model.fit(
        X, Y,
        batch_size=len(X),
        epochs=N_EPOCHS,
        verbose=0,
        validation_split=1 / 3,
        callbacks=tf.keras.callbacks.EarlyStopping(patience=10)
    )
    if isinstance(train_data[0], dict):
        data = np.array([np.array(list(x.values())) for x in val_data])
    else:
        data = np.array(val_data)
    X_val = np.array([data[i:i + win_size] for i, j in
                       zip(range(0, win_size * ((len(data) - steps_ahead) // win_size), win_size),
                           range((len(data) - win_size - steps_ahead) // win_size))])
    Y_val = np.array([data[i + win_size + steps_ahead] for i, j in
                       zip(range(0, win_size * ((len(data) - steps_ahead) // win_size), win_size),
                           range((len(data) - win_size - steps_ahead) // win_size))])
    X_val, Y_val = normalize_data(X_val, Y_val, min_, max_)
    if isinstance(train_data[0], dict):
        X_val, Y_val = to_ranges(X_val, axis=1), to_ranges(Y_val, axis=1)
        T = from_ranges(model.predict(X_val, verbose=0), axis=1)
        error = tf.keras.backend.eval(
            MMRE(denormalize_data(from_ranges(Y_val, axis=1), min_, max_), denormalize_data(T, min_, max_))
        )
    else:
        T = model.predict(X_val, verbose=0)
        error = tf.keras.backend.eval(
            MMRE(denormalize_data(Y_val, min_, max_), denormalize_data(T, min_, max_))
        )
    try:
        if error < study.best_value:
            model.save(f'{caminho_de_saida}/{steps_ahead} steps ahead/best_model.h5')
            pickle.dump(history.history, open(f'{caminho_de_saida}/{steps_ahead} steps ahead/best_model_history.pkl', 'wb'))
    except ValueError:
        # if no trials are completed yet save the first trial
        model.save(f'{caminho_de_saida}/{steps_ahead} steps ahead/best_model.h5')
        pickle.dump(history.history, open(f'{caminho_de_saida}/{steps_ahead} steps ahead/best_model_history.pkl', 'wb'))
    hist_path = f'{caminho_de_saida}/{steps_ahead} steps ahead/opt_hist.csv'
    if os.path.exists(hist_path):
        opt_hist_df = pd.concat([pd.read_csv(hist_path), pd.DataFrame.from_records([{**trial.params, 'score': error}])])
    else:
        opt_hist_df = pd.DataFrame.from_records([{**trial.params, 'score': error}])
    opt_hist_df.to_csv(hist_path, index=False)
    return error



steps_ahead_list = [1, 5, 20]
n_trials = 100
for net_type in ["LSTM"]: # "CNN",
    caminho_de_saida = f"E:/mestrado/Pesquisa/Dados reais/Saída da otimização de hiperparâmetros {net_type}/{conjunto_de_dados}"
    if net_type == "LSTM":
        objective = objective_lstm
    else:
        objective = objective_cnn
    for steps_ahead in steps_ahead_list:
        os.makedirs(f'{caminho_de_saida}/{steps_ahead} steps ahead', exist_ok=True)
        if not os.path.exists(f'{caminho_de_saida}/{steps_ahead} steps ahead/opt_hist.csv'):
            study = optuna.create_study(
                direction='minimize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=30),
                study_name=f'hyperparameter_opt_{net_type}_{steps_ahead}_steps_ahead'
            )
            study.optimize(lambda trial: objective(trial, study, steps_ahead), n_trials=n_trials)
        else:
            df_temp = pd.read_csv(f'{caminho_de_saida}/{steps_ahead} steps ahead/opt_hist.csv')
            if df_temp.shape[0] == 0:
                study = optuna.create_study(
                    direction='minimize',
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=30),
                    study_name=f'hyperparameter_opt_{net_type}_{steps_ahead}_steps_ahead'
                )
                study.optimize(lambda trial: objective(trial, study, steps_ahead), n_trials=n_trials)
            elif df_temp.shape[0] < n_trials:
                study = optuna.create_study(
                    direction='minimize',
                    study_name=f'hyperparameter_opt_{net_type}_{steps_ahead}_steps_ahead'
                )
                for _, row in df_temp.iterrows():
                    if net_type == "CNN":
                        distributions = {
                            'win_size': optuna.distributions.IntDistribution(50, 800),
                            'filters_conv_1': optuna.distributions.IntDistribution(50, 7500),
                            'kernel_size_conv_1': optuna.distributions.CategoricalDistribution([(2, 2), (3, 2)]),
                            'activation_conv_1': optuna.distributions.CategoricalDistribution(
                                ['relu', 'elu', 'sigmoid', 'linear', 'tanh', 'swish']),
                            'pool_size_1': optuna.distributions.CategoricalDistribution([(2, 1), (3, 2)]),
                            'pool_type_1': optuna.distributions.CategoricalDistribution(['max', 'average']),
                            'filters_conv_2': optuna.distributions.IntDistribution(6,
                                                                                   row['filters_conv_1'] // 2),
                            'kernel_size_conv_2': optuna.distributions.CategoricalDistribution([(2, 2), (3, 2)]),
                            'activation_conv_2': optuna.distributions.CategoricalDistribution(['relu', 'elu', 'sigmoid', 'linear', 'tanh', 'swish']),
                            'dense_neurons': optuna.distributions.IntDistribution(5, row['w4'] * row['h4'] * row['filters_conv_2'] - 1),
                            'optimizer': optuna.distributions.CategoricalDistribution(
                                ['adam', 'adadelta', 'adagrad', 'rmsprop', 'sgd']),
                            'loss': optuna.distributions.CategoricalDistribution([
                                'mean_squared_error', 'mean_squared_logarithmic_error',
                                'mean_absolute_percentage_error',
                                'mean_absolute_error'
                            ])
                        }
                    else:
                        distributions = {
                            'win_size': optuna.distributions.IntDistribution(50, 800),
                            'Número de Camadas': optuna.distributions.IntDistribution(1, 5),
                            'Unidades na camada 0': optuna.distributions.IntDistribution(4, 64),
                            'Dropout na camada 0': optuna.distributions.FloatDistribution(0.1, 0.5),
                            'Dropout recorrente na camada 0': optuna.distributions.FloatDistribution(0.1, 0.5),
                            'optimizer': optuna.distributions.CategoricalDistribution(
                                ['adam', 'adadelta', 'adagrad', 'rmsprop', 'sgd']),
                            'loss': optuna.distributions.CategoricalDistribution([
                                'mean_squared_error', 'mean_squared_logarithmic_error',
                                'mean_absolute_percentage_error',
                                'mean_absolute_error'
                            ])
                        }
                        for i in range(1, 5):
                            if f'Unidades na  camada {i-1}' in row.to_dict().keys():
                                distributions[f'Unidades na camada {i}'] = optuna.distributions.IntDistribution(4, row[f'Unidades na camada {i - 1}']),
                            else:
                                distributions[f'Unidades na camada {i}'] = optuna.distributions.IntDistribution(4, 64),
                                distributions[f'Dropout na camada {i}'] = optuna.distributions.FloatDistribution(
                                    0.1, 0.5),
                                distributions[
                                    f'Dropout recorrente na camada {i}'] = optuna.distributions.FloatDistribution(
                                    0.1, 0.5)
                    study.add_trial(
                        optuna.trial.create_trial(
                            params={c: v for c, v in row.to_dict().items() if
                                    c not in ['w2', 'w3', 'w4', 'h4', 'score']},
                            distributions=distributions,
                            value=row['score']
                        )
                    )
                study.optimize(lambda trial: objective(trial, study, steps_ahead), n_trials=n_trials-df_temp.shape[0])
                del df_temp
            # else:
            #     del df_temp
# os.system('shutdown -s')