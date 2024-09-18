import ast
import os
import pickle
import matplotlib.pyplot as plt
import optuna
import numpy as np
import pandas as pd
from utils import create_model, create_lstm_model, MMRE, MMRE_Loss, images_and_targets_from_data_series
from utils import plot_multiple_box_plot_series, plot_single_box_plot_series, normalize_data, to_ranges, from_ranges
from utils import denormalize_data, cria_ou_atualiza_arquivo_no_drive, retorna_arquivo_se_existe
from tqdm import tqdm
from typing import Literal
import tensorflow as tf
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

tipo_de_dados: Literal["Simulados", "Reais"] = "Reais"

if tipo_de_dados == "Simulados":
    id_pasta_base_drive = "1cBW25sKEV-1CKZ0Rwazf3qodb0m9GBt1"
    caminho_fonte_dados_local = "D:/mestrado/Pesquisa/Dados simulados"  # "C:/Users/User/Desktop/mestrado Felipe" #
else:
    id_pasta_base_drive = "1BYnWbci5nuYYG6iDIMFDOh3ctz7yX3H4"
    caminho_fonte_dados_local = "/home/CIN/lfbjc/mestrado_dados/Dados reais" # "C:/Users/User/Desktop/mestrado Felipe/Dados reais"  #


def objective_cnn(trial, study, train_data, val_data, pasta_base_saida, caminho_interno):
    caminho_completo_saida = os.path.join(pasta_base_saida, caminho_interno)
    available_kernel_sizes = [(2, 2), (3, 2)]
    if isinstance(train_data[0], dict):
        out_size = len(train_data[0].keys())
    else:
        out_size = len(train_data[0])
    print(f'OUT SIZE: {out_size}')
    win_size = trial.suggest_int('win_size', 10, min(len(train_data) // 10, 1000))
    filters_conv_1 = trial.suggest_int('filters_conv_1', 2, 5)
    kernel_size_conv_1 = trial.suggest_categorical('kernel_size_conv_1', available_kernel_sizes)
    activation_conv_1 = trial.suggest_categorical('activation_conv_1', ['relu', 'elu', 'sigmoid', 'linear', 'tanh', 'swish'])
    pool_size_1 = trial.suggest_categorical('pool_size_1', [(2, 1), (3, 1)])
    pool_type_1 = trial.suggest_categorical('pool_type_1', ['max', 'average'])
    # filters_conv_2 = trial.suggest_int('filters_conv_2', 6, filters_conv_1 // 2)
    w2 = (win_size - kernel_size_conv_1[0]) - pool_size_1[0] + 2
    h2 = (5 - kernel_size_conv_1[0]) - pool_size_1[1] + 2
    # w3 = (w2 - pool_size_1[0])//pool_size_1[0] + 1
    # kernel_size_conv_2 = trial.suggest_categorical('kernel_size_conv_2', available_kernel_sizes)
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
    if isinstance(train_data[0], dict):
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
        batch_size=1,
        epochs=N_EPOCHS,
        verbose=0,
        validation_split=1/3,
        callbacks=tf.keras.callbacks.EarlyStopping(patience=10)
    )
    X_test, Y_test = images_and_targets_from_data_series(
        val_data, input_win_size=win_size, steps_ahead=steps_ahead
    )
    X_test, Y_test = normalize_data(X_test, Y_test, min_, max_)
    if out_size > 1:
        X_test, Y_test = to_ranges(X_test), to_ranges(Y_test)
        T = from_ranges(model.predict(X_test, verbose=0), axis=1)
    else:
        T = model.predict(X_test, verbose=0)
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
        error = tf.keras.backend.eval(MMRE(Y_test[:, 0, :], T))
    else:
        for s in range(2, steps_ahead+1):
            T = model.predict(X_test, verbose=0)
            if out_size > 1:
                T = from_ranges(T, axis=1)
            T = T.reshape(T.shape[0], 1, -1, 1)
            T = denormalize_data(T, min_, max_)
            X_ = np.concatenate([X_test[:, 1:, :, :], T], axis=1)
            if s == steps_ahead:
                predicted = model.predict(X_)
                if out_size > 1:
                    predicted = from_ranges(predicted, axis=1)
                error = tf.keras.backend.eval(MMRE(Y_test[:, s-1, :], denormalize_data(predicted, min_, max_)))
    gauth = GoogleAuth()
    scope = ['https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('conta-de-servico.json', scope)
    gauth.credentials = creds

    # Criação do objeto drive
    drive = GoogleDrive(gauth)
    if np.isnan(error):
        print(debug)
    try:
        if error < study.best_value:
            caminho_modelo_local = f'{caminho_completo_saida}/best_model.h5'
            caminho_modelo_drive = f'{caminho_interno}/best_model.h5'
            model.save(caminho_modelo_local)
            cria_ou_atualiza_arquivo_no_drive(drive, id_pasta_base_drive, caminho_modelo_drive, caminho_modelo_local)
            caminho_pickle_history = f'{caminho_completo_saida}/best_model_history.pkl'
            caminho_history_drive = f'{caminho_interno}/best_model_history.pkl'
            pickle.dump(history.history, open(f'{caminho_completo_saida}/best_model_history.pkl', 'wb'))
            cria_ou_atualiza_arquivo_no_drive(drive, id_pasta_base_drive, caminho_history_drive, caminho_pickle_history)
    except ValueError:
        # if no trials are completed yet save the first trial
        caminho_modelo_local = f'{caminho_completo_saida}/best_model.h5'
        caminho_modelo_drive = f'{caminho_interno}/best_model.h5'
        model.save(caminho_modelo_local)
        cria_ou_atualiza_arquivo_no_drive(drive, id_pasta_base_drive, caminho_modelo_drive, caminho_modelo_local)
        caminho_pickle_history = f'{caminho_completo_saida}/best_model_history.pkl'
        caminho_history_drive = f'{caminho_interno}/best_model_history.pkl'
        pickle.dump(history.history, open(f'{caminho_completo_saida}/best_model_history.pkl', 'wb'))
        cria_ou_atualiza_arquivo_no_drive(drive, id_pasta_base_drive, caminho_history_drive, caminho_pickle_history)
    opt_hist_df = pd.DataFrame.from_records([trial.params])
    # opt_hist_df['w2'] = w2
    # opt_hist_df['w3'] = w3
    # opt_hist_df['w4'] = w4
    # opt_hist_df['h4'] = h4
    opt_hist_df['score'] = error
    hist_path = f'{caminho_completo_saida}/opt_hist.csv'
    append_condition = os.path.exists(hist_path)
    print('append_condition:', append_condition)
    opt_hist_df.to_csv(hist_path, mode='a' if append_condition else 'w', index=False, header=(not append_condition))
    hist_path_drive = f'{caminho_interno}/opt_hist.csv'
    cria_ou_atualiza_arquivo_no_drive(drive, id_pasta_base_drive, hist_path_drive, hist_path)
    return error


def objective_lstm(trial, study, train_data, val_data,  pasta_base_saida, caminho_interno):
    caminho_completo_saida = os.path.join(pasta_base_saida, caminho_interno)
    win_size = trial.suggest_int('win_size', 10, min(len(train_data) // 10, 13000))
    if isinstance(train_data[0], dict):
        data = np.array([np.array(list(x.values())) for x in train_data])
    else:
        data = np.array(train_data)
    min_, max_ = np.min(data), np.max(data)
    X = np.array([data[i:i+win_size] for i, j in zip(range(0, win_size*((len(data) - steps_ahead)//win_size), win_size), range((len(data) - win_size - steps_ahead)//win_size))])
    Y = np.array([data[i+win_size+steps_ahead] for i, j in zip(range(0, win_size*((len(data) - steps_ahead)//win_size), win_size), range((len(data) - win_size - steps_ahead)//win_size))])
    X, Y = normalize_data(X, Y, min_, max_)
    if isinstance(train_data[0], dict):
        X, Y = to_ranges(X, axis=1), to_ranges(Y, axis=1)
    num_layers = trial.suggest_int("Número de camadas", 1, 2)
    units=[]
    activations = []
    recurrent_activations = []
    dropouts = []
    recurrent_dropouts = []
    for camada in range(num_layers):
        if camada == 0:
            units.append(trial.suggest_int(f"Unidades na camada {camada}", 4, 32))
        else:
            units.append(trial.suggest_int(f"Unidades na camada {camada}", 3, units[-1]))
        activations.append("sigmoid")
        recurrent_activations.append("sigmoid")
        dropouts.append(trial.suggest_float(f"Dropout da camada {camada}", 0.1, 0.5))
        recurrent_dropouts.append(trial.suggest_float(f"Dropout recorrente da camada {camada}", 0.1, 0.5))
    model = create_lstm_model(X.shape[1:], units, activations, recurrent_activations, dropouts, recurrent_dropouts)
    N_EPOCHS = 300
    model.compile(
        optimizer=trial.suggest_categorical('optimizer', ['sgd']),
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
        validation_split=1/3,
        callbacks=tf.keras.callbacks.EarlyStopping(patience=10)
    )
    if isinstance(train_data[0], dict):
        data = np.array([np.array(list(x.values())) for x in val_data])
    else:
        data = np.array(val_data)
    X_test = np.array([data[i:i + win_size] for i, j in zip(range(0, win_size*((len(data) - steps_ahead)//win_size), win_size), range((len(data) - win_size - steps_ahead)//win_size))])
    Y_test = np.array([data[i + win_size + steps_ahead] for i, j in zip(range(0, win_size*((len(data) - steps_ahead)//win_size), win_size), range((len(data) - win_size - steps_ahead)//win_size))])
    X_test, Y_test = normalize_data(X_test, Y_test, min_, max_)
    if isinstance(train_data[0], dict):
        X_test, Y_test = to_ranges(X_test, axis=1), to_ranges(Y_test, axis=1)
        T = from_ranges(model.predict(X_test, verbose=0), axis=1)
        error = tf.keras.backend.eval(
            MMRE(denormalize_data(from_ranges(Y_test, axis=1), min_, max_), denormalize_data(T, min_, max_))
        )
    else:
        T = model.predict(X_test, verbose=0)
        error = tf.keras.backend.eval(
            MMRE(denormalize_data(Y_test, min_, max_), denormalize_data(T, min_, max_))
        )
    gauth = GoogleAuth()
    scope = ['https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('conta-de-servico.json', scope)
    gauth.credentials = creds

    # Criação do objeto drive
    drive = GoogleDrive(gauth)
    try:
        if error < study.best_value:
            caminho_modelo_local = f'{caminho_completo_saida}/best_model.h5'
            caminho_modelo_drive = f'{caminho_interno}/best_model.h5'
            model.save(caminho_modelo_local)
            cria_ou_atualiza_arquivo_no_drive(drive, id_pasta_base_drive, caminho_modelo_drive, caminho_modelo_local)
            caminho_pickle_history = f'{caminho_completo_saida}/best_model_history.pkl'
            caminho_history_drive = f'{caminho_interno}/best_model_history.pkl'
            pickle.dump(history.history, open(f'{caminho_completo_saida}/best_model_history.pkl', 'wb'))
            cria_ou_atualiza_arquivo_no_drive(drive, id_pasta_base_drive, caminho_history_drive, caminho_pickle_history)
    except ValueError:
        # if no trials are completed yet save the first trial
        caminho_modelo_local = f'{caminho_completo_saida}/best_model.h5'
        caminho_modelo_drive = f'{caminho_interno}/best_model.h5'
        model.save(caminho_modelo_local)
        cria_ou_atualiza_arquivo_no_drive(drive, id_pasta_base_drive, caminho_modelo_drive, caminho_modelo_local)
        caminho_pickle_history = f'{caminho_completo_saida}/best_model_history.pkl'
        caminho_history_drive = f'{caminho_interno}/best_model_history.pkl'
        pickle.dump(history.history, open(f'{caminho_completo_saida}/best_model_history.pkl', 'wb'))
        cria_ou_atualiza_arquivo_no_drive(drive, id_pasta_base_drive, caminho_history_drive, caminho_pickle_history)
    hist_path = f'{caminho_completo_saida}/opt_hist.csv'
    if os.path.exists(hist_path):
        opt_hist_df = pd.concat([pd.read_csv(hist_path), pd.DataFrame.from_records([{**trial.params, 'score': error}])])
    else:
        opt_hist_df = pd.DataFrame.from_records([{**trial.params, 'score': error}])
    opt_hist_df.to_csv(hist_path, index=False)
    hist_path_drive = f'{caminho_interno}/opt_hist.csv'
    # cria_ou_atualiza_arquivo_no_drive(drive, id_pasta_base_drive, hist_path_drive, hist_path)
    return error


aggregation_type = 'boxplot' # 'median' #
cols_alvo = {
    # "demanda energética - kaggle": "TOTALDEMAND",
    # "cafe": "money",
    # "beijing": "pm2.5",
    # "KAGGLE - HOUSE HOLD ENERGY CONSUMPTION": "USAGE",
    "WIND POWER GERMANY": "MW",
}
steps_ahead_list = [1, 5, 20]
n_trials = 100
objective_by_model_type = {
    'LSTM': objective_lstm,
    'CNN': objective_cnn
}
model_type = "CNN"
for partition_size in [100]:  # [100, None]:
    if tipo_de_dados == "Simulados":
        pastas_entrada = []
        for config in range(1, 8):
            data_indices = list(range(10))
            for data_index in data_indices:
                pastas_entrada.append(f"config {config}/partição de tamanho {partition_size}/{data_index}")
    else:
        pastas_entrada = list(cols_alvo.keys())
    for pasta_entrada in pastas_entrada:
        if pasta_entrada == "demanda energética - kaggle":
            local_steps_ahead = [5, 1]
        else:
            local_steps_ahead = steps_ahead_list
        if tipo_de_dados == "Simulados":
            val_file_name = ''
            if partition_size is not None:
                caminho_dados_drive = f'Dados/config {config}/{data_set_index}/partition size {partition_size}'
                train_file_name = 'train.csv'
            else:
                aggregation_type = 'Dados brutos'
                caminho_dados_drive = f'Dados/config {config}/{data_set_index}'
                train_file_name = 'raw_train_series.csv'
            saida_complemento = f"Saída da otimização de hiperparâmetros {model_type} conf{config}/{aggregation_type}/{data_set_index}"
        else:
            if partition_size is not None:
                caminho_dados_drive = f"Dados tratados/{pasta_entrada}/agrupado em boxplots"
                saida_complemento = f"{model_type}/Saída da otimização de hiperparâmetros/com boxplot/{pasta_entrada}"
            else:
                caminho_dados_drive = f"Dados tratados/{pasta_entrada}/sem agrupamento"
                saida_complemento = f"{model_type}/Saída da otimização de hiperparâmetros/sem agrupamento/{pasta_entrada}"
            train_file_name = 'train.csv'
            val_file_name = 'val.csv'

        caminho_de_saida = f"{caminho_fonte_dados_local}/{saida_complemento}"
        objective = objective_by_model_type[model_type]
        for steps_ahead in local_steps_ahead:
            # plot_single_box_plot_series(train_data)
            saida_drive = f"{saida_complemento}/{steps_ahead} steps ahead/"
            caminho_completo_saida = f'{caminho_de_saida}/{steps_ahead} steps ahead/'
            os.makedirs(caminho_completo_saida, exist_ok=True)
            objective_kwargs = {
                'pasta_base_saida': caminho_fonte_dados_local,
                'caminho_interno': f"{saida_complemento}/{steps_ahead} steps ahead/"
            }
            gauth = GoogleAuth()
            scope = ['https://www.googleapis.com/auth/drive']
            creds = ServiceAccountCredentials.from_json_keyfile_name('conta-de-servico.json', scope)
            gauth.credentials = creds

            # Criação do objeto drive
            drive = GoogleDrive(gauth)
            proceed = False
            if not val_file_name:
                train_and_val_file = retorna_arquivo_se_existe(drive, id_pasta_base_drive, f'{caminho_dados_drive}/{train_file_name}')
                if train_and_val_file is not None:
                    proceed = True
                    pasta_dados = f"{caminho_fonte_dados_local}/{caminho_dados_drive}"
                    os.makedirs(pasta_dados, exist_ok=True)
                    train_and_val_file.GetContentFile(f"{pasta_dados}/{train_file_name}")
                    if aggregation_type == 'boxplot':
                        train_and_val = pd.read_csv(f"{pasta_dados}/{train_file_name}").to_dict('records')
                    elif aggregation_type == 'med':
                        train_and_val = pd.read_csv(f'{pasta_dados}/{train_file_name}')['med'].map(lambda x: [x]).values
                    else:
                        train_and_val = pd.read_csv(f'{pasta_dados}/{train_file_name}')['s'].map(lambda x: [x]).values
                    objective_kwargs['train_data'] = train_and_val[:int(2 / 3 * len(train_and_val))]
                    objective_kwargs['val_data'] = train_and_val[int(2 / 3 * len(train_and_val)):]
            else:
                pasta_dados = f"{caminho_fonte_dados_local}/{caminho_dados_drive}"
                if not os.path.exists(f"{pasta_dados}/{train_file_name}") or not os.path.exists(f"{pasta_dados}/{val_file_name}"):
                    train_file = retorna_arquivo_se_existe(drive, id_pasta_base_drive, f'{caminho_dados_drive}/{train_file_name}')
                    val_file = retorna_arquivo_se_existe(drive, id_pasta_base_drive, f'{caminho_dados_drive}/{val_file_name}')
                    if train_file is not None and val_file is not None:
                        proceed = True
                        pasta_dados = f"{caminho_fonte_dados_local}/{caminho_dados_drive}"
                        os.makedirs(pasta_dados, exist_ok=True)
                        train_file.GetContentFile(f"{pasta_dados}/{train_file_name}")
                        val_file.GetContentFile(f"{pasta_dados}/{val_file_name}")
                else:
                    proceed = True
                if proceed:
                    train_data = pd.read_csv(f"{pasta_dados}/{train_file_name}")
                    val_data = pd.read_csv(f"{pasta_dados}/{val_file_name}")
                    if partition_size is None:
                        train_data = train_data[[cols_alvo.get(pasta_entrada)]]
                        val_data = val_data[[cols_alvo.get(pasta_entrada)]]
                    else:
                        train_data = train_data[["whislo", "q1", "med", "q3", "whishi"]]
                        val_data = val_data[["whislo", "q1", "med", "q3", "whishi"]]
                    objective_kwargs['train_data'] = train_data.values
                    objective_kwargs['val_data'] = val_data.values
            if proceed:
                study = optuna.create_study(
                    direction='minimize',
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=30),
                    study_name=f'hyperparameter_opt {pasta_entrada}, {model_type}, {steps_ahead} steps ahead'
                )
                objective_kwargs['study'] = study
                opt_hist_file_drive = retorna_arquivo_se_existe(drive, id_pasta_base_drive, f'{saida_drive}/opt_hist.csv')
                if opt_hist_file_drive is None:
                    study.optimize(lambda trial: objective(trial=trial, **objective_kwargs), n_trials=n_trials)
                else:
                    opt_hist_file_drive.GetContentFile(f'{caminho_completo_saida}/opt_hist.csv')
                    opt_hist_df = pd.read_csv(f'{caminho_completo_saida}/opt_hist.csv')
                    for _, row in opt_hist_df.iterrows():
                        train_data_size = int(2 / 3 * len(train_and_val))
                        if model_type == "CNN":
                            if aggregation_type == "boxplot":
                                available_kernel_sizes = [(2, 2), (3, 2)]
                            else:
                                available_kernel_sizes = [(2, 1), (3, 1)]
                            win_size = int(row['win_size'])
                            kernel_size_conv_1 = ast.literal_eval(row['kernel_size_conv_1'])
                            pool_size_1 = ast.literal_eval(row['pool_size_1'])
                            filters_conv_1 = int(row['filters_conv_1'])
                            w2 = (win_size - kernel_size_conv_1[0]) - pool_size_1[0] + 2
                            h2 = (5 - kernel_size_conv_1[0]) - pool_size_1[1] + 2
                            distributions = {
                                'win_size': optuna.distributions.IntDistribution(10, int(train_data_size/10)),
                                'filters_conv_1': optuna.distributions.IntDistribution(2, 10),
                                'kernel_size_conv_1': optuna.distributions.CategoricalDistribution(available_kernel_sizes),
                                'activation_conv_1': optuna.distributions.CategoricalDistribution(['relu', 'elu', 'sigmoid', 'linear', 'tanh', 'swish']),
                                'pool_size_1': optuna.distributions.CategoricalDistribution([(2, 1), (3, 1)]),
                                'pool_type_1': optuna.distributions.CategoricalDistribution(['max', 'average']),
                                'dense_neurons': optuna.distributions.IntDistribution(5, w2 * h2 * filters_conv_1 - 1),
                                'optimizer': optuna.distributions.CategoricalDistribution(['adam', 'adadelta', 'adagrad', 'rmsprop', 'sgd']),
                                'loss': optuna.distributions.CategoricalDistribution([
                                    'mean_squared_error', 'mean_squared_logarithmic_error',
                                    'mean_absolute_percentage_error',
                                    'mean_absolute_error'
                                ])
                            }
                        else:
                            distributions = {
                                'win_size': optuna.distributions.IntDistribution(10, int(train_data_size/10)),
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
                                if f'Unidades na  camada {i - 1}' in row.to_dict().keys():
                                    distributions[f'Unidades na camada {i}'] = optuna.distributions.IntDistribution(
                                        4, row[f'Unidades na camada {i - 1}']),
                                else:
                                    distributions[f'Unidades na camada {i}'] = optuna.distributions.IntDistribution(
                                        4, 64),
                                    distributions[
                                        f'Dropout na camada {i}'] = optuna.distributions.FloatDistribution(
                                        0.1, 0.5),
                                    distributions[
                                        f'Dropout recorrente na camada {i}'] = optuna.distributions.FloatDistribution(
                                        0.1, 0.5)
                        def value_from_row_value(c, v):
                            print(f'{c}: {v}')
                            if isinstance(v, str):
                                try:
                                    return ast.literal_eval(v)
                                except ValueError:
                                    return v
                            else:
                                return v
                        study.add_trial(
                            optuna.trial.create_trial(
                                params={c: value_from_row_value(c, v) for c, v in row.to_dict().items() if c not in ['w2', 'w3', 'w4', 'h4', 'score']},
                                distributions=distributions,
                                value=row['score']
                            )
                        )
                        study.optimize(lambda trial: objective(trial=trial, **objective_kwargs),
                                       n_trials=n_trials - opt_hist_df.shape[0])
# with open(f"{caminho_fonte_dados_local}/TERMINOU.txt", "w") as termino_arquivo:
#     termino_arquivo.write("TERMINOU!")
# os.system('shutdown /s')
# c7 8s1
# c6 8s1