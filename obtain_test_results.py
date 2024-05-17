import ast
import os
import pickle
import re

import tensorflow as tf
import numpy as np
import pandas as pd
from utils import MMRE, images_and_targets_from_data_series, normalize_data, denormalize_data, from_ranges, to_ranges
import traceback

partition_size = 100
pasta_raiz =  f'C:/Users/User/Desktop/mestrado Felipe'
saida_df_redes_neurais = f'{pasta_raiz}/redes_neurais.csv'
for config in [1, 2, 4]:
    caminho_cnns = f"{pasta_raiz}/Saída da otimização de hiperparâmetros CNN conf{config}/boxplot"
    caminho_lstms = f"{pasta_raiz}/Saída da otimização de hiperparâmetros LSTM conf{config}/boxplot/"
    arima_path = f"{pasta_raiz}/ARIMA/config {config}/boxplot"
    var_path = f"{pasta_raiz}/VAR/config {config}/boxplot"
    arima_df = pd.read_csv(f"{arima_path}/particao de tamanho {partition_size}.csv")
    arima_df['mean_test_score']=[np.nan]*arima_df.shape[0]
    arima_df['test_score_whislo']=[np.nan]*arima_df.shape[0]
    arima_df['test_score_q1']=[np.nan]*arima_df.shape[0]
    arima_df['test_score_med']=[np.nan]*arima_df.shape[0]
    arima_df['test_score_q3']=[np.nan]*arima_df.shape[0]
    arima_df['test_score_whishi']=[np.nan]*arima_df.shape[0]
    visited_models = set()
    for base_model_file_name in os.listdir(f"{arima_path}/{partition_size}"):
        print(base_model_file_name)
        if base_model_file_name not in visited_models:
            _, data_index, steps_ahead, base_col = base_model_file_name.replace('.pkl', '').split('_')
            data_index = int(data_index)
            steps_ahead = int(steps_ahead)
            test_df = pd.read_csv(f'{pasta_raiz}/Dados/config 4/{data_index}/partition size {partition_size}/test.csv')
            test_data = test_df.to_dict('records')
            index = list(np.logical_and(arima_df['data_index'] == data_index, arima_df['steps_ahead'] == steps_ahead)).index(True)
            p = ast.literal_eval(arima_df['params'].iloc[index])[0]
            relative_errors_cols = []
            for col in test_df.columns:
                curr_model_fname = base_model_file_name.replace(base_col, col)
                visited_models.add(curr_model_fname)
                with open(f"{arima_path}/{partition_size}/{curr_model_fname}", 'rb') as model_file:
                    model = pickle.load(model_file)
                    relative_errors_test = []
                    for i in range(test_df.shape[0]):
                        if i > p and i + 1 + steps_ahead < len(test_data):
                            input_data, target = test_df[col].iloc[:i + 1], test_df[col].iloc[i + 1 + steps_ahead]
                            forecast = list(model.apply(input_data).forecast(steps_ahead))[-1]
                            relative_errors_test.append(np.abs((target - forecast) / (forecast + 0.0001)))
                    arima_df[f'test_score_{col}'] = np.mean(relative_errors_test)
                    relative_errors_cols.append(arima_df[f'test_score_{col}'])
            arima_df.loc[index, 'mean_test_score'] = np.mean(relative_errors_cols)
    arima_df.to_csv(f"{arima_path}/particao de tamanho {partition_size}.csv")
    def roda_var(model, data, lags, steps_ahead):
        relative_errors = []
        for i, row in data.iterrows():
            if i + lags + steps_ahead < data.shape[0]:
                # Fazer previsões para o próximo período
                entrada_vale = data.values[i:i + lags]
                target_vale = data.iloc[i + steps_ahead + lags]
                forecast = model.forecast(entrada_vale, steps=steps_ahead)[0]
                relative_errors.append(np.abs((target_vale - forecast) / (forecast + 0.0001)))
        # Calcular o MMRE
        return np.mean(relative_errors)


    var_df = pd.read_csv(f"{var_path}/particao de tamanho {partition_size}.csv")
    var_df['test_score']=[np.nan]*var_df.shape[0]
    for model_file_name in os.listdir(var_path):
        if model_file_name.endswith('.pkl'):
            print(model_file_name)
            _, data_index, steps_ahead = model_file_name.replace('.pkl', '').split('_')
            data_index = int(data_index)
            steps_ahead = int(steps_ahead)
            test_df = pd.read_csv(f'{pasta_raiz}/Dados/config 4/{data_index}/partition size {partition_size}/test.csv')
            test_data = test_df.to_dict('records')
            index = list(np.logical_and(var_df['data_index'] == data_index, var_df['steps_ahead'] == steps_ahead)).index(True)
            lags = var_df['lags'].iloc[index]
            with open(f"{var_path}/{model_file_name}", 'rb') as model_file:
                model = pickle.load(model_file)
                var_df.loc[index, 'test_score'] = roda_var(model, test_df, lags, steps_ahead)
    var_df.to_csv(f"{var_path}/particao de tamanho {partition_size}.csv")
    arquivo_conjuntos_com_erro = open(f'{pasta_raiz}/conjuntos com erro.log', 'w')
    redes_neurais_df = pd.DataFrame(columns=['data_index', 'steps_ahead', 'model_type', 'test_score'])
    for steps_ahead in [1, 5, 20]:
        for data_index in range(10):
            print(data_index, steps_ahead)
            train_df = pd.read_csv(f'{pasta_raiz}/Dados/config {config}/{data_index}/partition size {partition_size}/train.csv')
            test_df = pd.read_csv(f'{pasta_raiz}/Dados/config {config}/{data_index}/partition size {partition_size}/test.csv')
            test_data = test_df.to_dict('records')
            min_ = train_df['whislo'].min()
            max_ = train_df['whishi'].max()
            print(min_, max_)
            del train_df
            print("CNN")
            try:
                if config in [1, 2]:
                    pasta_cnn_opt = f"{pasta_raiz}/conf {config}/CNN/CNN {data_index}-{steps_ahead}"
                else:
                    pasta_cnn_opt = f"{caminho_cnns}/{data_index}/{steps_ahead} steps ahead"
                candidate_model_files = [p for p in os.listdir(pasta_cnn_opt) if re.search('best_model_trial_[0-9]+.h5', p)]
                if candidate_model_files:
                    model_file = max(candidate_model_files, key=lambda x: int(x.replace('best_model_trial_', '').replace('.h5', '')))
                else:
                    model_file = 'best_model.h5'
                cnn_model = tf.keras.models.load_model(f"{pasta_cnn_opt}/{model_file}", custom_objects={'MMRE': MMRE})
                cnn_model.summary()
                opt_hist_df = pd.read_csv(f"{pasta_cnn_opt}/opt_hist.csv")
                win_size = opt_hist_df[opt_hist_df['score'] == opt_hist_df['score'].min()]['win_size'].iloc[0]
                X_test, Y_test = images_and_targets_from_data_series(
                    test_data, input_win_size=win_size, steps_ahead=steps_ahead
                )
                X_test, Y_test = normalize_data(X_test, Y_test, min_, max_)
                X_test, Y_test = to_ranges(X_test), to_ranges(Y_test)
                T = from_ranges(cnn_model.predict(X_test, verbose=0), axis=1)
                error = 0
                if steps_ahead == 1:
                    error = tf.keras.backend.eval(MMRE(Y_test[:, 0, :], T))
                else:
                    for s in range(2, steps_ahead + 1):
                        T = cnn_model.predict(X_test, verbose=0)
                        T = from_ranges(T, axis=1)
                        T = T.reshape(T.shape[0], 1, -1, 1)
                        T = denormalize_data(T, min_, max_)
                        X_ = np.concatenate([X_test[:, 1:, :, :], T], axis=1)
                        if s == steps_ahead:
                            predicted = cnn_model.predict(X_)
                            predicted = from_ranges(predicted, axis=1)
                            error = tf.keras.backend.eval(MMRE(Y_test[:, s - 1, :], denormalize_data(predicted, min_, max_)))
                redes_neurais_df=pd.concat([redes_neurais_df, pd.DataFrame.from_records([{
                    'data_index': data_index,  'steps_ahead': steps_ahead, 'model_type': 'CNN', 'test_score': error
                }])])
            except Exception as e:
                print(f"config {config} CNN {data_index}, {steps_ahead} steps ahead - error\n")
                arquivo_conjuntos_com_erro.write(f"config {config} CNN {data_index}, {steps_ahead} steps ahead\n")
                arquivo_conjuntos_com_erro.write(str(e))
                arquivo_conjuntos_com_erro.write('\n')
                arquivo_conjuntos_com_erro.write(traceback.format_exc())
                arquivo_conjuntos_com_erro.write('\n' + '#'*50 + '\n')
            print("LSTM")
            try:
                if config in [1, 2]:
                    pasta_lstm_opt = f"{pasta_raiz}/conf {config}/LSTM/LSTM {data_index}-{steps_ahead}"
                else:
                    pasta_lstm_opt = f"{caminho_lstms}/{data_index}/{steps_ahead} steps ahead"
                candidate_model_files = [p for p in os.listdir(pasta_lstm_opt) if re.search('best_model_trial_[0-9]+.h5', p)]
                if candidate_model_files:
                    model_file = max(candidate_model_files, key=lambda x: int(x.replace('best_model_trial_', '').replace('.h5', '')))
                else:
                    model_file = 'best_model.h5'
                lstm_model = tf.keras.models.load_model(f"{pasta_lstm_opt}/{model_file}", custom_objects={'MMRE': MMRE})
                opt_hist_df = pd.read_csv(f"{pasta_lstm_opt}/opt_hist.csv")
                win_size = opt_hist_df[opt_hist_df['score'] == opt_hist_df['score'].min()]['win_size'].iloc[0]
                data = np.array([np.array(list(x.values())) for x in test_data])
                X_test = np.array([data[i:i + win_size] for i, j in
                                   zip(range(0, win_size * ((len(data) - steps_ahead) // win_size), win_size),
                                       range((len(data) - win_size - steps_ahead) // win_size))])
                Y_test = np.array([data[i + win_size + steps_ahead] for i, j in
                                   zip(range(0, win_size * ((len(data) - steps_ahead) // win_size), win_size),
                                       range((len(data) - win_size - steps_ahead) // win_size))])
                X_test, Y_test = normalize_data(X_test, Y_test, min_, max_)
                X_test, Y_test = to_ranges(X_test, axis=1), to_ranges(Y_test, axis=1)
                T = from_ranges(lstm_model.predict(X_test, verbose=0), axis=1)
                error = tf.keras.backend.eval(
                    MMRE(denormalize_data(from_ranges(Y_test, axis=1), min_, max_), denormalize_data(T, min_, max_))
                )
                redes_neurais_df = pd.concat([redes_neurais_df, pd.DataFrame.from_records([{
                    'data_index': data_index, 'steps_ahead': steps_ahead, 'model_type': 'LSTM', 'test_score': error
                }])])
            except Exception as e:
                print(f"config {config} LSTM {data_index}, {steps_ahead} steps ahead - error\n")
                arquivo_conjuntos_com_erro.write(f"config {config} LSTM {data_index}, {steps_ahead} steps ahead\n")
                arquivo_conjuntos_com_erro.write(str(e))
                arquivo_conjuntos_com_erro.write('\n')
                arquivo_conjuntos_com_erro.write(traceback.format_exc())
                arquivo_conjuntos_com_erro.write('\n' + '#' * 50 +'\n')
    redes_neurais_df.to_csv(f"{pasta_raiz}/redes neurais - config {config}.csv", index=False)