import os
import pickle

import numpy as np
import pandas as pd

partition_size = 100
pasta_raiz =  f'E:/mestrado/Pesquisa/Dados simulados'
saida_df_redes_neurais = f'{pasta_raiz}/redes_neurais.csv'
arima_path = "{pasta_raiz}/ARIMA/config 4/boxplot"
var_path = "{pasta_raiz}/VAR/config 4/boxplot"
test_df = pd.read_csv(f'{pasta_raiz}/Dados/config {config}/{data_set_index}/partition size {partition_size}/test.csv')
caminho_cnns = f"{pasta_raiz}/Saída da otimização de hiperparâmetros CNN conf4/boxplot/"
caminho_lstms = f"{pasta_raiz}/Saída da otimização de hiperparâmetros LSTM conf4/boxplot/"

arima_df = pd.read_csv(f"{arima_path}/particao de tamanho {partition_size}.csv")
arima_df['mean_test_score']=[np.nan]*arima_df.shape[0]
arima_df['test_score_whislo']=[np.nan]*arima_df.shape[0]
arima_df['test_score_q1']=[np.nan]*arima_df.shape[0]
arima_df['test_score_med']=[np.nan]*arima_df.shape[0]
arima_df['test_score_q3']=[np.nan]*arima_df.shape[0]
arima_df['test_score_whishi']=[np.nan]*arima_df.shape[0]
visited_models = set()
for model_file_name in os.listdir(f"{arima_path}/{partition_size}"):
    if base_model_file_name not in visited_models:
        _, _, _, data_index, _, steps_ahead, base_col = base_model_file_name.replace('.pkl', '').split('_')
        index = np.logical_and(arima_df['data_index'] == data_index, arima_df['steps_ahead'] == steps_ahead).index(True)
        relative_errors_cols = []
        for col in test_df.columns:
            curr_model_fname = base_model_file_name.replace(base_col, col)
            visited_models.add(curr_model_fname)
            with open(f"{arima_path}/{partition_size}/{curr_model_fname}", 'rb') as model_file:
                model = pickle.load(model_file)
                relative_errors_test = []
                for i, row in test_df.iterrows():
                    if i > p and i + 1 + steps_ahead < test_data.shape[0]:
                        input_data, target = row[col].iloc[:i + 1], row[col].iloc[i + 1 + steps_ahead]
                        forecast = list(model.apply(input_data).forecast(steps_ahead))[-1]
                        relative_errors_test.append(np.abs((target - forecast) / (forecast + 0.0001)))
                arima_df[f'test_score_{col}'] = np.mean(relative_errors_test)
                relative_errors_cols.append(arima_df[f'test_score_{col}'])
        arima_df.loc[index, 'mean_test_score'] = np.mean(relative_errors_cols)

var_df = pd.read_csv(f"{var_path}/particao de tamanho {partition_size}.csv")
var_df['test_score']=[np.nan]*var_df.shape[0]
for model_file_name in os.listdir(f"{var_path}/{partition_size}"):
    _, _, _, data_index, _, steps_ahead = model_file_name.replace('.pkl', '').split('_')
    index = np.logical_and(arima_df['data_index'] == data_index, arima_df['steps_ahead'] == steps_ahead).index(True)
    with open(f"{var_path}/{partition_size}/{model_file_name}", 'rb') as model_file:
        model = pickle.load(model_file)
        relative_errors_test = []
        for i, row in test_df.iterrows():
            if i > p and i + 1 + steps_ahead < test_data.shape[0]:
                input_data, target = test_df.iloc[:i + 1], test_df.iloc[i + 1 + steps_ahead]
                forecast = list(model.apply(input_data).forecast(steps_ahead))[-1]
                relative_errors_test.append(np.abs((target - forecast) / (forecast + 0.0001)))
        var_df.loc[index, 'test_score'] = np.mean(relative_errors_test)

train_df = pd.read_csv(f'{pasta_raiz}/Dados/config {config}/{data_set_index}/partition size {partition_size}/train.csv')
min_ = train_df.min(axis=None)
max_ = train_df.max(axis=None)
del train_df
test_data = test_df.to_dict('records')
redes_neurais_df = pd.DataFrame(columns=['data_index', 'steps_ahead', 'model_type', 'test_score'])
for steps_ahead in [1, 5, 20]:
    for data_index in range(10):
        with tf.keras.models.load_model(f"{caminho_cnns}/{steps_ahead} steps ahead/{data_index}/best_model.h5") as cnn_model:
            X_test, Y_test = images_and_targets_from_data_series(
                test_data, input_win_size=win_size, steps_ahead=steps_ahead
            )
            X_test, Y_test = normalize_data(X_test, Y_test, min_, max_)
            X_test, Y_test = to_ranges(X_test), to_ranges(Y_test)
            T = from_ranges(model.predict(X_test, verbose=0), axis=1)
            error = 0
            if steps_ahead == 1:
                error = tf.keras.backend.eval(MMRE(Y_test[:, 0, :], T))
            else:
                for s in range(2, steps_ahead + 1):
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
                        error = tf.keras.backend.eval(MMRE(Y_test[:, s - 1, :], denormalize_data(predicted, min_, max_)))
            redes_neurais_df=pd.concat([redes_neurais_df, pd.DataFrame.from_records([{
                'data_index': data_index,  'steps_ahead': steps_ahead, 'model_type': 'CNN', 'test_score': error
            }])])
        with tf.keras.models.load_model(f"{caminho_lstms}/{steps_ahead} steps ahead/{data_index}/best_model.h5") as lstm_model:
            data = np.array([np.array(list(x.values())) for x in test_data])
            X_test = np.array([data[i:i + win_size] for i, j in
                               zip(range(0, win_size * ((len(data) - steps_ahead) // win_size), win_size),
                                   range((len(data) - win_size - steps_ahead) // win_size))])
            Y_test = np.array([data[i + win_size + steps_ahead] for i, j in
                               zip(range(0, win_size * ((len(data) - steps_ahead) // win_size), win_size),
                                   range((len(data) - win_size - steps_ahead) // win_size))])
            X_test, Y_test = normalize_data(X_test, Y_test, min_, max_)
            X_test, Y_test = to_ranges(X_test, axis=1), to_ranges(Y_test, axis=1)
            T = from_ranges(model.predict(X_test, verbose=0), axis=1)
            error = tf.keras.backend.eval(
                MMRE(denormalize_data(from_ranges(Y_test, axis=1), min_, max_), denormalize_data(T, min_, max_))
            )
            redes_neurais_df = pd.concat([redes_neurais_df, pd.DataFrame.from_records([{
                'data_index': data_index, 'steps_ahead': steps_ahead, 'model_type': 'LSTM', 'test_score': error
            }])])