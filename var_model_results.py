import os

import optuna
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR, ARIMA
from utils import plot_single_box_plot_series, plot_multiple_box_plot_series

def roda_var(model, val_data, lags, steps_ahead):
    relative_errors = []
    for i, row in val_data.iterrows():
        if i + lags + steps_ahead < val_data.shape[0]:
            # Fazer previsões para o próximo período
            entrada_vale = val_data.values[i:i + lags]
            target_vale = val_data.iloc[i + steps_ahead + lags]
            forecast = model.forecast(entrada_vale, steps=steps_ahead)[0]
            relative_errors.append(np.abs((target_vale - forecast) / (forecast + 0.0001)))
    # Calcular o MMRE
    return np.mean(relative_errors)


def salva(caminho_de_saida, data_index, steps_ahead, best_error, best_params, params_column_name='params'):
    if os.path.exists(caminho_de_saida):
        results_df = pd.read_csv(caminho_de_saida)
    else:
        results_df = pd.DataFrame(columns=['data_index', 'steps_ahead', params_column_name, 'score'])
    cond = np.logical_and(results_df['data_index'] == data_index, results_df['steps_ahead'] == steps_ahead)
    if results_df[cond].shape[0] == 0:
        results_df = pd.concat([
            results_df,
            pd.DataFrame.from_records([{
                'data_index': data_index, 'steps_ahead': steps_ahead,
                params_column_name: best_params, 'score': best_error
            }])
        ])
    else:
        results_df.loc[cond, 'score'] = [best_error]
        results_df.loc[cond, params_column_name] = [str(best_params)]
    # print(caminho_de_saida)
    results_df.to_csv(caminho_de_saida, index=False)

if __name__ == '__main__':
    model_name = "ARIMA" # "VAR" #
    aggregation_type = "boxplot" # only relevant for ARIMA
    config = 4
    # data_index = 2
    partition_size = 360 # 360, 250, 100 # o 250 (mediana) deu falha na 4s1p250 LU decomposition error
    steps_ahead_list = [1, 5, 20]
    for data_index in range(1, 10):
        if data_index == 1:
            local_steps_ahead_list=[5,20]
        else:
            local_steps_ahead_list = steps_ahead_list
        caminho_de_saida = f"E:/mestrado/Pesquisa/Dados simulados/{model_name}/config {config}/{aggregation_type}/particao de tamanho {partition_size}.csv"
        os.makedirs(os.path.dirname(caminho_de_saida), exist_ok=True)
        caminho_dados = f'E:/mestrado/Pesquisa/Dados simulados/Dados/config {config}/{data_index}/partition size {partition_size}/'
        train_path = f'{caminho_dados}/train.csv'
        train_df = pd.read_csv(train_path)
        train_df, val_df = train_df.iloc[:int(2/3*train_df.shape[0])].reset_index(drop=True), train_df.iloc[int(2/3*train_df.shape[0]):].reset_index(drop=True)
        for steps_ahead in local_steps_ahead_list:
            best_error = np.inf
            best_params = None
            if model_name == "VAR":
                for lags in range(1, 10):
                    model = VAR(train_df)
                    model_fitted = model.fit(maxlags=lags)
                    resultado = roda_var(model_fitted, val_df, lags, steps_ahead)
                    if resultado < best_error:
                        print(lags)
                        print(resultado, best_error)
                        best_error = resultado
                        best_params = lags
                        salva(caminho_de_saida, data_index, steps_ahead, best_error, best_params, params_column_name='lags')
            else:
                def arima_para_coluna(coluna, order):
                    try:
                        model = ARIMA(train_df[coluna].values, order=order)
                        p, d, q = order
                        model_fitted = model.fit()
                        val_data = pd.DataFrame.from_records({coluna: val_df[coluna].values}).reset_index(drop=True)
                        relative_errors = []
                        for i, row in val_data.iterrows():
                            if i > p and i + 1 + steps_ahead < val_data.shape[0]:
                                input_data, target = val_data.iloc[:i + 1], val_data.iloc[i + 1 + steps_ahead]
                                forecast = list(model_fitted.apply(input_data).forecast(steps_ahead))[-1]
                                relative_errors.append(np.abs((target - forecast) / (forecast + 0.0001)))
                        return np.mean(relative_errors)
                    except:
                        return np.inf

                def objective(trial, study):
                    p = trial.suggest_int('p', 1, 10)
                    d = trial.suggest_int('d', 0, 8)
                    q = trial.suggest_int('q', 0, 8)
                    order = (p, d, q)
                    if aggregation_type == "median":
                        resultado = arima_para_coluna('med', order)
                    else:
                        resultado = 0
                        num_cols = len(train_df.columns)
                        for col in train_df.columns:
                            resultado += arima_para_coluna(col, order)/num_cols
                    try:
                        best_value = study.best_value
                    except ValueError:
                        best_value = np.inf
                    if resultado < best_value:
                        best_error = resultado
                        best_params = (p, d, q)
                        salva(caminho_de_saida, data_index, steps_ahead, best_error, best_params)
                    return resultado

                study = optuna.create_study(direction='minimize', study_name=f'ARIMA {aggregation_type} {data_index}: s{steps_ahead} p{partition_size}')
                study.optimize(lambda trial: objective(trial, study), n_trials=30)

