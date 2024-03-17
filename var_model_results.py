import os
import pickle
import optuna
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR, ARIMA
from utils import plot_single_box_plot_series, plot_multiple_box_plot_series, cria_ou_atualiza_arquivo_no_drive
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

id_pasta_arima = "1gJplBlGG7U1LNQmMIngJMkNymPNEZQKk"
caminho_dados_simulados_local = "~/arquivos_mestrado/Dados simulados"

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


def salva(drive, caminho_de_saida, data_index, steps_ahead, best_error, best_params, params_column_name='params'):
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
    if drive is not None:
        cria_ou_atualiza_arquivo_no_drive(
            drive, id_pasta_arima, caminho_de_saida.split('/')[-1], caminho_de_saida
        )

if __name__ == '__main__':
    model_name = "ARIMA" # "VAR" #
    aggregation_type = "boxplot" # only relevant for ARIMA
    config = 4
    # data_index = 2
    partition_size = 100 # 360, 250, 100
    steps_ahead_list = [1, 5, 20]
    for data_index in range(2, 10):
        # if data_index == 1:
        #     local_steps_ahead_list=[5,20]
        # else:
        local_steps_ahead_list = steps_ahead_list
        caminho_de_saida = f"{caminho_dados_simulados_local}/{model_name}/config {config}/{aggregation_type}/particao de tamanho {partition_size}.csv"
        pasta_saida = '/'.join(caminho_de_saida.replace('\\', '/').split('/')[:-1])
        os.makedirs(os.path.dirname(caminho_de_saida), exist_ok=True)
        caminho_dados = f'{caminho_dados_simulados_local}/Dados/config {config}/{data_index}/partition size {partition_size}/'
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
                        pickle.dump(model_fitted, open(f"{pasta_saida}/bestModel_{data_index}_{steps_ahead}.pkl", 'wb'))
                        salva(None, caminho_de_saida, data_index, steps_ahead, best_error, best_params, params_column_name='lags')
            else:
                def arima_para_coluna(coluna, order):
                    try:
                        model = ARIMA(train_df[coluna].values, order=order)
                        p, d, q = order
                        model_fitted = model.fit()
                        val_data = pd.DataFrame.from_records({coluna: val_df[coluna].values}).reset_index(drop=True)
                        relative_errors_val = []
                        for i, row in val_data.iterrows():
                            if i > p and i + 1 + steps_ahead < val_data.shape[0]:
                                input_data, target = val_data.iloc[:i + 1], val_data.iloc[i + 1 + steps_ahead]
                                forecast = list(model_fitted.apply(input_data).forecast(steps_ahead))[-1]
                                relative_errors_val.append(np.abs((target - forecast) / (forecast + 0.0001)))
                        return model_fitted, np.mean(relative_errors_val)
                    except:
                        return None, np.inf

                def objective(trial, study):
                    p = trial.suggest_int('p', 1, 10)
                    d = trial.suggest_int('d', 0, 2)
                    q = trial.suggest_int('q', 0, 8)
                    order = (p, d, q)
                    cols_to_models = {}
                    if aggregation_type == "median":
                        cols_to_models["med"], resultado = arima_para_coluna('med', order)
                    else:
                        resultado = 0
                        num_cols = len(train_df.columns)
                        for col in train_df.columns:
                            cols_to_models[col], resultado_col = arima_para_coluna(col, order)
                            resultado += resultado_col/num_cols
                    try:
                        best_value = study.best_value
                    except ValueError:
                        best_value = np.inf
                    if resultado < best_value:
                        best_error = resultado
                        best_params = (p, d, q)
                        # Autenticação
                        gauth = GoogleAuth()
                        scope = ['https://www.googleapis.com/auth/drive']
                        creds = ServiceAccountCredentials.from_json_keyfile_name('conta-de-servico.json', scope)
                        gauth.credentials = creds

                        # Criação do objeto drive
                        drive = GoogleDrive(gauth)
                        for col, model in cols_to_models.items():
                            os.makedirs(f"{pasta_saida}/{partition_size}", exist_ok=True)
                            caminho_pickle_modelo = f"{pasta_saida}/{partition_size}/bestModel_{data_index}_{steps_ahead}_{col}.pkl"
                            pickle.dump(model, open(caminho_pickle_modelo, 'wb'))
                            caminho_modelo_drive = f'{partition_size}/bestModel_{data_index}_{steps_ahead}_{col}.pkl'
                            cria_ou_atualiza_arquivo_no_drive(
                                drive, id_pasta_arima, caminho_modelo_drive, caminho_pickle_modelo
                            )
                        salva(drive, caminho_de_saida, data_index, steps_ahead, best_error, best_params)
                    return resultado

                study = optuna.create_study(direction='minimize', study_name=f'ARIMA {aggregation_type} {data_index}: s{steps_ahead} p{partition_size}')
                study.optimize(lambda trial: objective(trial, study), n_trials=30)

