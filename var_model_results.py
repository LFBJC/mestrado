import datetime
import os
import pickle
import warnings
from typing import Literal
import optuna
import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.api import VAR, ARIMA
from utils import plot_single_box_plot_series, plot_multiple_box_plot_series, cria_ou_atualiza_arquivo_no_drive, \
    retorna_arquivo_se_existe
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

tipo_de_dados: Literal["Simulados", "Reais"] = "Reais"
id_pasta_base_drive = "1cBW25sKEV-1CKZ0Rwazf3qodb0m9GBt1"
if tipo_de_dados == "Simulados":
    id_pasta_arima = "1vZHX9FNmRHSfDyklS473sqgNXXiygwB6"
    caminho_fonte_dados_local = "D:/mestrado/Pesquisa/Dados simulados"  # "C:/Users/User/Desktop/mestrado Felipe" #
else:
    id_pasta_arima = "1CPw6H8WUijzisrF1GjVk_xnGlHkZ2-KM"
    caminho_fonte_dados_local = "D:/mestrado/Pesquisa/Dados reais" # "C:/Users/User/Desktop/mestrado Felipe/Dados reais"  #


def roda_var(model, val_data, lags, steps_ahead):
    relative_errors = []
    for i, row in val_data.iterrows():
        if i + lags + steps_ahead < val_data.shape[0]:
            # Fazer previsões para o próximo período
            entrada_val = val_data.values[i:i + lags]
            target_val = val_data.iloc[i + steps_ahead + lags]
            forecast = model.forecast(entrada_val, steps=steps_ahead)[0]
            relative_errors.append(np.abs((target_val - forecast) / (forecast + 1e-10)))
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
    model_name = "ARIMA"  # "VAR"  #
    steps_ahead_list = [1, 5, 20]
    print(f'model_name {model_name}')
    partition_size = None  # 100  #
    if tipo_de_dados == "Simulados":
        pastas_entrada = []
        for config in range(1, 7):
            # if config == 4:
            #     data_indices = list(range(4, 10))
            # elif config == 6:
            #     data_indices = list(range(9))
            # else:
            data_indices = list(range(10))
            for data_index in data_indices:
                pastas_entrada.append(f"config {config}/partição de tamanho {partition_size}/{data_index}")
    else:
        pastas_entrada = ["demanda energética - kaggle"]
    for pasta_entrada in pastas_entrada:
        local_steps_ahead_list = steps_ahead_list
        if tipo_de_dados == "Simulados":
            config = pasta_entrada.split('/')[0].replace('config ', '')
            data_index = pasta_entrada.split('/')[-1]
            if partition_size is not None:
                caminho_saida_drive = f"config {config}/partição de tamanho {partition_size}"
                caminho_de_saida = f"{caminho_fonte_dados_local}/{model_name}/{caminho_saida_drive}/partição de tamanho {partition_size}.csv"
                pasta_saida = '/'.join(caminho_de_saida.replace('\\', '/').split('/')[:-1])
                caminho_dados_drive = f'Dados/config {config}/{data_index}/partition size {partition_size}'
                caminho_dados = f'{caminho_fonte_dados_local}/{caminho_dados_drive}'
                train_path = f'{caminho_dados}/train.csv'
                train_path_drive = f'{caminho_dados_drive}/train.csv'
                partition_size_str = f'partição de tamanho {partition_size}'
            else:
                caminho_saida_drive = f"config {config}/Dados puros"
                caminho_de_saida = f"{caminho_fonte_dados_local}/{model_name}/{caminho_saida_drive}/{model_name} with raw series.csv"
                pasta_saida = '/'.join(caminho_de_saida.replace('\\', '/').split('/')[:-1])
                caminho_dados_drive = f'Dados/config {config}/{data_index}'
                train_path = f'{caminho_dados}/raw_train_series.csv'
                train_path_drive = f'{caminho_dados_drive}/raw_train_series.csv'
                partition_size_str = 'Dados puros'
            gauth = GoogleAuth()
            scope = ['https://www.googleapis.com/auth/drive']
            creds = ServiceAccountCredentials.from_json_keyfile_name('conta-de-servico.json', scope)
            gauth.credentials = creds

            # Criação do objeto drive
            drive = GoogleDrive(gauth)
            train_and_val_file = retorna_arquivo_se_existe(drive, id_pasta_base_drive, train_path_drive)
            print(f'train_and_val_file {train_and_val_file}')
            if train_and_val_file is not None:
                os.makedirs(caminho_dados, exist_ok=True)
                train_and_val_file.GetContentFile(train_path)
                train_df = pd.read_csv(train_path)
                splitting_point = int(2 / 3 * train_df.shape[0])
                val_df = train_df.iloc[splitting_point:].reset_index(drop=True)
                train_df = train_df.iloc[:splitting_point].reset_index(drop=True)
        else:
            caminho_saida_drive = pasta_entrada
            caminho_de_saida = f"{caminho_fonte_dados_local}/{model_name}/{caminho_saida_drive}/resultados.csv"
            pasta_saida = '/'.join(caminho_de_saida.replace('\\', '/').split('/')[:-1])
            if partition_size is None:
                caminho_dados_drive = f'Dados tratados/{pasta_entrada}/sem agrupamento'
            else:
                caminho_dados_drive = f'Dados tratados/{pasta_entrada}/agrupado em boxplots'
            caminho_dados = f'{caminho_fonte_dados_local}/{caminho_dados_drive}'
            train_path = f'{caminho_dados}/train.csv'
            val_path = f'{caminho_dados}/val.csv'
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            if partition_size is None:
                train_df = train_df[["TOTALDEMAND"]]
                val_df = val_df[["TOTALDEMAND"]]
            else:
                train_df = train_df[[c for c in train_df.columns if train_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]]
                val_df = val_df[[c for c in val_df.columns if val_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]]
        os.makedirs(pasta_saida, exist_ok=True)
        for steps_ahead in local_steps_ahead_list:
            print(f'*STEPS {steps_ahead}*')
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
                        pickle.dump(model_fitted, open(f"{pasta_saida}/bestModel_{pasta_entrada.replace('/', '-')}_{steps_ahead}.pkl", 'wb'))

                        gauth = GoogleAuth()
                        scope = ['https://www.googleapis.com/auth/drive']
                        creds = ServiceAccountCredentials.from_json_keyfile_name('conta-de-servico.json', scope)
                        gauth.credentials = creds

                        # Criação do objeto drive
                        drive = GoogleDrive(gauth)

                        if tipo_de_dados == "Simulados":
                            salva(drive, caminho_de_saida, data_index, steps_ahead, best_error, best_params, params_column_name='lags')
                        # else:
                        #     salva(drive, caminho_de_saida, pasta_entrada, steps_ahead, best_error, best_params,
                        #           params_column_name='lags')
            else:
                def arima_para_coluna(coluna, order):
                    try:
                        model = ARIMA(train_df[coluna].values, order=order)
                        p, d, q = order
                        model_fitted = model.fit()
                        val_data = pd.DataFrame.from_records({coluna: val_df[coluna].values}).reset_index(drop=True)
                        relative_errors_val = []
                        for i, row in tqdm(val_data.iterrows(), total=val_data.shape[0]):
                            if i > p and i + 1 + steps_ahead < val_data.shape[0]:
                                input_data, target = val_data.iloc[:i + 1], val_data.iloc[i + 1 + steps_ahead]
                                # t0 = datetime.datetime.now()
                                forecast = list(model_fitted.extend(input_data).forecast(steps_ahead))[-1]
                                # print(datetime.datetime.now() - t0)
                                relative_errors_val.append(np.abs((target - forecast) / (forecast + 0.0001)))
                        return model_fitted, np.mean(relative_errors_val)
                    except:
                        import traceback
                        print(traceback.format_exc())
                        return None, np.inf

                def objective(trial, study):
                    p = trial.suggest_int('p', 1, 10)
                    d = trial.suggest_int('d', 0, 2)
                    q = trial.suggest_int('q', 0, 8)
                    order = (p, d, q)
                    cols_to_models = {}
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
                        # drive = GoogleDrive(gauth)
                        for col, model in cols_to_models.items():
                            os.makedirs(f"{pasta_saida}/{caminho_saida_drive}", exist_ok=True)
                            caminho_pickle_modelo = f"{pasta_saida}/bestModel_{pasta_entrada.replace('/', '-')}_{steps_ahead}_{col}.pkl"
                            pickle.dump(model, open(caminho_pickle_modelo, 'wb'))
                            # caminho_modelo_drive = f"{caminho_saida_drive}/bestModel_{pasta_entrada.replace('/', '-')}_{steps_ahead}_{col}.pkl"
                            # cria_ou_atualiza_arquivo_no_drive(
                            #     drive, id_pasta_arima, caminho_modelo_drive, caminho_pickle_modelo
                            # )
                        if tipo_de_dados == "Simulados":
                            salva(drive, caminho_de_saida, data_index, steps_ahead, best_error, best_params,
                                  params_column_name='lags')
                        # else:
                        #     salva(drive, caminho_de_saida, pasta_entrada, steps_ahead, best_error, best_params,
                        #           params_column_name='lags')
                    return resultado

                def stop_on_zero(study, trial):
                    for ii, t in enumerate(study.trials):
                        if t.value <= 1e-15:
                            study.stop()

                study = optuna.create_study(direction='minimize', study_name=f'{model_name} {pasta_entrada}')
                study.optimize(lambda trial: objective(trial, study), n_trials=30, callbacks=[stop_on_zero])

