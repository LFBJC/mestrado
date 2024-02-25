import os
import sys

import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import plot_single_box_plot_series, plot_multiple_box_plot_series

entrada = r"E:\mestrado\Pesquisa\Dados reais\Dados tratados\demanda energética - kaggle"
caminho_de_saida = "E:/mestrado/Pesquisa/Dados reais/resultados VAR.csv"
lags = 1
steps_ahead_list = [1, 5, 20]
if os.path.exists(caminho_de_saida):
    results_df = pd.read_csv(caminho_de_saida)
    max_data_index = max(results_df['Data_set'])
    results_df = pd.concat([
        results_df,
        pd.DataFrame({
            'Data_set': [entrada]
        })
    ], ignore_index=True)
else:
    results_df = pd.DataFrame({
        'Data_set': [entrada]
    })
    for steps_ahead in steps_ahead_list:
        results_df[f'VAR_result ({steps_ahead} steps ahead)'] = [np.nan]
for steps_ahead in steps_ahead_list:
    if np.isnan(results_df[f'VAR_result ({steps_ahead} steps ahead)'][results_df["Data_set"] == entrada].iloc[0]):
        df = pd.read_csv(f'{entrada}/train.csv')
        # plot_single_box_plot_series(df.to_dict('records'), title='Dados de Treinamento')
        model = VAR(df[[c for c in df.columns if c != 'SETTLEMENTDATE']])
        model_fitted = model.fit(maxlags=lags)
        df_test = pd.read_csv(f'{entrada}/test.csv')

        relative_errors = []
        entradas_teste = []
        series_forecasts = []
        series_targets = []
        for i, row in df_test.iterrows():
            if i + lags + steps_ahead < df_test.shape[0]:
                # Fazer previsões para o próximo período
                entradas_teste.extend(df_test[[c for c in df.columns if c != 'SETTLEMENTDATE']].iloc[i:i+lags].to_dict('records'))
                forecast = model_fitted.forecast(df_test[[c for c in df.columns if c != 'SETTLEMENTDATE']].values[i:i+lags], steps=steps_ahead)[0]
                series_forecasts.append({c: v for c, v in zip(df_test.columns, forecast) if c != 'SETTLEMENTDATE'})
                series_targets.append(df_test[[c for c in df.columns if c != 'SETTLEMENTDATE']].iloc[i+steps_ahead+lags].to_dict())
                # Calcular os erros relativos
                relative_errors.append(np.abs((df_test[[c for c in df.columns if c != 'SETTLEMENTDATE']].values[i+steps_ahead+lags] - forecast) / (forecast + 0.0001)))
        # plot_single_box_plot_series(entradas_teste, title='Entrada de teste')
        # plot_multiple_box_plot_series([series_targets, series_forecasts])

        # Calcular o MMRE
        mmre = np.mean(relative_errors)
        results_df.loc[results_df['Data_set'] == entrada, f'VAR_result ({steps_ahead} steps ahead)'] = mmre

        print("Mean Magnitude of Relative Errors (MMRE):", mmre)
        print('#' * 80)
results_df.to_csv(caminho_de_saida, index=False)
