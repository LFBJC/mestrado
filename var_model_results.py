import os
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from utils import plot_single_box_plot_series, plot_multiple_box_plot_series

config = 5
caminho_dados = f'E:/mestrado/Pesquisa/Dados simulados/Dados/config {config}'
caminho_de_saida = f"E:/mestrado/Pesquisa/Dados simulados/resultados VAR conf {config}.csv"
num_sets = 1000
lags = 1
steps_ahead_list = [1, 5, 20]
if os.path.exists(caminho_de_saida):
    results_df = pd.read_csv(caminho_de_saida)
    max_data_index = max(results_df['Data_index'])
    if max_data_index < num_sets - 1:
        results_df = pd.concat([
            results_df,
            pd.DataFrame({
                'Data_index': np.arange(max_data_index, num_sets)
            })
        ], ignore_index=True)
else:
    results_df = pd.DataFrame({
        'Data_index': np.arange(num_sets)
    })
    for steps_ahead in steps_ahead_list:
        results_df[f'VAR_result ({steps_ahead} steps ahead)'] = [np.nan] * num_sets
indices_to_use = np.arange(num_sets)
for data_index in indices_to_use:
    for steps_ahead in steps_ahead_list:
        if np.isnan(results_df[f'VAR_result ({steps_ahead} steps ahead)'].iloc[data_index]):
            df = pd.read_csv(f'{caminho_dados}/{data_index}/train.csv')
            model = VAR(df)
            model_fitted = model.fit(maxlags=lags)
            df_test = pd.read_csv(f'{caminho_dados}/{data_index}/test.csv')

            relative_errors = []
            entradas_teste = []
            series_forecasts = []
            series_targets = []
            for i, row in df_test.iterrows():
                if i + lags + steps_ahead < df_test.shape[0]:
                    # Fazer previsões para o próximo período
                    entradas_teste.extend(df_test.iloc[i:i+lags].to_dict('records'))
                    forecast = model_fitted.forecast(df_test.values[i:i+lags], steps=steps_ahead)[0]
                    series_forecasts.append({c: v for c, v in zip(df_test.columns, forecast)})
                    series_targets.append(df_test.iloc[i+steps_ahead+lags].to_dict())
                    # Calcular os erros relativos
                    relative_errors.append(np.abs((df_test.values[i + steps_ahead + lags] - forecast) / (forecast + 0.0001)))
            # Calcular o MMRE
            mmre = np.mean(relative_errors)
            results_df.loc[results_df['Data_index'] == data_index, f'VAR_result ({steps_ahead} steps ahead)'] = mmre

            print("Mean Magnitude of Relative Errors (MMRE):", mmre)
            print('#' * 80)
results_df.to_csv(caminho_de_saida, index=False)
