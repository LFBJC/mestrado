import os
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

caminho_de_saida = "E:/mestrado/Pesquisa/Dados simulados/resultados VAR.csv"
total_seeds = 1000
steps_ahead_list = [1, 5, 20]
if os.path.exists(caminho_de_saida):
    results_df = pd.read_csv(caminho_de_saida)
    max_data_index = max(results_df['Data_index'])
    if max_data_index < total_seeds - 1:
        results_df = pd.concat([
            results_df,
            pd.DataFrame({
                'Data_index': np.arange(max_data_index, total_seeds)
            })
        ], ignore_index=True)
else:
    results_df = pd.DataFrame({
        'Data_index': np.arange(total_seeds)
    })
    for steps_ahead in steps_ahead_list:
        results_df[f'VAR_result ({steps_ahead} steps ahead)'] = [np.nan] * total_seeds
indices_to_use = np.arange(1000)
for data_index in indices_to_use:
    for steps_ahead in steps_ahead_list:
        filename = f'{data_index}.csv'
        print('#' * 80)
        print(filename)
        print(f'{steps_ahead} steps ahead')
        df = pd.read_csv('E:/mestrado/Pesquisa/Dados simulados/Dados'+filename)
        model = VAR(df)
        model_fitted = model.fit(1)

        # Fazer previsões para o próximo período
        forecast = model_fitted.forecast(df.values, steps=steps_ahead)

        # Calcular os erros relativos
        relative_errors = np.abs((forecast - df.values[-1]) / df.values[-1])

        # Calcular o MMRE
        mmre = np.mean(relative_errors)
        results_df.loc[results_df['Data_index'] == data_index, f'VAR_result ({steps_ahead} steps ahead)'] = mmre

        print("Mean Magnitude of Relative Errors (MMRE):", mmre)
        print('#' * 80)
results_df.to_csv(caminho_de_saida, index=False)
