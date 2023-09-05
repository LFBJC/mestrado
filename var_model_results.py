import os
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

total_seeds = 1000
if os.path.exists('results_summary.csv'):
    results_df = pd.read_csv('results_summary.csv')
    max_data_index = max(results_df['Data_index'])
    if max_data_index < total_seeds - 1:
        results_df = pd.concat([
            results_df,
            pd.DataFrame({
                'Data_index': np.arange(max_data_index, total_seeds),
                'VAR_result': [np.nan] * (total_seeds - max_data_index),
                'CNN_result': [np.nan] * (total_seeds - max_data_index)
            })
        ], ignore_index=True)
else:
    results_df = pd.DataFrame({
        'Data_index': np.arange(total_seeds), 'VAR_result': [np.nan] * total_seeds, 'CNN_result': [np.nan] * total_seeds,
    })
indices_to_use = np.arange(35,50)
for data_index in indices_to_use:
    filename = f'{data_index}.csv'
    print('#' * 80)
    print(filename)
    df = pd.read_csv('data/'+filename)
    model = VAR(df)
    model_fitted = model.fit(1)

    # Fazer previsões para o próximo período
    forecast = model_fitted.forecast(df.values, steps=1)

    # Calcular os erros relativos
    relative_errors = np.abs((forecast - df.values[-1]) / df.values[-1])

    # Calcular o MMRE
    mmre = np.mean(relative_errors)
    results_df.loc[results_df['Data_index'] == data_index, 'VAR_result'] = mmre

    print("Mean Magnitude of Relative Errors (MMRE):", mmre)
    print('#' * 80)
results_df.to_csv('results_summary.csv', index=False)
