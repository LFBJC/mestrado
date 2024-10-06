import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
conjuntos = ["beijing", "cafe", "demanda energética - kaggle", "KAGGLE - HOUSE HOLD ENERGY CONSUMPTION"]
pasta_modelos = r"C:\mestrado\Pesquisa\Dados reais\VAR"
pasta_dados_tratados = r"C:\mestrado\Pesquisa\Dados reais\Dados tratados"


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

var_df = pd.DataFrame(columns=["Conjunto", "Passos à frente", "Score"])
for conjunto in conjuntos:
    print(conjunto)
    for model_file_name in tqdm(os.listdir(pasta_modelos+ f"/{conjunto}")):
        if model_file_name.endswith('.pkl'):
            print(model_file_name)
            _, _, steps_ahead, lags = model_file_name.replace('.pkl', '').split('_')
            steps_ahead = int(steps_ahead)
            lags = int(lags)
            teste_df = pd.read_csv(f"{pasta_dados_tratados}/{conjunto}/agrupado em boxplots/test.csv")
            teste_df = teste_df[['whislo', 'q1', 'med', 'q3', 'whishi']]
            print(teste_df.to_string())
            with open(f"{pasta_modelos}/{conjunto}/{model_file_name}", 'rb') as model_file:
                model = pickle.load(model_file)
                score = roda_var(model, teste_df, lags, steps_ahead)
            var_df = pd.concat([var_df, pd.DataFrame([{"Conjunto": conjunto, "Passos à frente": steps_ahead, "Score": score}])], axis=0)
    print(var_df.to_string())
var_df.to_excel(pasta_modelos+"/resultados de teste.xlsx")