import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
# COMO EU ESQUECI DE SALVAR ESSA INFORMAÇÃO EM ALGUNS CASOS ISSO AJUDA
pasta_arima = "C:/mestrado/Pesquisa/Dados reais/ARIMA - 5 CURVAS"
conjuntos = ["Amazon"] # "USD_JPY Dados Históricos.csv", "Dados Históricos - Bitcoin.csv", "BRL_USD Dados Históricos.csv", "EUR_USD Dados Históricos.csv", "GBP_USD Dados Históricos.csv", "beijing", "demanda energética - kaggle", "KAGGLE - HOUSE HOLD ENERGY CONSUMPTION"]
out = pd.DataFrame()
for conjunto in conjuntos:
    pasta_modelos = f"{pasta_arima}/{conjunto}"
    pasta_dados = f"C:/mestrado/Pesquisa/Dados reais/Dados tratados/{conjunto}"
    boxplot_cols = ['whishi', 'q3', 'med', 'q1', 'whislo']
    val_df_boxplots = pd.read_csv(f"{pasta_dados}/agrupado em boxplots/val.csv")
    val_df_bruto = pd.read_csv(f"{pasta_dados}/sem agrupamento/val.csv")
    for file in tqdm(os.listdir(pasta_modelos)):
        if file.endswith('.pkl'):
            info = file.replace('.pkl', '').split('_')
            col = info[-1]
            if col != "Data":
                steps = info[-2]
                if col in boxplot_cols:
                    tipo = "BOXPLOT"
                    val_data = val_df_boxplots[col].values
                else:
                    tipo = "DADOS PUROS"
                    val_data = val_df_bruto[col].values
                # print(val_data)
                model = pickle.load(open(f"{pasta_modelos}/{file}", "rb"))
                p = model.model.order[0]
                relative_errors_val = []
                for i in range(len(val_data)):
                    if i > p and i + 1 + int(steps) < len(val_data):
                        input_data, target = val_data[:i + 1], val_data[i + 1 + int(steps)]
                        forecast = list(model.apply(input_data).forecast(int(steps)))[-1]
                        relative_errors_val.append(np.abs((target - forecast) / (forecast + 0.0001)))
                score = np.mean(relative_errors_val)
                out = pd.concat([out, pd.DataFrame(
                    [{'Conjunto': conjunto, 'Tipo': tipo, 'Coluna': col, 'Número de passos à frente': steps,
                      'P': p, 'D': model.model.order[1], 'Q': model.model.order[2], 'Score': score}]
                )]).reset_index(drop=True)
    out.sort_values(by=['Conjunto', 'Tipo', 'Coluna', 'Número de passos à frente'], inplace=True)
    out.reset_index(drop=True, inplace=True)
out.to_excel(f"{pasta_arima}/resultados de validacao.xlsx", index=False)