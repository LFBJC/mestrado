import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
conjuntos = ["GBP_USD Dados Históricos.csv"] # "USD_JPY Dados Históricos.csv", "Dados Históricos - Bitcoin.csv", "BRL_USD Dados Históricos.csv", "EUR_USD Dados Históricos.csv", "USD_CHF Dados Históricos.csv"]
for conjunto in conjuntos:
    pasta_modelos = f"C:/mestrado/Pesquisa/Dados reais/ARIMA - 1 CURVA/{conjunto}"
    pasta_dados = f"C:/mestrado/Pesquisa/Dados reais/Dados tratados/{conjunto}"
    boxplot_cols = ['whishi', 'q3', 'med', 'q1', 'whislo']
    out = pd.DataFrame()
    pasta_saida = pasta_modelos
    teste_df_boxplots = pd.read_csv(f"{pasta_dados}/agrupado em boxplots/test.csv")
    teste_df_bruto = pd.read_csv(f"{pasta_dados}/sem agrupamento/test.csv")
    for file in tqdm(os.listdir(pasta_modelos)):
        if file.endswith('.pkl'):
            info = file.replace('.pkl', '').split('_')
            col = info[-1]
            if col != "Data":
                steps = info[-2]
                if col in boxplot_cols:
                    tipo = "BOXPLOT"
                    test_data = teste_df_boxplots[col].values
                else:
                    tipo = "DADOS PUROS"
                    test_data = teste_df_bruto[col].values
                print(test_data)
                model = pickle.load(open(f"{pasta_modelos}/{file}", "rb"))
                p = model.model.order[0]
                relative_errors_test = []
                for i in range(len(test_data)):
                    if i > p and i + 1 + int(steps) < len(test_data):
                        input_data, target = test_data[:i + 1], test_data[i + 1 + int(steps)]
                        forecast = list(model.apply(input_data).forecast(int(steps)))[-1]
                        relative_errors_test.append(np.abs((target - forecast) / (forecast + 0.0001)))
                score = np.mean(relative_errors_test)
                out = pd.concat([out, pd.DataFrame([{'Tipo': tipo, 'Coluna': col, 'Número de passos à frente': steps, 'Score': score}])]).reset_index(drop=True)
    out.sort_values(by=['Tipo', 'Coluna', 'Número de passos à frente'], inplace=True)
    out.reset_index(drop=True, inplace=True)
    out.to_excel(f"{pasta_saida}/resultados de teste.xlsx", index=False)
