import pandas as pd
import os

pasta_de_entrada = "E:/mestrado/Pesquisa/Dados reais/Dados brutos/demanda energética - kaggle"
caminho_de_saida = "E:/mestrado/Pesquisa/Dados reais/Dados tratados/demanda_energetica_kaggle_agregado.csv"
for csv_path in os.listdir(pasta_de_entrada):
    ano_e_mes = csv_path.replace("PRICE_AND_DEMAND_", "").replace("_NSW1.csv", "")
    ano_e_mes = ano_e_mes[:4] + '-' + ano_e_mes[4:]
