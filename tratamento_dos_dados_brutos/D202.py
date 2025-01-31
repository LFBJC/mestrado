import pandas as pd
import numpy as np
import os
from hv_block_cv import hv_block_cv_three_sets

pasta_dados_reais = "C:/mestrado/Pesquisa/Dados reais"
pasta_novos_conjuntos = f"{pasta_dados_reais}/Dados brutos"
caminho = f"{pasta_novos_conjuntos}/KAGGLE - HOUSE HOLD ENERGY CONSUMPTION/D202.csv"
pasta_de_saida = "C:/mestrado/Pesquisa/Dados reais/Dados tratados/KAGGLE - HOUSE HOLD ENERGY CONSUMPTION"
df = pd.read_csv(caminho)
df["DATE AND HOUR"] = df["DATE"] + df["START TIME"].map(lambda x: ' ' + x.split(':')[0] + 'h')
splits = hv_block_cv_three_sets(df, n_splits=5, h=50, v=30)
for i, (train_idx, val_idx, test_idx) in enumerate(splits):
    train_data, val_data, test_data = df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
    os.makedirs(f"{pasta_de_saida}/sem agrupamento/Split {i+1}/", exist_ok=True)
    train_data.to_csv(f"{pasta_de_saida}/sem agrupamento/Split {i+1}/train.csv")
    val_data.to_csv(f"{pasta_de_saida}/sem agrupamento/Split {i+1}/val.csv")
    test_data.to_csv(f"{pasta_de_saida}/sem agrupamento/Split {i+1}/test.csv")

grouped = pd.DataFrame({
    'whislo': df.groupby(by="DATE AND HOUR")["USAGE"].min(),
    'q1': df.groupby(by="DATE AND HOUR")["USAGE"].apply(lambda data: np.quantile(data, 0.25)),
    'med': df.groupby(by="DATE AND HOUR")["USAGE"].apply(lambda data: np.quantile(data, 0.5)),
    'q3': df.groupby(by="DATE AND HOUR")["USAGE"].apply(lambda data: np.quantile(data, 0.75)),
    'whishi': df.groupby(by="DATE AND HOUR")["USAGE"].max()
})
splits = hv_block_cv_three_sets(grouped, n_splits=5, h=50, v=30)
for i, (train_idx, val_idx, test_idx) in enumerate(splits):
    train_data, val_data, test_data = grouped.iloc[train_idx], grouped.iloc[val_idx], grouped.iloc[test_idx]
    os.makedirs(f"{pasta_de_saida}/agrupado em boxplots/Split {i+1}", exist_ok=True)
    train_data.to_csv(f"{pasta_de_saida}/agrupado em boxplots/Split {i+1}/train.csv")
    val_data.to_csv(f"{pasta_de_saida}/agrupado em boxplots/Split {i+1}/val.csv")
    test_data.to_csv(f"{pasta_de_saida}/agrupado em boxplots/Split {i+1}/test.csv")