# dados provenientes do site investing.com
import pandas as pd
import numpy as np
import os
from hv_block_cv import hv_block_cv_three_sets

pasta_dos_dados = "C:/mestrado/Pesquisa/Dados reais/Dados brutos"

for conjunto in ["USD_CHF Dados Históricos.csv", "GBP_USD Dados Históricos.csv", "USD_JPY Dados Históricos.csv", "Dados Históricos - Bitcoin.csv", "BRL_USD Dados Históricos.csv", "EUR_USD Dados Históricos.csv"]:
    pasta_de_saida = f"C:/mestrado/Pesquisa/Dados reais/Dados tratados/{conjunto}"
    df = pd.read_csv(f"{pasta_dos_dados}/{conjunto}")
    df = df.sort_values(by="Data")
    df["Último"] = df["Último"].str.translate(str.maketrans({",": ".", ".": ""})).astype(float)
    splits = hv_block_cv_three_sets(df, n_splits=5, h=50, v=30)
    for i, (train_idx, val_idx, test_idx) in enumerate(splits):
        train_data, val_data, test_data = df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
        os.makedirs(f"{pasta_de_saida}/sem agrupamento/Split {i+1}/", exist_ok=True)
        train_data.to_csv(f"{pasta_de_saida}/sem agrupamento/Split {i+1}/train.csv")
        val_data.to_csv(f"{pasta_de_saida}/sem agrupamento/Split {i+1}/val.csv")
        test_data.to_csv(f"{pasta_de_saida}/sem agrupamento/Split {i+1}/test.csv")
    df["Data"] = df["Data"].map(lambda x: ".".join(x.split(".")[:-1]))
    grouped = pd.DataFrame({
        'whislo': df.groupby("Data")["Último"].min(),
        'q1': df.groupby("Data")["Último"].apply(lambda data: np.quantile(data, 0.25)),
        'med': df.groupby("Data")["Último"].apply(lambda data: np.quantile(data, 0.5)),
        'q3': df.groupby("Data")["Último"].apply(lambda data: np.quantile(data, 0.75)),
        'whishi': df.groupby("Data")["Último"].max()
    })
    splits = hv_block_cv_three_sets(grouped, n_splits=5, h=50, v=30)
    for i, (train_idx, val_idx, test_idx) in enumerate(splits):
        train_data, val_data, test_data = grouped.iloc[train_idx], grouped.iloc[val_idx], grouped.iloc[test_idx]
        os.makedirs(f"{pasta_de_saida}/agrupado em boxplots/Split {i + 1}", exist_ok=True)
        train_data.to_csv(f"{pasta_de_saida}/agrupado em boxplots/Split {i + 1}/train.csv")
        val_data.to_csv(f"{pasta_de_saida}/agrupado em boxplots/Split {i + 1}/val.csv")
        test_data.to_csv(f"{pasta_de_saida}/agrupado em boxplots/Split {i + 1}/test.csv")
