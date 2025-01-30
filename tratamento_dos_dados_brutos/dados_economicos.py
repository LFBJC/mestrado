# dados provenientes do site investing.com
import pandas as pd
import numpy as np
import os

pasta_dos_dados = "C:/mestrado/Pesquisa/Dados reais/Dados brutos"

for conjunto in ["USD_CHF Dados Históricos.csv"]: # "GBP_USD Dados Históricos.csv", "USD_JPY Dados Históricos.csv", "Dados Históricos - Bitcoin.csv", "BRL_USD Dados Históricos.csv", "EUR_USD Dados Históricos.csv"]:
    pasta_de_saida = f"C:/mestrado/Pesquisa/Dados reais/Dados tratados/{conjunto}"
    df = pd.read_csv(f"{pasta_dos_dados}/{conjunto}")
    df = df.sort_values(by="Data")
    df["Último"] = df["Último"].str.translate(str.maketrans({",": ".", ".": ""})).astype(float)
    s0, s1 = int(0.7 * df.shape[0]), int(0.85 * df.shape[0])
    train_data, val_data, test_data = df.iloc[:s0], df.iloc[s0:s1], df.iloc[s1:]
    os.makedirs(f"{pasta_de_saida}/sem agrupamento", exist_ok=True)
    train_data.to_csv(f"{pasta_de_saida}/sem agrupamento/train.csv")
    val_data.to_csv(f"{pasta_de_saida}/sem agrupamento/val.csv")
    test_data.to_csv(f"{pasta_de_saida}/sem agrupamento/test.csv")
    df["Data"] = df["Data"].map(lambda x: ".".join(x.split(".")[:-1]))
    grouped = pd.DataFrame({
        'whislo': df.groupby("Data")["Último"].min(),
        'q1': df.groupby("Data")["Último"].apply(lambda data: np.quantile(data, 0.25)),
        'med': df.groupby("Data")["Último"].apply(lambda data: np.quantile(data, 0.5)),
        'q3': df.groupby("Data")["Último"].apply(lambda data: np.quantile(data, 0.75)),
        'whishi': df.groupby("Data")["Último"].max()
    })
    s0, s1 = int(0.7 * grouped.shape[0]), int(0.85 * grouped.shape[0])
    train_data, val_data, test_data = grouped.iloc[:s0], grouped.iloc[s0:s1], grouped.iloc[s1:]
    os.makedirs(f"{pasta_de_saida}/agrupado em boxplots", exist_ok=True)
    train_data.to_csv(f"{pasta_de_saida}/agrupado em boxplots/train.csv")
    val_data.to_csv(f"{pasta_de_saida}/agrupado em boxplots/val.csv")
    test_data.to_csv(f"{pasta_de_saida}/agrupado em boxplots/test.csv")
