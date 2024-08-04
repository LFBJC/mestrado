import os
import pandas as pd
import numpy as np

pasta_dos_dados = "D:/mestrado/Pesquisa/Dados reais/Dados brutos/NOVOS CONJUNTOS/cafe"
pasta_de_saida = "D:/mestrado/Pesquisa/Dados reais/Dados tratados/cafe"
df = pd.read_csv(f"{pasta_dos_dados}/index.csv")
s0, s1 = int(0.7*df.shape[0]), int(0.85*df.shape[0])
train_data, val_data, test_data = df.iloc[:s0], df.iloc[s0:s1], df.iloc[s1:]
os.makedirs(f"{pasta_de_saida}/sem agrupamento", exist_ok=True)
train_data.to_csv(f"{pasta_de_saida}/sem agrupamento/train.csv")
val_data.to_csv(f"{pasta_de_saida}/sem agrupamento/val.csv")
test_data.to_csv(f"{pasta_de_saida}/sem agrupamento/test.csv")

df["datetime"] = df["datetime"].str.replace(" ([0-9]+:)+[0-9]+", "", regex=True)
# print(df.groupby("datetime")['money'].min())
grouped = pd.DataFrame({
    'whislo': df.groupby("datetime")["money"].min(),
    'q1': df.groupby("datetime")["money"].apply(lambda data: np.quantile(data, 0.25)),
    'med': df.groupby("datetime")["money"].apply(lambda data: np.quantile(data, 0.5)),
    'q3': df.groupby("datetime")["money"].apply(lambda data: np.quantile(data, 0.75)),
    'whishi': df.groupby("datetime")["money"].max()
})
s0, s1 = int(0.7*grouped.shape[0]), int(0.85*grouped.shape[0])
train_data, val_data, test_data = grouped.iloc[:s0], grouped.iloc[s0:s1], grouped.iloc[s1:]
os.makedirs(f"{pasta_de_saida}/agrupado em boxplots", exist_ok=True)
train_data.to_csv(f"{pasta_de_saida}/agrupado em boxplots/train.csv")
val_data.to_csv(f"{pasta_de_saida}/agrupado em boxplots/val.csv")
test_data.to_csv(f"{pasta_de_saida}/agrupado em boxplots/test.csv")
