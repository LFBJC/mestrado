import os
import pandas as pd
import numpy as np

pasta_dos_dados = r"E:/mestrado/Pesquisa/Dados reais/Dados brutos/demanda energética - kaggle"
pasta_de_saida = r"E:/mestrado/Pesquisa/Dados reais/Dados tratados"
df = pd.DataFrame()
for csv_name in os.listdir(pasta_dos_dados):
    df = pd.concat([df, pd.read_csv(f"{pasta_dos_dados}/{csv_name}")])

df["SETTLEMENTDATE"] = df["SETTLEMENTDATE"].str.replace(" ([0-9]+:)+[0-9]+", "", regex=True)
grouped = pd.DataFrame({
    'whislo': df.groupby("SETTLEMENTDATE")["TOTALDEMAND"].min(),
    'q1': df.groupby("SETTLEMENTDATE")["TOTALDEMAND"].apply(lambda data: np.quantile(data, 0.25)),
    'med': df.groupby("SETTLEMENTDATE")["TOTALDEMAND"].apply(lambda data: np.quantile(data, 0.5)),
    'q3': df.groupby("SETTLEMENTDATE")["TOTALDEMAND"].apply(lambda data: np.quantile(data, 0.75)),
    'whishi': df.groupby("SETTLEMENTDATE")["TOTALDEMAND"].max()
})
s0, s1 = int(0.5*grouped.shape[0]), int(0.75*grouped.shape[0])
train_data, val_data, test_data = grouped.iloc[:s0], grouped.iloc[s0:s1], grouped.iloc[s1:]
train_data.to_csv(f"{pasta_de_saida}/train.csv")
val_data.to_csv(f"{pasta_de_saida}/val.csv")
test_data.to_csv(f"{pasta_de_saida}/test.csv")
