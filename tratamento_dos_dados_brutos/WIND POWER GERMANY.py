import pandas as pd
import numpy as np
import os
pasta_dados_reais = "C:/Users/lfbjc/OneDrive - SM SMART ENERGY SOLUTIONS LTDA/backup/mestrado/Pesquisa/Dados reais"
pasta_novos_conjuntos = f"{pasta_dados_reais}/Dados brutos/NOVOS CONJUNTOS"
caminho = f"{pasta_novos_conjuntos}/WIND POWER GERMANY/data.csv"
pasta_de_saida = "D:/mestrado/Pesquisa/Dados reais/Dados tratados/WIND POWER GERMANY"
df = pd.read_csv(caminho)
df = df[~pd.isna(df["MW"])]
s0, s1 = int(0.7*df.shape[0]), int(0.85*df.shape[0])
train_data, val_data, test_data = df.iloc[:s0], df.iloc[s0:s1], df.iloc[s1:]
os.makedirs(f"{pasta_de_saida}/sem agrupamento", exist_ok=True)
train_data.to_csv(f"{pasta_de_saida}/sem agrupamento/train.csv")
val_data.to_csv(f"{pasta_de_saida}/sem agrupamento/val.csv")
test_data.to_csv(f"{pasta_de_saida}/sem agrupamento/test.csv")
df["dt"] = df["dt"].map(lambda x: x.split(':')[0] + 'h')
grouped = pd.DataFrame({
    'whislo': df.groupby(by="dt")["MW"].min(),
    'q1': df.groupby(by="dt")["MW"].apply(lambda data: np.quantile(data, 0.25)),
    'med': df.groupby(by="dt")["MW"].apply(lambda data: np.quantile(data, 0.5)),
    'q3': df.groupby(by="dt")["MW"].apply(lambda data: np.quantile(data, 0.75)),
    'whishi': df.groupby(by="dt")["MW"].max()
})
s0, s1 = int(0.7*grouped.shape[0]), int(0.85*grouped.shape[0])
train_data, val_data, test_data = grouped.iloc[:s0], grouped.iloc[s0:s1], grouped.iloc[s1:]
os.makedirs(f"{pasta_de_saida}/agrupado em boxplots", exist_ok=True)
train_data.to_csv(f"{pasta_de_saida}/agrupado em boxplots/train.csv")
val_data.to_csv(f"{pasta_de_saida}/agrupado em boxplots/val.csv")
test_data.to_csv(f"{pasta_de_saida}/agrupado em boxplots/test.csv")
