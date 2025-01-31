import os
import pandas as pd
import numpy as np
from hv_block_cv import hv_block_cv_three_sets

pasta_dos_dados = "C:/mestrado/Pesquisa/Dados reais/Dados brutos/beijing/PM2.5/"
pasta_de_saida = "C:/mestrado/Pesquisa/Dados reais/Dados tratados/beijing"
df = pd.read_csv(f"{pasta_dos_dados}/data.csv")
df = df[~pd.isna(df["pm2.5"])]
splits = hv_block_cv_three_sets(df, n_splits=5, h=50, v=30)
for i, (train_idx, val_idx, test_idx) in enumerate(splits):
    train_data, val_data, test_data = df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
    os.makedirs(f"{pasta_de_saida}/sem agrupamento/Split {i+1}", exist_ok=True)
    train_data.to_csv(f"{pasta_de_saida}/sem agrupamento/Split {i+1}/train.csv")
    val_data.to_csv(f"{pasta_de_saida}/sem agrupamento/Split {i+1}/val.csv")
    test_data.to_csv(f"{pasta_de_saida}/sem agrupamento/Split {i+1}/test.csv")

grouped = pd.DataFrame({
    'whislo': df.groupby(by=["year", "month", "day"])["pm2.5"].min(),
    'q1': df.groupby(by=["year", "month", "day"])["pm2.5"].apply(lambda data: np.quantile(data, 0.25)),
    'med': df.groupby(by=["year", "month", "day"])["pm2.5"].apply(lambda data: np.quantile(data, 0.5)),
    'q3': df.groupby(by=["year", "month", "day"])["pm2.5"].apply(lambda data: np.quantile(data, 0.75)),
    'whishi': df.groupby(by=["year", "month", "day"])["pm2.5"].max()
})
splits = hv_block_cv_three_sets(grouped, n_splits=5, h=50, v=30)
for i, (train_idx, val_idx, test_idx) in enumerate(splits):
    train_data, val_data, test_data = grouped.iloc[train_idx], grouped.iloc[val_idx], grouped.iloc[test_idx]
    os.makedirs(f"{pasta_de_saida}/agrupado em boxplots/Split {i+1}", exist_ok=True)
    train_data.to_csv(f"{pasta_de_saida}/agrupado em boxplots/Split {i+1}/train.csv")
    val_data.to_csv(f"{pasta_de_saida}/agrupado em boxplots/Split {i+1}/val.csv")
    test_data.to_csv(f"{pasta_de_saida}/agrupado em boxplots/Split {i+1}/test.csv")
