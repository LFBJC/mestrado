import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import reduce
# TODO INVESTIGAR

caminho_dados_brutos = "E:/mestrado/Pesquisa/Dados reais/Dados brutos"
caminho_intermediario = "E:/mestrado/Pesquisa/Dados reais/Tratamentos intermediários/dados_climaticos.csv"
base_saida = "E:/mestrado/Pesquisa/Dados reais/Dados tratados/Dados Climaticos"
estacao = ""
pais = ""
latitude = ""
longitude = ""
elevacao = ""
metadado = ""
meses_do_ano = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]
# df = pd.DataFrame(columns=["Estação", "País", "Latitude", "Longitude", "Elevação (em metros)", "Mês", "Ano"])
# with open(f"{caminho_dados_brutos}/WWR_Region00_2011-2016_tables.txt", "r") as f:
#     lines = f.readlines()
#     for line in tqdm(lines):
#         try:
#             if line.startswith("Station No. "):
#                 info = line.split(',')
#                 estacao, pais = info[0], info[-1]
#                 estacao = estacao.replace("Station No. ", "")
#                 estacao = estacao[:estacao.index(":")]
#                 pais = pais.strip()
#             elif line.startswith("Latitude: "):
#                 latitude = re.search(r"Latitude:\s*[0-9]+\s+[0-9]+(N|S)", line).group().replace('Latitude:', '').strip().replace(' ', 'º') + '\''
#                 longitude = re.search(r"Longitude:\s*[0-9]+\s+[0-9]+(W|E)", line).group().replace('Longitude:', '').strip().replace(' ', 'º') + '\''
#                 elevacao = re.search(r"Station Elevation \(M.S.L.\):\s*-?[0-9]+\s+meters", line).group().replace('Station Elevation (M.S.L.):', '').replace('meters', '').strip()
#                 elevacao = float(elevacao)
#             elif line.startswith("Year") or re.sub(r'\s', '', line) == "":
#                 pass
#             elif re.search("^(19|20)[0-9][0-9]", line):
#                 ano_e_valores = line.split()
#                 ano, valores = ano_e_valores[0], ano_e_valores[1:]
#                 if len(valores) not in [0, 12, 13]:
#                     raise ValueError()
#                 records = []
#                 for i, mes in enumerate(meses_do_ano):
#                     if len(valores) > i:
#                         # A IDEIA AQUI É ADICIONAR UMA COLUNA COM O NOVO METADADO AO DATAFRAME, MAS NÃO ESTÁ ACONTECENDO
#                         record = {
#                             "Estação": estacao, "País": pais, "Latitude": latitude, "Longitude": longitude, "Elevação (em metros)": elevacao,
#                             "Mês": mes, "Ano": ano
#                         }
#                         filtra_condicoes = lambda colunas, valores:  reduce(lambda x, y: x & y, [df[coluna] == valor for coluna, valor in zip(colunas, valores)])
#                         cond = filtra_condicoes(record.keys(), record.values())
#                         if df[cond].shape[0] > 0:
#                             if valores[i].isnumeric():
#                                 df.loc[df[cond].index, metadado] = float(valores[i])
#                         else:
#                             if valores[i].replace(',', '').replace('.', '').isnumeric():
#                                 record[metadado] = float(valores[i].replace(',', '.'))
#                             records.append(record)
#                 if records:
#                     df = pd.concat([df, pd.DataFrame.from_records(records)])
#                 df.to_csv(caminho_intermediario, index=False)
#                 # print(df.shape)
#             else:
#                 metadado = line.strip()
#                 # Adicionar a nova coluna ao DataFrame se ainda não estiver presente
#                 if metadado not in df.columns:
#                     df[metadado] = np.nan
#         except Exception as e:
#             print(line)
#             raise e
# df.to_csv(caminho_intermediario, index=False)
df = pd.read_csv(caminho_intermediario)
for measure in ['Mean Daily Air Temperature (in deg Celsius)', 'Total Monthly Precipitation (in millimeters)',
                'Mean Station Pressure (in hPa)', 'Mean Sea Level Pressure (in hPa)',
                'Mean Daily Maximum Air Temperature (in deg Celsius)',
                'Mean Daily Minimum Air Temperature (in deg Celsius)',
                'Mean Daily Relative Humidity (in percent)']:
    pasta_de_saida = f"{base_saida}/{measure}"
    os.makedirs(pasta_de_saida, exist_ok=True)
    df_no_nans = df[~df[measure].isna()]
    df_no_nans['XYZ'] = df['Latitude'] + df['Longitude'] + df['Elevação (em metros)'].astype(str)
    grouped = pd.DataFrame({
        'whislo': df_no_nans.groupby("XYZ")[measure].min(),
        'q1': df_no_nans.groupby("XYZ")[measure].apply(lambda data: np.quantile(data, 0.25)),
        'med': df_no_nans.groupby("XYZ")[measure].apply(lambda data: np.quantile(data, 0.5)),
        'q3': df_no_nans.groupby("XYZ")[measure].apply(lambda data: np.quantile(data, 0.75)),
        'whishi': df_no_nans.groupby("XYZ")[measure].max()
    })
    s0, s1 = int(0.7 * grouped.shape[0]), int(0.85 * grouped.shape[0])
    train_data, val_data, test_data = grouped.iloc[:s0], grouped.iloc[s0:s1], grouped.iloc[s1:]
    train_data.to_csv(f"{pasta_de_saida}/train.csv")
    val_data.to_csv(f"{pasta_de_saida}/val.csv")
    test_data.to_csv(f"{pasta_de_saida}/test.csv")

