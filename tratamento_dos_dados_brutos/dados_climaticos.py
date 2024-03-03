import re

import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import reduce

caminho_dados_brutos = "E:/mestrado/Pesquisa/Dados reais/Dados brutos"
caminho_saida = "E:/mestrado/Pesquisa/Dados reais/Dados tratados/dados_climaticos.csv"
estacao = ""
pais = ""
latitude = ""
longitude = ""
elevacao = ""
metadado = ""
meses_do_ano = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]
df = pd.DataFrame(columns=["Estação", "País", "Latitude", "Longitude", "Elevação (em metros)", "Mês", "Ano"])
with open(f"{caminho_dados_brutos}/WWR_Region00_2011-2016_tables.txt", "r") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        try:
            if line.startswith("Station No. "):
                info = line.split(',')
                estacao, pais = info[0], info[-1]
                estacao = estacao.replace("Station No. ", "")
                estacao = estacao[:estacao.index(":")]
                pais = pais.strip()
            elif line.startswith("Latitude: "):
                latitude = re.search(r"Latitude:\s*[0-9]+\s+[0-9]+(N|S)", line).group().replace('Latitude:', '').strip().replace(' ', 'º') + '\''
                longitude = re.search(r"Longitude:\s*[0-9]+\s+[0-9]+(W|E)", line).group().replace('Longitude:', '').strip().replace(' ', 'º') + '\''
                elevacao = re.search(r"Station Elevation \(M.S.L.\):\s*-?[0-9]+\s+meters", line).group().replace('Station Elevation (M.S.L.):', '').replace('meters', '').strip()
                elevacao = float(elevacao)
            elif line.startswith("Year") or re.sub(r'\s', '', line) == "":
                pass
            elif re.search("^(19|20)[0-9][0-9]", line):
                ano_e_valores = line.split()
                ano, valores = ano_e_valores[0], ano_e_valores[1:]
                if len(valores) not in [0, 12, 13]:
                    raise ValueError()
                records = []
                for i, mes in enumerate(meses_do_ano):
                    if len(valores) > i:
                        # A IDEIA AQUI É ADICIONAR UMA COLUNA COM O NOVO METADADO AO DATAFRAME, MAS NÃO ESTÁ ACONTECENDO
                        record = {
                            "Estação": estacao, "País": pais, "Latitude": latitude, "Longitude": longitude, "Elevação (em metros)": elevacao,
                            "Mês": mes, "Ano": ano
                        }
                        filtra_condicoes = lambda colunas, valores:  reduce(lambda x, y: x & y, [df[coluna] == valor for coluna, valor in zip(colunas, valores)])
                        cond = filtra_condicoes(record.keys(), record.values())
                        if df[cond].shape[0] > 0:
                            if valores[i].isnumeric():
                                df.loc[df[cond].index, metadado] = float(valores[i])
                        else:
                            if valores[i].isnumeric():
                                record[metadado] = float(valores[i])
                            records.append(record)
                if records:
                    df = pd.concat([df, pd.DataFrame.from_records(records)])
                df.to_csv(caminho_saida, index=False)
                # print(df.shape)
            else:
                metadado = line.strip()
                # Adicionar a nova coluna ao DataFrame se ainda não estiver presente
                if metadado not in df.columns:
                    df[metadado] = np.nan
        except Exception as e:
            print(line)
            raise e
df.to_csv(caminho_saida, index=False)

