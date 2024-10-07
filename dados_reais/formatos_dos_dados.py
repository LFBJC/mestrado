import os
import pandas as pd
pasta_dados = "C:/mestrado/Pesquisa/Dados reais/Dados tratados"
for conjunto in os.listdir(pasta_dados):
    print(conjunto+":")
    for tipo_de_divisao in os.listdir(f"{pasta_dados}/{conjunto}"):
        print(f"\t{tipo_de_divisao}:")
        for csv_file in os.listdir(f"{pasta_dados}/{conjunto}/{tipo_de_divisao}"):
            if csv_file.endswith('.csv'):
                df = pd.read_csv(f"{pasta_dados}/{conjunto}/{tipo_de_divisao}/{csv_file}")
                print(f"\t\t\t{csv_file}: {df.shape}")