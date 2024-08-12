import pandas as pd
caminho_fonte_dados_local = "C:/Users/lfbjc/OneDrive - SM SMART ENERGY SOLUTIONS LTDA/backup/mestrado/Pesquisa/Dados reais"
cols_alvo = {
        "demanda energética - kaggle": "TOTALDEMAND",
        "cafe": "money",
        "beijing": "pm2.5",
        "Amazon": "Volume",
        "Netflix": "Volume"
}
pastas_entrada = list(cols_alvo.keys())
for pasta_entrada in pastas_entrada:
    print(pasta_entrada)
    caminho = f"{caminho_fonte_dados_local}/Dados tratados/{pasta_entrada}/agrupado em boxplots"
    df_train = pd.read_csv(f"{caminho}/train.csv")
    df_val = pd.read_csv(f"{caminho}/val.csv")
    print(df_train.shape)
    print(df_val.shape)