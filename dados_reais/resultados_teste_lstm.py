import os.path

import numpy as np
import pandas as pd
import tensorflow as tf
from utils import MMRE, normalize_data, denormalize_data, from_ranges, to_ranges

conjuntos = ["beijing", "cafe", "demanda energética - kaggle", "KAGGLE - HOUSE HOLD ENERGY CONSUMPTION"]
pasta_modelos = "C:/mestrado/Pesquisa/Dados reais/LSTM/Saída da otimização de hiperparâmetros"
pasta_dados_tratados = "C:/mestrado/Pesquisa/Dados reais/Dados tratados"
redes_neurais_df = pd.DataFrame(columns=['Modelo', 'Conjunto', 'Passos à frente', 'Score'])
pasta_saida = "C:/mestrado/Pesquisa/Dados reais/LSTM"
colunas_por_conjunto = {
    "demanda energética - kaggle": "TOTALDEMAND",
    "cafe": "money",
    "beijing": "pm2.5",
    "KAGGLE - HOUSE HOLD ENERGY CONSUMPTION": "USAGE",
    "WIND POWER GERMANY": "MW",
}
for tipo_de_agrupamento in ["com boxplot", "sem agrupamento"]:
    print(tipo_de_agrupamento)
    for conjunto in conjuntos:
        print(conjunto)
        if tipo_de_agrupamento == "com boxplot":
            tipo_de_agrupamento_pasta_dados = "agrupado em boxplots"
            columns = ['whislo', 'q1', 'med', 'q3', 'whishi']
            train_df = pd.read_csv(f"{pasta_dados_tratados}/{conjunto}/{tipo_de_agrupamento_pasta_dados}/train.csv")
            min_ = train_df['whislo'].min()
            max_ = train_df['whishi'].max()
        else:
            tipo_de_agrupamento_pasta_dados = tipo_de_agrupamento
            columns = [colunas_por_conjunto[conjunto]]
            train_df = pd.read_csv(f"{pasta_dados_tratados}/{conjunto}/{tipo_de_agrupamento_pasta_dados}/train.csv")
            min_ = train_df[columns[0]].min()
            max_ = train_df[columns[0]].max()
        for steps_ahead in [1, 5, 20]:
            print(f"{steps_ahead} steps ahead")
            opt_hist_df = pd.read_csv(f"{pasta_modelos}/{tipo_de_agrupamento}/{conjunto}/{steps_ahead} steps ahead/opt_hist.csv")
            win_size = opt_hist_df[opt_hist_df['score'] == opt_hist_df['score'].min()]['win_size'].iloc[0]
            train_df = pd.read_csv(f"{pasta_dados_tratados}/{conjunto}/{tipo_de_agrupamento_pasta_dados}/train.csv")
            teste_df = pd.read_csv(f"{pasta_dados_tratados}/{conjunto}/{tipo_de_agrupamento_pasta_dados}/test.csv")
            test_data = teste_df[columns].to_dict('records')
            data = np.array([np.array(list(x.values())) for x in test_data])
            X_test = np.array([data[i:i + win_size] for i, j in
                               zip(range(0, win_size * ((len(data) - steps_ahead) // win_size), win_size),
                                   range((len(data) - win_size - steps_ahead) // win_size))])
            Y_test = np.array([data[i + win_size + steps_ahead] for i, j in
                               zip(range(0, win_size * ((len(data) - steps_ahead) // win_size), win_size),
                                   range((len(data) - win_size - steps_ahead) // win_size))])
            X_test, Y_test = normalize_data(X_test, Y_test, min_, max_)
            X_test, Y_test = to_ranges(X_test, axis=1), to_ranges(Y_test, axis=1)
            print(f"{pasta_modelos}/{tipo_de_agrupamento}/{conjunto}/{steps_ahead} steps ahead/best_model.h5")
            lstm_model = tf.keras.models.load_model(f"{pasta_modelos}/{tipo_de_agrupamento}/{conjunto}/{steps_ahead} steps ahead/best_model.h5", custom_objects={'MMRE': MMRE})
            print(lstm_model.summary())
            T = from_ranges(lstm_model.predict(X_test, verbose=0), axis=1)
            error = tf.keras.backend.eval(
                MMRE(denormalize_data(from_ranges(Y_test, axis=1), min_, max_), denormalize_data(T, min_, max_))
            )
            redes_neurais_df = pd.concat([redes_neurais_df, pd.DataFrame.from_records([{
                'Modelo': f'LSTM - {tipo_de_agrupamento}', 'Conjunto': conjunto, 'Passos à frente': steps_ahead, 'Score': error
            }])])
redes_neurais_df.to_excel(f"{pasta_saida}/Resultados de Teste.xlsx", index=False)
