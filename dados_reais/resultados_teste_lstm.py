import numpy as np
import pandas as pd
import tensorflow as tf
from utils import MMRE, normalize_data, denormalize_data, from_ranges, to_ranges

conjuntos = ["beijing", "cafe", "demanda energética - kaggle", "KAGGLE - HOUSE HOLD ENERGY CONSUMPTION"]
pasta_modelos = "C:/mestrado/Pesquisa/Dados reais/LSTM/Saída da otimização de hiperparâmetros"
pasta_dados_tratados = "C:/mestrado/Pesquisa/Dados reais/Dados tratados"
redes_neurais_df = pd.DataFrame(columns=['Conjunto', 'Passos à Frente', 'Tipo de Modelo', 'Score de Teste'])
for tipo_de_agrupamento in ["com boxplot", "sem agrupamento"]:
    if tipo_de_agrupamento == "com boxplot":
        tipo_de_agrupamento_pasta_dados = "agrupado em boxplots"
    else:
        tipo_de_agrupamento_pasta_dados = tipo_de_agrupamento
    print(tipo_de_agrupamento)
    for conjunto in conjuntos:
        print(conjunto)
        for steps_ahead in [1, 5, 20]:
            print(f"{steps_ahead} steps ahead")
            opt_hist_df = pd.read_csv(f"{pasta_modelos}/{tipo_de_agrupamento}/{conjunto}/{steps_ahead} steps ahead/opt_hist.csv")
            win_size = opt_hist_df[opt_hist_df['score'] == opt_hist_df['score'].min()]['win_size'].iloc[0]
            train_df = pd.read_csv(f"{pasta_dados_tratados}/{conjunto}/{tipo_de_agrupamento_pasta_dados}/train.csv")
            teste_df = pd.read_csv(f"{pasta_dados_tratados}/{conjunto}/{tipo_de_agrupamento_pasta_dados}/test.csv")
            min_ = train_df['whislo'].min()
            max_ = train_df['whishi'].max()
            test_data = teste_df[['whislo', 'q1', 'med', 'q3', 'whishi']].to_dict('records')
            data = np.array([np.array(list(x.values())) for x in test_data])
            X_test = np.array([data[i:i + win_size] for i, j in
                               zip(range(0, win_size * ((len(data) - steps_ahead) // win_size), win_size),
                                   range((len(data) - win_size - steps_ahead) // win_size))])
            Y_test = np.array([data[i + win_size + steps_ahead] for i, j in
                               zip(range(0, win_size * ((len(data) - steps_ahead) // win_size), win_size),
                                   range((len(data) - win_size - steps_ahead) // win_size))])
            X_test, Y_test = normalize_data(X_test, Y_test, min_, max_)
            X_test, Y_test = to_ranges(X_test, axis=1), to_ranges(Y_test, axis=1)
            lstm_model = tf.keras.models.load_model(f"{pasta_modelos}/{tipo_de_agrupamento}/{conjunto}/{steps_ahead} steps ahead/best_model.h5", custom_objects={'MMRE': MMRE})
            T = from_ranges(lstm_model.predict(X_test, verbose=0), axis=1)
            error = tf.keras.backend.eval(
                MMRE(denormalize_data(from_ranges(Y_test, axis=1), min_, max_), denormalize_data(T, min_, max_))
            )
            redes_neurais_df = pd.concat([redes_neurais_df, pd.DataFrame.from_records([{
                'Conjunto': conjuntos, 'Passos à Frente': steps_ahead, 'Tipo de Modelo': 'LSTM', 'Score de Teste': error
            }])])
