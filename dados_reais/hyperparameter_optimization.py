import ast
import os
import pickle
import matplotlib.pyplot as plt
import optuna
import numpy as np
import pandas as pd
from utils import create_model, create_lstm_model, MMRE, MMRE_Loss, images_and_targets_from_data_series,\
    plot_multiple_box_plot_series, plot_single_box_plot_series, normalize_data, to_ranges, from_ranges, \
    denormalize_data, cria_ou_atualiza_arquivo_no_drive, retorna_arquivo_se_existe
from tqdm import tqdm
import tensorflow as tf

caminho_dados_reais = "E:/mestrado/Pesquisa/Dados reais" #
caminho_dados_tratados = f"{caminho_dados_reais}/Dados tratados"

def objective_cnn(trial, study, train_data, val_data, caminho_completo_saida):
    if isinstance(train_data.iloc[0], dict):
        out_size = len(train_data.iloc[0].keys())
        available_kernel_sizes = [(2, 2), (3, 2)]
    else:
        available_kernel_sizes = [(2, 1), (3, 1)]
        out_size = 1
    print(f'OUT SIZE: {out_size}')
    win_size = trial.suggest_int('win_size', 10, len(train_data)//10)
    filters_conv_1 = trial.suggest_int('filters_conv_1', 2, 10)
    kernel_size_conv_1 = trial.suggest_categorical('kernel_size_conv_1', available_kernel_sizes)
    activation_conv_1 = trial.suggest_categorical('activation_conv_1', ['relu', 'elu', 'sigmoid', 'linear', 'tanh', 'swish'])
    pool_size_1 = trial.suggest_categorical('pool_size_1', [(2, 1), (3, 1)])
    pool_type_1 = trial.suggest_categorical('pool_type_1', ['max', 'average'])
    w2 = (win_size - kernel_size_conv_1[0]) - pool_size_1[0] + 2
    h2 = (5 - kernel_size_conv_1[0]) - pool_size_1[1] + 2
    dense_neurons = trial.suggest_int('dense_neurons', 5, w2 * h2 * filters_conv_1 - 1)
    model = create_model(
        input_shape=(win_size, out_size, 1),
        filters_conv_1=filters_conv_1, kernel_size_conv_1=kernel_size_conv_1,
        activation_conv_1=activation_conv_1,
        pool_size_1=pool_size_1, pool_type_1=pool_type_1,
        # filters_conv_2=filters_conv_2, kernel_size_conv_2=kernel_size_conv_2,
        # activation_conv_2=activation_conv_2,
        dense_neurons=dense_neurons, dense_activation='sigmoid'
    )
    N_EPOCHS = 1000
    X, Y = images_and_targets_from_data_series(
        train_data.to_dict('records'), input_win_size=win_size, steps_ahead=steps_ahead
    )
    if out_size > 1:
        train_data_array = np.array([list(x.values()) for x in train_data])
    else:
        train_data_array = np.array(train_data)
    min_, max_ = np.min(train_data_array), np.max(train_data_array)
    X, Y = normalize_data(X, Y, min_, max_)
    if out_size > 1:
        X, Y = to_ranges(X), to_ranges(Y)
    model.compile(
        optimizer=trial.suggest_categorical('optimizer', ['adam', 'adadelta', 'adagrad', 'rmsprop', 'sgd']),
        loss=trial.suggest_categorical('loss', [
            'mean_squared_error', 'mean_squared_logarithmic_error', 'mean_absolute_percentage_error',
            'mean_absolute_error' # , MMRE_Loss(inverse_normalizations)
        ]),
        # loss=MMRE_Loss(inverse_normalizations),
        metrics=[MMRE]
    )
    Y_ignoring_steps_ahead = Y[:, 0, :]
    history = model.fit(
        X, Y_ignoring_steps_ahead,
        batch_size=X.shape[0],
        epochs=N_EPOCHS,
        verbose=0,
        validation_split=1/3,
        callbacks=tf.keras.callbacks.EarlyStopping(patience=10)
    )
    X_test, Y_test = images_and_targets_from_data_series(
        val_data, input_win_size=win_size, steps_ahead=steps_ahead
    )
    X_test, Y_test = normalize_data(X_test, Y_test, min_, max_)
    if out_size > 1:
        X_test, Y_test = to_ranges(X_test), to_ranges(Y_test)
        T = from_ranges(model.predict(X_test, verbose=0), axis=1)
    else:
        T = model.predict(X_test, verbose=0)
    T = denormalize_data(T, min_, max_)
    error = 0
    if steps_ahead == 1:
        error = tf.keras.backend.eval(MMRE(Y_test[:, 0, :], T))
    else:
        for s in range(2, steps_ahead+1):
            T = model.predict(X_test, verbose=0)
            if out_size > 1:
                T = from_ranges(T, axis=1)
            T = T.reshape(T.shape[0], 1, -1, 1)
            T = denormalize_data(T, min_, max_)
            X_ = np.concatenate([X_test[:, 1:, :, :], T], axis=1)
            if s == steps_ahead:
                predicted = model.predict(X_)
                if out_size > 1:
                    predicted = from_ranges(predicted, axis=1)
                error = tf.keras.backend.eval(MMRE(Y_test[:, s-1, :], denormalize_data(predicted, min_, max_)))
    if np.isnan(error):
        print(debug)
    try:
        if error < study.best_value:
            caminho_modelo_local = f'{caminho_completo_saida}/best_model.h5'
            model.save(caminho_modelo_local)
            pickle.dump(history.history, open(f'{caminho_completo_saida}/best_model_history.pkl', 'wb'))
    except ValueError:
        # if no trials are completed yet save the first trial
        caminho_modelo_local = f'{caminho_completo_saida}/best_model.h5'
        model.save(caminho_modelo_local)
        pickle.dump(history.history, open(f'{caminho_completo_saida}/best_model_history.pkl', 'wb'))
    opt_hist_df = pd.DataFrame.from_records([trial.params])
    opt_hist_df['score'] = error
    hist_path = f'{caminho_completo_saida}/opt_hist.csv'
    append_condition = os.path.exists(hist_path)
    print('append_condition:', append_condition)
    opt_hist_df.to_csv(hist_path, mode='a' if append_condition else 'w', index=False, header=(not append_condition))
    return error


def objective_lstm(trial, study, train_data, val_data,  caminho_completo_saida):
    win_size = trial.suggest_int('win_size', 10, len(train_data) // 10)
    data = np.array([np.array(list(x.values())) for x in train_data])
    min_, max_ = np.min(data), np.max(data)
    X = np.array([data[i:i+win_size] for i, j in zip(range(0, win_size*((len(data) - steps_ahead)//win_size), win_size), range((len(data) - win_size - steps_ahead)//win_size))])
    Y = np.array([data[i+win_size+steps_ahead] for i, j in zip(range(0, win_size*((len(data) - steps_ahead)//win_size), win_size), range((len(data) - win_size - steps_ahead)//win_size))])
    X, Y = normalize_data(X, Y, min_, max_)
    if isinstance(train_data.iloc[0], dict):
        X, Y = to_ranges(X, axis=1), to_ranges(Y, axis=1)
    # plot_single_box_plot_series(train_data)
    os.makedirs(f'{caminho_de_saida}/{steps_ahead} steps ahead/{data_set_index}', exist_ok=True)
    num_layers = trial.suggest_int("Número de camadas", 1, 2)
    units=[]
    activations = []
    recurrent_activations = []
    dropouts = []
    recurrent_dropouts = []
    for camada in range(num_layers):
        if camada == 0:
            units.append(trial.suggest_int(f"Unidades na camada {camada}", 4, 64))
        else:
            units.append(trial.suggest_int(f"Unidades na camada {camada}", 3, units[-1]))
        activations.append("sigmoid")
        recurrent_activations.append("sigmoid")
        dropouts.append(trial.suggest_float(f"Dropout da camada {camada}", 0.1, 0.5))
        recurrent_dropouts.append(trial.suggest_float(f"Dropout recorrente da camada {camada}", 0.1, 0.5))
    model = create_lstm_model(X.shape[1:], units, activations, recurrent_activations, dropouts, recurrent_dropouts)
    N_EPOCHS = 1000
    model.compile(
        optimizer=trial.suggest_categorical('optimizer', ['adam', 'adadelta', 'adagrad', 'rmsprop', 'sgd']),
        loss=trial.suggest_categorical('loss', [
            'mean_squared_error', 'mean_squared_logarithmic_error', 'mean_absolute_percentage_error',
            'mean_absolute_error'
        ]),
        metrics=[MMRE]
    )
    history = model.fit(
        X, Y,
        batch_size=len(X),
        epochs=N_EPOCHS,
        verbose=0,
        validation_split=1/3,
        callbacks=tf.keras.callbacks.EarlyStopping(patience=10)
    )
    if isinstance(train_data.iloc[0], dict):
        data = np.array([np.array(list(x.values())) for x in val_data])
    else:
        data = np.array(val_data)
    X_test = np.array([data[i:i + win_size] for i, j in zip(range(0, win_size*((len(data) - steps_ahead)//win_size), win_size), range((len(data) - win_size - steps_ahead)//win_size))])
    Y_test = np.array([data[i + win_size + steps_ahead] for i, j in zip(range(0, win_size*((len(data) - steps_ahead)//win_size), win_size), range((len(data) - win_size - steps_ahead)//win_size))])
    X_test, Y_test = normalize_data(X_test, Y_test, min_, max_)
    if isinstance(train_data.iloc[0], dict):
        X_test, Y_test = to_ranges(X_test, axis=1), to_ranges(Y_test, axis=1)
        T = from_ranges(model.predict(X_test, verbose=0), axis=1)
        error = tf.keras.backend.eval(
            MMRE(denormalize_data(from_ranges(Y_test, axis=1), min_, max_), denormalize_data(T, min_, max_))
        )
    else:
        T = model.predict(X_test, verbose=0)
        error = tf.keras.backend.eval(
            MMRE(denormalize_data(Y_test, min_, max_), denormalize_data(T, min_, max_))
        )
    try:
        if error < study.best_value:
            caminho_modelo_local = f'{caminho_completo_saida}/best_model.h5'
            model.save(caminho_modelo_local)
            pickle.dump(history.history, open(f'{caminho_completo_saida}/best_model_history.pkl', 'wb'))
    except ValueError:
        # if no trials are completed yet save the first trial
        caminho_modelo_local = f'{caminho_completo_saida}/best_model.h5'
        model.save(caminho_modelo_local)
        pickle.dump(history.history, open(f'{caminho_completo_saida}/best_model_history.pkl', 'wb'))
    hist_path = f'{caminho_completo_saida}/opt_hist.csv'
    if os.path.exists(hist_path):
        opt_hist_df = pd.concat([pd.read_csv(hist_path), pd.DataFrame.from_records([{**trial.params, 'score': error}])])
    else:
        opt_hist_df = pd.DataFrame.from_records([{**trial.params, 'score': error}])
    opt_hist_df.to_csv(hist_path, index=False)
    return error


aggregation_type = 'boxplot' # 'median' #
steps_ahead_list = [1, 5, 20]
n_trials = 100
objective_by_model_type = {
    'LSTM': objective_lstm,
    'CNN': objective_cnn
}
model_type = "CNN"
pastas_entrada = ["demanda energética - kaggle"]
# pastas_entrada += [f"Dados Climaticos/{x}" for x in os.listdir(f"{caminho_dados_tratados}/Dados Climaticos") if "INUTIL" not in x]

for pasta_entrada in pastas_entrada:
    saida_complemento = f"Saída da otimização de hiperparâmetros {model_type} {pasta_entrada}"
    caminho_de_saida = f"{caminho_dados_reais}/{saida_complemento}"
    objective = objective_by_model_type[model_type]
    for steps_ahead in steps_ahead_list:
        # plot_single_box_plot_series(train_data)
        caminho_completo_saida = f'{caminho_de_saida}/{steps_ahead} steps ahead/'
        os.makedirs(caminho_completo_saida, exist_ok=True)
        objective_kwargs = {'caminho_completo_saida': caminho_completo_saida}
        pasta_dados = f"{caminho_dados_tratados}/{pasta_entrada}"
        os.makedirs(pasta_dados, exist_ok=True)
        print(f"{pasta_dados}/train.csv")
        train_df = pd.read_csv(f"{pasta_dados}/train.csv")
        train_df = train_df[[c for c in train_df.columns if train_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]]
        val_df = pd.read_csv(f"{pasta_dados}/val.csv")
        val_df = val_df[[c for c in val_df.columns if val_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]]
        objective_kwargs['train_data'] = train_df
        objective_kwargs['val_data'] = val_df
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=30),
            study_name=f'hyperparameter_opt {model_type} {pasta_entrada}, {steps_ahead} steps ahead'
        )
        objective_kwargs['study'] = study
        if not os.path.exists(f'{caminho_completo_saida}/opt_hist.csv'):
            study.optimize(lambda trial: objective(trial=trial, **objective_kwargs), n_trials=n_trials)
        else:
            opt_hist_file_drive.GetContentFile(f'{caminho_completo_saida}/opt_hist.csv')
            opt_hist_df = pd.read_csv(f'{caminho_completo_saida}/opt_hist.csv')
            for _, row in opt_hist_df.iterrows():
                train_data_size = train_df.shape[0]
                if model_type == "CNN":
                    if aggregation_type == "boxplot":
                        available_kernel_sizes = [(2, 2), (3, 2)]
                    else:
                        available_kernel_sizes = [(2, 1), (3, 1)]
                    win_size = int(row['win_size'])
                    kernel_size_conv_1 = ast.literal_eval(row['kernel_size_conv_1'])
                    pool_size_1 = ast.literal_eval(row['pool_size_1'])
                    filters_conv_1 = int(row['filters_conv_1'])
                    w2 = (win_size - kernel_size_conv_1[0]) - pool_size_1[0] + 2
                    h2 = (5 - kernel_size_conv_1[0]) - pool_size_1[1] + 2
                    distributions = {
                        'win_size': optuna.distributions.IntDistribution(10, int(train_data_size/10)),
                        'filters_conv_1': optuna.distributions.IntDistribution(2, 10),
                        'kernel_size_conv_1': optuna.distributions.CategoricalDistribution(available_kernel_sizes),
                        'activation_conv_1': optuna.distributions.CategoricalDistribution(['relu', 'elu', 'sigmoid', 'linear', 'tanh', 'swish']),
                        'pool_size_1': optuna.distributions.CategoricalDistribution([(2, 1), (3, 1)]),
                        'pool_type_1': optuna.distributions.CategoricalDistribution(['max', 'average']),
                        'dense_neurons': optuna.distributions.IntDistribution(5, w2 * h2 * filters_conv_1 - 1),
                        'optimizer': optuna.distributions.CategoricalDistribution(['adam', 'adadelta', 'adagrad', 'rmsprop', 'sgd']),
                        'loss': optuna.distributions.CategoricalDistribution([
                            'mean_squared_error', 'mean_squared_logarithmic_error',
                            'mean_absolute_percentage_error',
                            'mean_absolute_error'
                        ])
                    }
                else:
                    distributions = {
                        'win_size': optuna.distributions.IntDistribution(10, int(train_data_size/10)),
                        'Número de Camadas': optuna.distributions.IntDistribution(1, 5),
                        'Unidades na camada 0': optuna.distributions.IntDistribution(4, 64),
                        'Dropout na camada 0': optuna.distributions.FloatDistribution(0.1, 0.5),
                        'Dropout recorrente na camada 0': optuna.distributions.FloatDistribution(0.1, 0.5),
                        'optimizer': optuna.distributions.CategoricalDistribution(
                            ['adam', 'adadelta', 'adagrad', 'rmsprop', 'sgd']),
                        'loss': optuna.distributions.CategoricalDistribution([
                            'mean_squared_error', 'mean_squared_logarithmic_error',
                            'mean_absolute_percentage_error',
                            'mean_absolute_error'
                        ])
                    }
                    for i in range(1, 5):
                        if f'Unidades na  camada {i - 1}' in row.to_dict().keys():
                            distributions[f'Unidades na camada {i}'] = optuna.distributions.IntDistribution(
                                4, row[f'Unidades na camada {i - 1}']),
                        else:
                            distributions[f'Unidades na camada {i}'] = optuna.distributions.IntDistribution(
                                4, 64),
                            distributions[
                                f'Dropout na camada {i}'] = optuna.distributions.FloatDistribution(
                                0.1, 0.5),
                            distributions[
                                f'Dropout recorrente na camada {i}'] = optuna.distributions.FloatDistribution(
                                0.1, 0.5)
                def value_from_row_value(c, v):
                    print(f'{c}: {v}')
                    if isinstance(v, str):
                        try:
                            return ast.literal_eval(v)
                        except ValueError:
                            return v
                    else:
                        return v
                study.add_trial(
                    optuna.trial.create_trial(
                        params={c: value_from_row_value(c, v) for c, v in row.to_dict().items() if c not in ['w2', 'w3', 'w4', 'h4', 'score']},
                        distributions=distributions,
                        value=row['score']
                    )
                )
                study.optimize(lambda trial: objective(trial=trial, **objective_kwargs),
                               n_trials=n_trials - opt_hist_df.shape[0])