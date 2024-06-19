import os
import threading
import warnings
from typing import List
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, Input, Model
from tensorflow.keras.losses import Loss
from tensorflow.python.ops.numpy_ops import np_config
from scipy.integrate import solve_ivp
from keras import backend as K
from typing import Literal
import math
from tqdm import tqdm
from functools import partial
# from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
# import rpy2.robjects as robjects
# from rpy2.robjects import conversion, default_converter
# from rpy2.robjects.packages import importr
# utils = importr('utils')
np_config.enable_numpy_behavior()



def cria_ou_atualiza_arquivo_no_drive(drive, id_pasta, caminho_arquivo_drive, caminho_local_arquivo):
    caminho_arquivo_drive = caminho_arquivo_drive.replace('\\', '/')
    if '/' in caminho_arquivo_drive:
        parent_id = cria_caminho_no_drive(drive, id_pasta, '/'.join(caminho_arquivo_drive.split('/')[:-1]))
        nome_arquivo = caminho_arquivo_drive.split('/')[-1]
    else:
        parent_id = id_pasta
        nome_arquivo = caminho_arquivo_drive
    file_list = drive.ListFile({'q': f"'{parent_id}' in parents and trashed=false"}).GetList()
    file_exists = False
    existing_file_id = None
    for file in file_list:
        if file['title'] == nome_arquivo:
            file_exists = True
            existing_file_id = file['id']
            break

    if file_exists:
        # Se o arquivo existe, atualiza o conteúdo
        file = drive.CreateFile({'id': existing_file_id})
    else:
        # Se o arquivo não existe, cria um novo arquivo
        file = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": parent_id}]})

    # Define o conteúdo do arquivo
    file.SetContentFile(caminho_local_arquivo)

    # Faz o upload ou atualização para o Drive
    file.Upload()


def cria_caminho_no_drive(drive, id_pasta_raiz, caminho):
    caminho = caminho.replace('\\', '/')
    if caminho.endswith('/'):
        caminho = caminho[:-1]
    if '/' in caminho:
        nome_pasta_atual = caminho.split('/')[0]
        proximas_pastas = caminho.split('/')[1:]
    else:
        nome_pasta_atual = caminho
        proximas_pastas = []
    file_list = drive.ListFile({'q': f"'{id_pasta_raiz}' in parents and trashed=false"}).GetList()
    file_exists = False
    current_folder_id = None
    for file in file_list:
        if file['title'] == nome_pasta_atual:
            file_exists = True
            current_folder_id = file['id']
            break
    if not file_exists:
        # Se o arquivo não existe, cria um novo arquivo
        new_folder = drive.CreateFile({"title": nome_pasta_atual, "parents": [{"id": id_pasta_raiz}], 'mimeType': 'application/vnd.google-apps.folder'})
        new_folder.Upload()
        current_folder_id = new_folder['id']
    if proximas_pastas:
        id_pasta_interna = cria_caminho_no_drive(drive, current_folder_id, '/'.join(proximas_pastas))
        return id_pasta_interna
    else:
        return current_folder_id


def retorna_arquivo_se_existe(drive, id_pasta_raiz, caminho):
    caminho = caminho.replace('\\', '/')
    if caminho.endswith('/'):
        caminho = caminho[:-1]
    file_list = drive.ListFile({'q': f"'{id_pasta_raiz}' in parents and trashed=false"}).GetList()
    for nome_pasta_atual in caminho.split('/')[:-1]:
        file_exists = False
        current_folder_id = None
        for file in file_list:
            if file['title'] == nome_pasta_atual:
                file_exists = True
                current_folder_id = file['id']
                break
        if file_exists:
            file_list = drive.ListFile({'q': f"'{current_folder_id}' in parents and trashed=false"}).GetList()
        else:
            return None
    nome_arquivo = caminho.split('/')[-1]
    file_exists = False
    ret = None
    for file in file_list:
        if file['title'] == nome_arquivo:
            file_exists = True
            ret = drive.CreateFile({'id': file['id']})
            break
    if file_exists:
        return ret
    else:
        return None


class MMRE_Loss(Loss):
    def __init__(self, inverse_normalizations):
        self.inverse_normalizations = inverse_normalizations
        super().__init__()

    def __call__(self, y_true, y_pred, sample_weight=None):
        print(y_true.shape, y_pred.shape)
        denormalized_y_true = tf.identity(y_true)
        denormalized_y_pred = tf.identity(y_pred)
        for dim in range(y_true.shape[1]):
            cond = tf.repeat(tf.range(tf.shape(y_true)[1]).reshape(1, -1), y_true.shape[0], axis=0) == dim
            print(cond.shape)
            print(tf.experimental.numpy.sum(denormalized_y_true[:, :dim], axis=-1).shape)
            add_to_y_true = tf.concat([tf.zeros((tf.shape(y_true)[0], 1), dtype=y_true.dtype), y_true[:, :-1]], axis=1)
            denormalized_y_true = denormalized_y_true + add_to_y_true
            del add_to_y_true
            add_to_y_pred = tf.concat([tf.zeros((tf.shape(y_pred)[0], 1), dtype=y_pred.dtype), y_pred[:, :-1]], axis=1)
            denormalized_y_pred = denormalized_y_pred + add_to_y_pred
            del add_to_y_pred
        def inverse_norm_gen():
            for fn in self.inverse_normalizations:
                yield fn
        denormalized_y_true = tf.map_fn(lambda i: next(inverse_norm_gen()), denormalized_y_true)
        denormalized_y_pred = tf.map_fn(lambda i: next(inverse_norm_gen()), denormalized_y_pred)
        loss = MMRE(denormalized_y_true, denormalized_y_pred)
        if sample_weight is not None:
            loss = tf.multiply(loss, sample_weight)
        return loss


def random_walk(n_samples: int = 10000, begin_value: float=None):
    if begin_value is None:
        begin_value = np.random.rand() # valor aleatório entre 0 e 1
    out = [begin_value]
    for step in range(n_samples-1):
        out.append(out[-1] + np.random.normal())
    return out


def logistic_map(n_samples: int = 10000, begin_value: float=None):
    if begin_value is None:
        begin_value = np.random.rand()  # valor aleatório entre 0 e 1
    out = [begin_value]
    for step in range(n_samples-1):
        out.append(0.7*out[-1]*(1 - out[-1]))
    return out


def henom_map(n_samples: int = 10000, begin_value: float=None):
    if begin_value is None:
        begin_value = np.random.rand() # valor aleatório entre 0 e 1
    out = [begin_value, 1 - 1.4*begin_value**2]
    for step in range(n_samples-1):
        out.append(1 - 1.4*(out[-1]**2) + 0.3*out[-2])
    return out


def generate_stochastic_van_der_pol_series(
        n_samples: int = 10000, begin_value: float=None, begin_speed: float=None,
        mu_mean: float=None, mu_variation: float=None, sigma_mean: float=None, sigma_variation: float=None
):
    """

    :param n_samples:
    :param begin_value:
    :param begin_speed:
    :param mu_mean:
    :param mu_variation:
    :param sigma_mean:
    :param sigma_variation:
    :return: a tuple containing the series and keyword arguments for creating a new series from where this series stopped
    """
    if begin_value is None:
        begin_value = np.random.uniform(-5, 5)
    if begin_speed is None:
        begin_speed = np.random.uniform(-5, 5)
    if mu_mean is None:
        mu_mean = np.random.uniform(0, 3)
    if mu_variation is None:
        mu_variation = np.random.rand()
    if sigma_mean is None:
        sigma_mean = np.random.rand()
    if sigma_variation is None:
        sigma_variation = np.random.rand()
    x = [begin_value]
    v = [begin_speed]
    print("Condições iniciais:")
    print(f"x = {x[0]}, v={v[0]}")
    print("Outras informações:")
    print(f"μ_mean = {mu_mean}, σ_mean = {sigma_mean}")
    print(f"μ_var = {mu_variation}, σ_var = {sigma_variation}")
    dt = 0.01
    for i in range(n_samples - 1):
        # print(f"time: {i}")
        mu = mu_mean + mu_variation * np.sin(2 * np.pi * i * dt)
        sigma = sigma_mean + sigma_variation * np.cos(2 * np.pi * i * dt)
        dx, dv = van_der_pol(x[i - 1], v[i - 1], mu, sigma, dt, x[i])
        # x.append(x[i - 1] + dx * dt)
        # v.append(v[i - 1] + dv * dt)
        # Verificar limites das variáveis
        if abs(x[i - 1] + dx * dt) < 1e10 and abs(v[i - 1] + dv * dt) < 1e10:
            x.append(x[i - 1] + dx * dt)
            v.append(v[i - 1] + dv * dt)
        else:
            print(f"Estouro de memória em i = {i}. Abortando simulação.")
            raise Exception()
    if len(x) != n_samples:
        print("ERROR")
        print(len(x))
        print(n_samples)
        raise Exception()
    return x, {
        'begin_speed': v[-1],
        'mu_mean': mu_mean,
        'mu_variation': mu_variation,
        'sigma_mean': sigma_mean,
        'sigma_variation': sigma_variation
    }


def van_der_pol(x, v, mu, sigma, dt, xi):
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x + sigma * xi
    return dxdt, dvdt


def NAR(n_samples: int = 10000, begin_value: float=None):
    if begin_value is None:
        begin_value = np.random.rand() # valor aleatório entre 0 e 1
    out = [begin_value]
    for step in range(n_samples-1):
        out.append(0.7*np.abs(out[-1])/(np.abs(out[-1])+2) + np.random.normal())
    return out


def STAR1(n_samples: int = 10000, begin_value: float=None):
    if begin_value is None:
        begin_value = np.random.rand() # valor aleatório entre 0 e 1
    out = [begin_value]
    for step in range(n_samples-1):
        out.append(0.8*out[-1]-0.8*out[-1]/(1 - np.exp(-10*out[-1])) + np.random.normal())
    return out


def STAR2(n_samples: int = 10000, begin_values: List[float]=None):
    if begin_values is None:
        begin_values = [np.random.rand(), np.random.rand()] # valores aleatórios entre 0 e 1
    out = begin_values
    for step in range(n_samples-1):
        out.append(0.3*out[-1]+0.6*out[-2]+(0.1-0.9*out[-1]+0.8*out[-2])/(1 + np.exp(-10*out[-1])) + np.random.normal())
    return out


def aggregate_data_by_chunks(data, chunk_size: int = 10):
    box_plots = []
    for chunk in range(int(np.ceil(len(data)/chunk_size))):
        chunk_data = data[chunk*chunk_size:chunk*chunk_size + chunk_size]
        box_plots.append({
            'whislo': min(chunk_data),
            'q1': np.quantile(chunk_data, 0.25),
            'med': np.quantile(chunk_data, 0.5),
            'q3': np.quantile(chunk_data, 0.75),
            'whishi': max(chunk_data)
        })
    return box_plots


def plot_single_box_plot_series(box_plot_series, splitters=[], title=''):
    fig, ax = plt.subplots()
    if title != '':
        ax.set_title(title)
    ax.bxp(box_plot_series, showfliers=False)
    if splitters:
        for splitter in splitters:
            ax.axvline(splitter)
    plt.show()


def plot_multiple_box_plot_series(box_plot_series, save_path='', show=True):
    if len(box_plot_series) > 1:
        assert all([len(x) == len(box_plot_series[0]) for x in box_plot_series[1:]])
        # positions = list(range(len(box_plot_series[0])))*len(box_plot_series)
        colors = ['green', 'red', 'blue']
        if len(box_plot_series) > 3:
            import random
            for _ in range(len(box_plot_series) - 3):
                r = lambda: random.randint(0, 255)
                colors.append('#%02X%02X%02X' % (r(), r(), r()))
        fig, ax = plt.subplots()
        for i, b_series in enumerate(box_plot_series):
            ax.bxp(b_series, boxprops={'color': colors[i]}, showfliers=False)
        if save_path != '':
            plt.savefig(fname=save_path)
        if show:
            plt.show()


def images_and_targets_from_data_series(data, input_win_size=20, steps_ahead = 1):
    if steps_ahead <= 0:
        raise ValueError("Can't predict negative steps ahead")
    if isinstance(data[0], dict):
        data = list(map(lambda x: list(x.values()), data))
    else:
        data = [[x] for x in data]
    images = []
    all_targets = []
    for i in range(len(data)-1-steps_ahead-input_win_size):
        image = np.array(data[i:input_win_size+i])
        image = np.expand_dims(image, len(image.shape))
        targets = np.array([data[input_win_size + i + s] for s in range(steps_ahead)])
        images.append(image)
        all_targets.append(targets)
    return np.array(images), np.array(all_targets)


def normalize_data(inputs, targets, min_, max_):
    inputs_ret =  inputs.copy()
    targets_ret = targets.copy()
    inputs_ret = (inputs_ret- min_)/(max_ - min_)
    targets_ret = (targets_ret - min_)/(max_ - min_)
    return inputs_ret, targets_ret


def to_ranges(x, axis=2):
    ret = x.copy()
    ret[tuple([slice(None)]*(axis) + [slice(1,None)])] -= ret[tuple([slice(None)]*(axis) + [slice(None, -1)])]
    return ret


def from_ranges(x, axis=2):
    ret = x.copy()
    for i in range(1, ret.shape[axis]):
        ret[tuple([slice(None)]*(axis) + [i])] += ret[tuple([slice(None)]*(axis) + [i -1])]
    return ret


def denormalize_data(inputs, min_, max_):
    ret = inputs.copy()
    ret = (ret)*(max_ - min_) + min_
    return ret


def create_model(
    input_shape=(20, 5, 1), filters_conv_1=32, kernel_size_conv_1=(4, 1), activation_conv_1='relu',
    pool_size_1=(2, 2), pool_type_1: Literal["max", "average"] = "max",
    # filters_conv_2=16, kernel_size_conv_2=(1, 2), activation_conv_2='relu',
    dense_neurons=16, dense_activation='relu'
):
    input = Input(shape=input_shape)
    conv_1 = layers.Conv2D(
        filters_conv_1, kernel_size_conv_1, activation=activation_conv_1, input_shape=input_shape
    )(input)
    if pool_type_1 == 'max':
        pooling_1 = layers.MaxPooling2D(pool_size_1)(conv_1)
    else:
        pooling_1 = layers.AveragePooling2D(pool_size_1)(conv_1)
    # conv_2 = layers.Conv2D(filters_conv_2, kernel_size_conv_2, activation=activation_conv_2)(pooling_1)
    # flatten = layers.Flatten()(conv_2)
    flatten = layers.Flatten()(pooling_1)
    hidden_dense = layers.Dense(dense_neurons, activation=dense_activation)(flatten)
    if input_shape[1] - 1 != 0:
        out_min = layers.Dense(1)(hidden_dense)
        out_ranges = layers.Dense(input_shape[1]-1, activation='relu')(hidden_dense)
        out = layers.concatenate([out_min, out_ranges])
    else:
        out = layers.Dense(1)(hidden_dense)
    model = Model(inputs=[input], outputs=[out])
    # model.summary()
    return model


def create_lstm_model(
    input_shape: tuple, num_units_by_layer: List[int], activations: List[str], recurrent_activations: List[str],
    dropouts: List[float], recurrent_dropouts: List[float]
):
    input_ = Input(shape=input_shape)
    lstm_out = None
    for i, (num_units, activation, recurrent_activation, dropout, recurrent_dropout) in enumerate(zip(
            num_units_by_layer, activations, recurrent_activations, dropouts, recurrent_dropouts
    )):
        if i == 0:
            lstm_out = layers.LSTM(
                num_units, activation=activation, recurrent_activation=recurrent_activation, dropout=dropout,
                recurrent_dropout=recurrent_dropout, return_sequences=(i<len(dropouts)-1)
            )(input_)
        else:
            lstm_out = layers.LSTM(
                num_units, activation=activation, recurrent_activation=recurrent_activation, dropout=dropout,
                recurrent_dropout=recurrent_dropout, return_sequences=(i < len(dropouts) - 1)
            )(lstm_out)
    if lstm_out is not None:
        if input_shape[1] - 1 != 0:
            out_min = layers.Dense(1)(lstm_out)
            out_ranges = layers.Dense(input_shape[1] - 1, activation='relu')(lstm_out)
            out = layers.concatenate([out_min, out_ranges])
        else:
            out = layers.Dense(1)(lstm_out)
        model = Model(inputs=[input_], outputs=[out])
        return model


def MMRE(y_true, y_pred):
    return K.mean(K.abs((y_true-y_pred)/(y_pred + K.epsilon())))


# def partitioning_and_prototype_selection(series, k_janelas=30, alpha=0.05, k_vizinhos=3):
#     # this R code is a copy of the code from Dailys
#     instalacoes = """
#         if (!require('tseries')) install.packages('tseries')
#         if (!require('FNN')) install.packages('FNN')
#         if (!require('lmtest')) install.packages('lmtest')
#         if (!require('fpp')) install.packages('fpp')
#         if (!require('xts')) install.packages('xts')
#     """
#     importacoes_de_bibliotecas_e_def_de_funcoes = '''
#         library(tseries)
#         library(FNN)
#         library(lmtest)
#         library(fpp)
#         library(xts)
#         # Função para calcular quantidade de janelas (serie, tamanho da janela, percentual de interseção)
# # retorna tamanho da serie, tamanho das janelas, quantidades de dados na interseção, quantidade de janelas)
# kJanelas <- function(x, k, pint){
#   n <- length(x)
#   j <- round(((n - (k * pint))/(k * (1 - pint))))
#   qint <- (k * (pint*100))/100
#   val <- c(n, j, qint, k)
#   return(val)
# }
#
# #Particionar a serie em janelas
# particionar <- function(y){
#   amostras <- list()
#   contador = 1
#   contador1 = qj[4]
#   temp = qj[2] - 1
#   for (i in 1:qj[2])
#   {
#     if(temp >= i){
#       nam <- paste("amostra",i, sep="_")
#       nome <- assign(nam, y[contador: contador1])
#       contador = contador + qj[4] - qj[3]
#       contador1 = contador + qj[4] - 1
#     }
#     else {
#       nam <- paste("amostra",i, sep="_")
#       temp1 = qj[1] - contador1
#       contador1 = contador1 + temp1
#       nome <- assign(nam, y[contador: contador1])
#     }
#     amostras[[i]] <- nome
#   }
#   return(amostras)
# }
#
# #Função para Seleção dos k-vizinhos (serie, quantidade de vizinhos)
# k_vizinhos <- function(x, k){
#   if (class(x) == "list"){
#     vizinhos <- list()
#     for (s in 1:qj[2]) {
#       v <- get.knn(x[[s]], k = k)
#       vizinhos[[s]] <- v$nn.index
#     }
#   }
#   else{
#     v <- get.knn(x, k = k)
#     vizinhos <- v$nn.index
#   }
#   return(vizinhos)
# }
#
# #Função para calcular IM removendo xi (variaveis x e y)
# IM <- function(xx, yy, y){
#   if(class(yy) == "list"){
#     IM_normalizada <- list()
#     for (j in 1:qj[2]) {
#       n = length(yy[[j]])
#       temporal <- c()
#       for (i in 1:n)
#       {
#         y <- yy[[j]]
#         X <- xx[[j]]
#         X <- X[-i]
#         temporal[i] <- mutinfo(X, y, k = 1, direct=TRUE)
#       }
#     #Normalizar dados [0, 1]
#       IM_normalizada[[j]] = (temporal-min(temporal))/(max(temporal))-(min(temporal))
#     }
#   }
#   else{
#       n = length(yy)
#       temporal <- c()
#       for (i in 1:n)
#       {
#         y <- y
#         X <- x
#         X <- X[-i]
#         temporal[i] <- mutinfo(X, y, k = 1, direct=TRUE)
#       }
#       #Normalizar dados [0, 1]
#       IM_normalizada = (temporal-min(temporal))/(max(temporal))-(min(temporal))
#     }
#   return(IM_normalizada)
# }
#
# #Função para Seleção dos protótipos(serie, vizinhos mas proximos, IM normalizada, alpha)
# prototipos <- function(serie, vizinhos, mutua, a, qj){
#   c = 1
#   if(class(serie) == "list"){
#     prototipo <- list()
#     serie_de_prototipos_y <- list()
#       for (k in 1:qj[2])
#       {
#         n = length(serie[[k]])
#         temp2 <- vector(mode = "numeric")
#         for (l in 1:n) {
#           cont = 0
#           temp1 <- vizinhos [[k]][l]
#           cdif <- mutua[[k]][l] - mutua[[k]][temp1]
#           if (cdif > a){
#             cont <- cont + 1
#           }
#           if (cont < c){
#             temp2[l] <- l
#           }
#         }
#         prototipo [[k]] <- temp2
#         temp3 <- na.omit(prototipo[[k]])
#         pos <- length(temp3)
#         num <- vector(mode = "numeric")
#         for(j in 1:pos)
#         {
#           p <- temp3[j]
#           num [j] <- serie [[k]][p]
#           serie_de_prototipos_y[[k]] <- num
#         }
#       }
#     }
#     else {
#         n = length(serie)
#         temp2 <- vector(mode = "numeric")
#         for (l in 1:n) {
#           cont = 0
#           temp1 <- vizinhos [l,]
#           cdif <- mutua[l] - mutua[temp1]
#           if (cdif > a){
#             cont <- cont + 1
#           }
#           if (cont < c){
#             temp2[l] <- l
#           }
#         }
#         prototipo <- temp2
#         temp3 <- na.omit(prototipo)
#         pos <- length(temp3)
#         num <- vector(mode = "numeric")
#         for(j in 1:pos)
#         {
#           p <- temp3[j]
#           num [j] <- serie [p]
#           serie_de_prototipos_y <- num
#         }
#       }
#     return(serie_de_prototipos_y)
#     }
#     '''
#     def_series_no_r = f'' \
#                       f'y <- c{tuple(series)}\n' \
#                       f'x <- lag(y)\n' \
#                       f'qj <- kJanelas(y, {k_janelas}, 0.4)\n' \
#                       f'amostras_y <- particionar (y)\n' \
#                       f'amostras_x <- particionar(x)'
#     execucao_do_codigo_em_si = f'''
#         vizinhos <- k_vizinhos(amostras_x,{k_vizinhos})
#
#         Infor_mutua <- IM(amostras_x, amostras_y, y)
#
#         serie_prototipos <- prototipos(amostras_y, vizinhos, Infor_mutua, {alpha}, qj)
#         serie_prototipos
#     '''
#     result = robjects.r(instalacoes + importacoes_de_bibliotecas_e_def_de_funcoes + def_series_no_r + execucao_do_codigo_em_si)
#     result_py = [list(x) for x in result]
#     return result_py


def partitioning_and_prototype_selection_v2(series, particoes, alpha=0.01, k=3, silent=True):
    serie_particionada = [
        (indice_particao, series[indice_particao:int((indice_particao + 1) * len(series) / particoes)])
        for indice_particao in range(particoes)
    ]
    # new_series = serie_particionada
    # Create a partial function with fixed arguments
    process_partial = partial(process_partition, alpha=alpha, k=k, silent=silent, num_partitions=particoes)
   
    with Pool(processes=min(particoes, 5)) as pool:
        new_series = list(pool.map(process_partial, serie_particionada))

    return new_series


def process_partition(tupla_index_W, alpha, k, silent, num_partitions):
    i, W = tupla_index_W
    return prototype_selection_for_partition(W, alpha, k, silent, index_partition=i, num_partitions=num_partitions)



def prototype_selection_for_partition(partition, alpha=0.01, k=3, silent=True, index_partition=1, num_partitions=10):
    if k < len(partition):
        if not silent:
            print('W:', partition)
        pbar = tqdm(
            total=len(partition)*2, colour='yellow', desc=f"progress for partition {index_partition+1}/{num_partitions}"
        )
        matriz_de_vizinhos = []
        vetor_de_informacao_mutua = []  # PSI
        Z = []
        for s, ys in enumerate(partition):
            vizinhos_mais_proximos = np.argsort([np.linalg.norm(x - ys) if i != s else np.inf for i, x in enumerate(partition)])[:k]
            matriz_de_vizinhos.append(vizinhos_mais_proximos)
            del vizinhos_mais_proximos
            Z.append(partition[:s] + partition[s + 1:])
            if len(partition) > 1:
                partition_r_vector_str =  f"c{tuple(partition)}"
            else:
                partition_r_vector_str = f"c({partition[0]})"
            if len(Z[s]) > 1:
                Z_s_r = f"c{tuple(Z[s])}"
            else:
                Z_s_r = f"c({Z[s][0]})"
            r_code = f"library(FNN)\nmutinfo({partition_r_vector_str}, {Z_s_r}, k = 1, direct=TRUE)"
            result = utils.capture_output(robjects.r(r_code))
            vetor_de_informacao_mutua.append(
                result[0]
            )
            pbar.update(1)
        if not silent:
            spaces = int(len(partition) / k + 1) * ' '
            for row in matriz_de_vizinhos:
                print('|' + spaces.join([str(x + 1) for x in row]) + '|')
            for s, ys in enumerate(partition):
                SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
                print(f'Z{s + 1}:'.translate(SUBSCRIPT), Z[s])
                print(f'Ψ{s + 1}:'.translate(SUBSCRIPT), vetor_de_informacao_mutua[s])
        vetor_de_informacao_mutua = np.array(vetor_de_informacao_mutua)
        dividendo = (vetor_de_informacao_mutua - vetor_de_informacao_mutua.min(axis=0))
        divisor = (vetor_de_informacao_mutua.max(axis=0) - vetor_de_informacao_mutua.min(axis=0))
        vetor_de_informacao_mutua = dividendo / divisor
        if not silent:
            print('Normalized Ψ:', vetor_de_informacao_mutua)
        delta = []
        for i, psi in enumerate(vetor_de_informacao_mutua):
            cont = 0
            for j in range(k):
                diff = psi - vetor_de_informacao_mutua[matriz_de_vizinhos[i][j]]
                if diff > alpha:
                    cont += 1
            pbar.update(1)
            if cont < k:
                delta.append(partition[i])
        return delta
    else:
        warnings.warn(f"k is {k} which is las than len(W)={len(partition)}. skipping!!!")
        return partition


def list_of_lists_to_list_of_boxplots(data):
    box_plots = []
    for chunk_data in data:
        box_plots.append({
            'whislo': min(chunk_data),
            'q1': np.quantile(chunk_data, 0.25),
            'med': np.quantile(chunk_data, 0.5),
            'q3': np.quantile(chunk_data, 0.75),
            'whishi': max(chunk_data)
        })
    return box_plots


def simple_partition(series, particoes):
    return [
        series[int(indice_particao*len(series) / particoes):int((indice_particao + 1) * len(series) / particoes)]
        for indice_particao in range(particoes)
    ]

if __name__ == "__main__":
    # plot_single_box_plot_series(aggregate_data_by_chunks(random_walk(), 100))
    # plot_multiple_box_plot_series([aggregate_data_by_chunks(random_walk(), 100), aggregate_data_by_chunks(random_walk(), 100)])
    # print(prototype_selection(np.arange(1000)))
    selected = partitioning_and_prototype_selection_v2([0.1064, 0.3803, 0.9427, 0.2161, 0.6775], particoes=1, silent=False)
    print('selected:', selected)
    # print('old method:', partitioning_and_prototype_selection([0.1064, 0.3803, 0.9427, 0.2161, 0.6775], k_janelas=5, alpha=0.01))
