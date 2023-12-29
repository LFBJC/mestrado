import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, Input, Model
from tensorflow.keras.losses import Loss
from tensorflow.python.ops.numpy_ops import np_config
from keras import backend as K
from typing import Literal
import rpy2.robjects as robjects
np_config.enable_numpy_behavior()


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
        out.append(out[-1] + (np.random.normal()*2 - 1))
    return out

def logistic_map(n_samples: int = 10000, begin_value: float=None):
    if begin_value is None:
        begin_value = np.random.rand() # valor aleatório entre 0 e 1
    out = [begin_value]
    for step in range(n_samples-1):
        out.append(0.7*out[-1]*(1 - out[-1]))
    return out

def henom_map(n_samples: int = 10000, begin_value: float=None):
    if begin_value is None:
        begin_value = np.random.rand() # valor aleatório entre 0 e 1
    out = [begin_value]
    for step in range(n_samples-1):
        out.append(1 - 1.4*(out[-1]**2) + 0.3*out[-2])
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


def plot_single_box_plot_series(box_plot_series, splitters=[]):
    fig, ax = plt.subplots()
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
    data = list(map(lambda x: list(x.values()), data))
    images = []
    all_targets = []
    inverse_normalizations = []
    normalizations = []
    for i in range(len(data)-1-steps_ahead-input_win_size):
        image = np.array(data[i:input_win_size+i])
        image = np.expand_dims(image, len(image.shape))
        inverse_normalization = lambda x: (np.max(image) - np.min(image))*x + np.min(image)
        normalization = lambda x: (x - np.min(image))/(np.max(image) - np.min(image))
        normalized_image = normalization(image)
        normalizations.append(normalization)
        targets = np.array([data[input_win_size + i + s] for s in range(steps_ahead)])
        images.append(normalized_image)
        all_targets.append(targets)
        inverse_normalizations.append(inverse_normalization)
        assert (all([b_plot[0] < b_plot[1] < b_plot[2] < b_plot[3] < b_plot[4] for targets in all_targets for b_plot in
                     targets]))
    return np.array(images), np.array(all_targets), inverse_normalizations, normalizations


def create_model(
    input_shape=(20, 5, 1), filters_conv_1=32, kernel_size_conv_1=(4, 1), activation_conv_1='relu',
    pool_size_1=(2, 2), pool_type_1: Literal["max", "average"] = "max",
    filters_conv_2=16, kernel_size_conv_2=(1, 2), activation_conv_2='relu',
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
    conv_2 = layers.Conv2D(filters_conv_2, kernel_size_conv_2, activation=activation_conv_2)(pooling_1)
    flatten = layers.Flatten()(conv_2)
    hidden_dense = layers.Dense(dense_neurons, activation=dense_activation)(flatten)
    out_min = layers.Dense(1)(hidden_dense)
    out_ranges = layers.Dense(input_shape[1]-1, activation='sigmoid')(hidden_dense)
    out = layers.concatenate([out_min, out_ranges])
    return Model(inputs=[input], outputs=[out])


def MMRE(y_true, y_pred):
    return K.mean(K.abs((y_true-y_pred)/(y_pred + K.epsilon())))


def partitioning_and_prototype_selection(series, k_janelas=30):
    # this R code is a copy of the code from Dailys
    importacoes_de_bibliotecas_e_def_de_funcoes = '''
        library(tseries)
        library(FNN)
        library(lmtest)
        library(fpp)
        library(xts)
        # Função para calcular quantidade de janelas (serie, tamanho da janela, percentual de interseção)
# retorna tamanho da serie, tamanho das janelas, quantidades de dados na interseção, quantidade de janelas) 
kJanelas <- function(x, k, pint){
  n <- length(x)
  j <- round(((n - (k * pint))/(k * (1 - pint))))
  qint <- (k * (pint*100))/100
  val <- c(n, j, qint, k)
  return(val) 
}

#Particionar a serie em janelas
particionar <- function(y){
  amostras <- list()
  contador = 1
  contador1 = qj[4]
  temp = qj[2] - 1
  for (i in 1:qj[2]) 
  {
    if(temp >= i){
      nam <- paste("amostra",i, sep="_")
      nome <- assign(nam, y[contador: contador1])
      contador = contador + qj[4] - qj[3]
      contador1 = contador + qj[4] - 1
    }
    else {
      nam <- paste("amostra",i, sep="_")
      temp1 = qj[1] - contador1
      contador1 = contador1 + temp1
      nome <- assign(nam, y[contador: contador1])
    }
    amostras[[i]] <- nome
  }
  return(amostras)
}  

#Função para Seleção dos k-vizinhos (serie, quantidade de vizinhos)
k_vizinhos <- function(x, k){
  if (class(x) == "list"){
    vizinhos <- list()
    for (s in 1:qj[2]) {
      v <- get.knn(x[[s]], k = k)
      vizinhos[[s]] <- v$nn.index
    }
  }
  else{
    v <- get.knn(x, k = k)
    vizinhos <- v$nn.index
  }
  return(vizinhos)
}

#Função para calcular IM removendo xi (variaveis x e y)
IM <- function(xx, yy, y){
  if(class(yy) == "list"){
    IM_normalizada <- list()
    for (j in 1:qj[2]) {
      n = length(yy[[j]])
      temporal <- c()
      for (i in 1:n) 
      {
        y <- yy[[j]]
        X <- xx[[j]]
        X <- X[-i]
        temporal[i] <- mutinfo(X, y, k = 1, direct=TRUE)
      } 
    #Normalizar dados [0, 1]
      IM_normalizada[[j]] = (temporal-min(temporal))/(max(temporal))-(min(temporal))
    }
  }
  else{
      n = length(yy)
      temporal <- c()
      for (i in 1:n) 
      {
        y <- y
        X <- x
        X <- X[-i]
        temporal[i] <- mutinfo(X, y, k = 1, direct=TRUE)
      } 
      #Normalizar dados [0, 1]
      IM_normalizada = (temporal-min(temporal))/(max(temporal))-(min(temporal))
    }
  return(IM_normalizada)
}

#Função para Seleção dos protótipos(serie, vizinhos mas proximos, IM normalizada, alpha)
prototipos <- function(serie, vizinhos, mutua, a, qj){
  c = 1
  if(class(serie) == "list"){
    prototipo <- list()
    serie_de_prototipos_y <- list()
      for (k in 1:qj[2]) 
      {
        n = length(serie[[k]])
        temp2 <- vector(mode = "numeric")
        for (l in 1:n) {
          cont = 0
          temp1 <- vizinhos [[k]][l]
          cdif <- mutua[[k]][l] - mutua[[k]][temp1]
          if (cdif > a){
            cont <- cont + 1
          }
          if (cont < c){
            temp2[l] <- l
          }
        }
        prototipo [[k]] <- temp2
        temp3 <- na.omit(prototipo[[k]])
        pos <- length(temp3)
        num <- vector(mode = "numeric")
        for(j in 1:pos)
        {
          p <- temp3[j]
          num [j] <- serie [[k]][p]
          serie_de_prototipos_y[[k]] <- num
        }
      }
    }
    else {
        n = length(serie)
        temp2 <- vector(mode = "numeric")
        for (l in 1:n) {
          cont = 0
          temp1 <- vizinhos [l,]
          cdif <- mutua[l] - mutua[temp1]
          if (cdif > a){
            cont <- cont + 1
          }
          if (cont < c){
            temp2[l] <- l
          }
        }
        prototipo <- temp2
        temp3 <- na.omit(prototipo)
        pos <- length(temp3)
        num <- vector(mode = "numeric")
        for(j in 1:pos)
        {
          p <- temp3[j]
          num [j] <- serie [p]
          serie_de_prototipos_y <- num
        }
      }
    return(serie_de_prototipos_y)
    }
    '''
    def_series_no_r = f'' \
                      f'y <- c{tuple(series)}\n' \
                      f'x <- lag(y)\n' \
                      f'qj <- kJanelas(y, {k_janelas}, 0.4)\n' \
                      f'amostras_y <- particionar (y)\n' \
                      f'amostras_x <- particionar(x)'
    execucao_do_codigo_em_si = '''
        vizinhos <- k_vizinhos(amostras_x,3)

        Infor_mutua <- IM(amostras_x, amostras_y, y)

        serie_prototipos <- prototipos(amostras_y, vizinhos, Infor_mutua, 0.05, qj)
        serie_prototipos
    '''
    result = robjects.r(importacoes_de_bibliotecas_e_def_de_funcoes + def_series_no_r + execucao_do_codigo_em_si)
    result_py = [list(x) for x in result]
    return result_py


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

if __name__ == "__main__":
    # plot_single_box_plot_series(aggregate_data_by_chunks(random_walk(), 100))
    # plot_multiple_box_plot_series([aggregate_data_by_chunks(random_walk(), 100), aggregate_data_by_chunks(random_walk(), 100)])
    print(prototype_selection(np.arange(1000)))
