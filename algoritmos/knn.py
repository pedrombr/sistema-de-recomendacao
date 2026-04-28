import pandas as pd
import numpy as np
from importdataset import carregar_dataset_rating

dataset_notas = carregar_dataset_rating()
matriz_notas = np.array(dataset_notas)

def knn_user_based():
    notas = matriz_notas[:, 2]
    usuarios = matriz_notas[:, 0]

    mascara_notas = (notas > 0)

    usuarios_com_notas = usuarios[mascara_notas]

    return len(usuarios_com_notas)

def normalizacao_z_score(matriz):
    matriz_np = np.array(matriz)

    desvio_padrao = np.std(matriz_np)

    if desvio_padrao == 0:
        return 'Desvio padrão == 0'

    media = np.mean(matriz_np)

    matriz_normalizada = (matriz_np - media) / desvio_padrao

    return matriz_normalizada
