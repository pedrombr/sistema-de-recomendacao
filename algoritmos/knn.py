import pandas as pd
import numpy as np
from importdataset import carregar_dataset_rating

dataset_notas = carregar_dataset_rating()
matriz_notas = np.array(dataset_notas)

notas = matriz_notas[:, 2]
usuarios = matriz_notas[:, 0]

mascara_notas = (notas > 0)
# O que fazer com os valores faltantes??
usuarios_com_notas = usuarios[mascara_notas]

def similaridade(vetor_X, vetor_Y):
    vetor_X = notas
    vetor_Y = usuarios

    produto_vetores = np.dot(vetor_X, vetor_Y)

    norma_vetor_X = np.linalg.norm(vetor_X)
    norma_vetor_Y = np.linalg.norm(vetor_Y)

    if norma_vetor_X == 0 and norma_vetor_Y == 0:
        return 0

    return produto_vetores / (norma_vetor_X * norma_vetor_Y)

def knn_user_based():


    return len(usuarios_com_notas)

def normalizacao_z_score(matriz):
    matriz_np = np.array(matriz)

    desvio_padrao = np.std(matriz_np)

    if desvio_padrao == 0:
        return 'Desvio padrão == 0'

    media = np.mean(matriz_np)

    matriz_normalizada = (matriz_np - media) / desvio_padrao

    return matriz_normalizada
