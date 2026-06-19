"""import pandas as pd
import numpy as np
from metricas import similaridade_cosseno, cosseno_ajustado, correlacao_pearson #matriz_notas, medias_usuarios

#mascara_notas = (notas > 0)
# O que fazer com os valores faltantes??
#usuarios_com_notas = usuarios[mascara_notas]

#Usuário x Filmes


def normalizacao_z_score(vetor_notas):

    notas_validas = vetor_notas > 0

    if len(notas_validas) == 0:
        return vetor_notas

    media = np.mean(vetor_notas[notas_validas])
    desvio_padrao = np.std(vetor_notas[notas_validas])

    if desvio_padrao == 0:
        return vetor_notas
        #return 'Desvio padrão == 0'

    vetor_normalizado = vetor_notas.copy().astype(float)

    vetor_normalizado[notas_validas] = (vetor_notas[notas_validas] - media) / desvio_padrao
    return vetor_normalizado

#notas_normalizadas = normalizacao_z_score(mascara_notas)
"""
