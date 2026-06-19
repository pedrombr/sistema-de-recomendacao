import numpy as np
#from importdataset import carregar_dataset_rating

#dataset_notas = carregar_dataset_rating()

#matriz_notas = np.array(dataset_notas)

#notas = matriz_notas[:, 2]
#usuarios = matriz_notas[:, 0]

#matriz_notas = dataset_notas.pivot_table(
#    index='UserID',
#    columns='MovieID',
#   values='Rating',
#    fill_value = 0
#)

#Parte do item-based, vai até a normalização
#valores_matriz = matriz_notas.values

#soma_notas_usuarios = np.sum(valores_matriz, axis=1)

#qtd_filmes_avaliados = np.sum(valores_matriz > 0, axis=1)

#medias_usuarios = np.zeros_like(soma_notas_usuarios, dtype=float)
#mascara_validas = qtd_filmes_avaliados > 0
#medias_usuarios[mascara_validas] = soma_notas_usuarios[mascara_validas] / qtd_filmes_avaliados[mascara_validas]

def similaridade_cosseno(vetor_X, vetor_Y):

    produto_vetores = np.dot(vetor_X, vetor_Y)

    norma_vetor_X = np.linalg.norm(vetor_X)
    norma_vetor_Y = np.linalg.norm(vetor_Y)

    if norma_vetor_X == 0 and norma_vetor_Y == 0:
        return 0

    similaridade = produto_vetores / (norma_vetor_X * norma_vetor_Y)

    return similaridade

def cosseno_ajustado(vetor_item_X, vetor_item_Y, vetor_medias_usuarios):
    correlacionados = (vetor_item_X > 0) & (vetor_item_Y > 0)
    medias_filtradas = vetor_medias_usuarios[correlacionados]

    if np.sum(correlacionados) < 2:
        return 0.0

    notas_X = vetor_item_X[correlacionados]
    notas_Y = vetor_item_Y[correlacionados]

    sub_X = notas_X - medias_filtradas
    sub_Y = notas_Y - medias_filtradas

    numerador = np.dot(sub_X, sub_Y)
    denominador = np.linalg.norm(sub_X) * np.linalg.norm(sub_Y)

    if denominador == 0:
        return 0.0

    return numerador / denominador


def correlacao_pearson(vetor_X, vetor_Y):
    #itens co-relacionados
    corelacionados = (vetor_X > 0) & (vetor_Y > 0)

    vetor_X_corelato = vetor_X[corelacionados]
    vetor_Y_corelato = vetor_Y[corelacionados]

    if len(vetor_X_corelato) < 5:
        return 0.0

    media_vetor_X = np.mean(vetor_X_corelato)
    media_vetor_Y = np.mean(vetor_Y_corelato)

    sub_vetor_X = np.subtract(vetor_X_corelato, media_vetor_X)
    sub_vetor_Y = np.subtract(vetor_Y_corelato, media_vetor_Y)

    numerador = np.dot(sub_vetor_X,sub_vetor_Y)

    denominador = np.linalg.norm(sub_vetor_X) * np.linalg.norm(sub_vetor_Y)

    if denominador == 0:
        return 0.0

    n = len(vetor_X_corelato)
    shrinkage = n / (n + 10)
    pearson = (numerador / denominador) * shrinkage

    return pearson

def rmse(y_predict, y_real):
    erro = np.sqrt(np.mean((y_real - y_predict) ** 2))
    return erro
