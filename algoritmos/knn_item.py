import pandas as pd
import numpy as np
from metricas import similaridade_cosseno, cosseno_ajustado, correlacao_pearson, rmse #matriz_notas, medias_usuarios

# lista de k para testar no KNN
valores_k_para_testar = [5, 10, 15, 20]

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


def knn_item_based(matriz_treino, medias_treino, user_id, k_vizinhos=5, n_recomendacoes=5, normalizar=False, metrica="cosseno_ajustado"):

    if user_id not in matriz_treino.index:
         return "Usuário não encontrado na base de treino"

    vetor_usuario = matriz_treino.loc[user_id].values

    filmes_assistidos = matriz_treino.columns[vetor_usuario > 0]
    filmes_nao_assistidos = matriz_treino.columns[vetor_usuario == 0]

    if len(filmes_assistidos) == 0:
        return "O usuário não avaliou nenhum filme para basearmos a recomendação."

    previsoes = []

    for filme_alvo in filmes_nao_assistidos:
        vetor_filme_alvo = matriz_treino[filme_alvo].values

        if normalizar:
            vetor_filme_alvo = normalizacao_z_score(vetor_filme_alvo)

        similaridades = []
        notas_do_usuario = []

        for filme_assistido in filmes_assistidos:
            vetor_filme_assistido = matriz_treino[filme_assistido].values

            if normalizar:
                vetor_filme_assistido = normalizacao_z_score(vetor_filme_assistido)

            sim = 0.0

            if metrica == 'cosseno_ajustado':
                sim = cosseno_ajustado(vetor_filme_alvo, vetor_filme_assistido, medias_treino)
            elif metrica == 'pearson':
                sim = correlacao_pearson(vetor_filme_alvo, vetor_filme_assistido)
            elif metrica == 'cosseno':
                sim = similaridade_cosseno(vetor_filme_alvo, vetor_filme_assistido)

            if sim > 0:
                similaridades.append(sim)
                nota_dada = matriz_treino.loc[user_id, filme_assistido]
                notas_do_usuario.append(nota_dada)

        similaridades = np.array(similaridades)
        notas_do_usuario = np.array(notas_do_usuario)

        if len(similaridades) == 0:
            continue

        k_real = min(k_vizinhos, len(similaridades))
        indices_top_k = np.argsort(similaridades)[-k_real:]

        sims_validas = similaridades[indices_top_k]
        notas_validas = notas_do_usuario[indices_top_k]

        numerador = np.dot(sims_validas, notas_validas)
        denominador = np.sum(np.abs(sims_validas))

        if denominador > 0:
            nota_prevista = numerador / denominador
            previsoes.append((filme_alvo, nota_prevista))

    previsoes.sort(key=lambda x: x[1], reverse=True)
    top_n_filmes = previsoes[:n_recomendacoes]

    return top_n_filmes


def prever_nota_item_based(matriz_treino, medias_treino, user_id, filme_id, k_vizinhos=5):

    if filme_id not in matriz_treino.columns or user_id not in matriz_treino.index:
        return 0

    # busca quais filmes o usuário já assistiu
    notas_do_usuario = matriz_treino.loc[user_id]
    filmes_assistidos = notas_do_usuario[notas_do_usuario > 0].index

    if len(filmes_assistidos) == 0:
        return 0

    # transforma a coluna do filme alvo em um vetor do numpy
    vetor_filme_alvo = matriz_treino[filme_id].values

    similaridades = {}
    for filme_assistido in filmes_assistidos:
        vetor_filme_assistido = matriz_treino[filme_assistido].values

        sim = cosseno_ajustado(vetor_filme_alvo, vetor_filme_assistido, medias_treino)

        if pd.notna(sim) and sim > 0:
            similaridades[filme_assistido] = sim

    if not similaridades:
        return 0

    # top K filmes mais parecidos
    similaridades_series = pd.Series(similaridades).nlargest(k_vizinhos)

    # média ponderada
    soma_pesos = similaridades_series.sum()
    soma_notas_pesadas = sum(similaridades_series[filme] * matriz_treino.at[user_id, filme] for filme in similaridades_series.index)

    return soma_notas_pesadas / soma_pesos
