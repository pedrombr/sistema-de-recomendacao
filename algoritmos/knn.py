import pandas as pd
import numpy as np
from importdataset import carregar_dataset_rating

dataset_notas = carregar_dataset_rating()
matriz_notas = np.array(dataset_notas)

notas = matriz_notas[:, 2]
usuarios = matriz_notas[:, 0]

#mascara_notas = (notas > 0)
# O que fazer com os valores faltantes??
#usuarios_com_notas = usuarios[mascara_notas]

#Usuário x Filmes

matriz_notas = dataset_notas.pivot_table(
    index='UserID',
    columns='MovieID',
    values='Rating',
    fill_value = 0
)

def normalizacao_z_score(vetor_notas):

    notas_validas = vetor_notas > 0

    if len(notas_validas) == 0:
        return vetor_notas

    media = np.mean(vetor_notas[notas_validas])
    desvio_padrao = np.std(vetor_notas[notas_validas])

    if desvio_padrao == 0:
        return 'Desvio padrão == 0'

    vetor_normalizado = vetor_notas.copy().astype(float)

    vetor_normalizado[notas_validas] = (vetor_notas[notas_validas] - media) / desvio_padrao
    return vetor_normalizado

#notas_normalizadas = normalizacao_z_score(mascara_notas)

def similaridade_cosseno(vetor_X, vetor_Y):

    produto_vetores = np.dot(vetor_X, vetor_Y)

    norma_vetor_X = np.linalg.norm(vetor_X)
    norma_vetor_Y = np.linalg.norm(vetor_Y)

    if norma_vetor_X == 0 and norma_vetor_Y == 0:
        return 0

    similaridade = produto_vetores / (norma_vetor_X * norma_vetor_Y)

    return similaridade

def knn_user_based(user_id, k_vizinhos=5, n_recomendacoes=5, normalizar=False):

    vetor_alvo = matriz_notas.loc[user_id].values
    vetor_alvo_calc = normalizacao_z_score(vetor_alvo) if normalizar else vetor_alvo

    similaridades = []
    usuarios_ids = []

    for outro_user in matriz_notas.index:
        if outro_user == user_id:
            continue

        vetor_outro = matriz_notas.loc[outro_user].values
        vetor_outro_calc = normalizacao_z_score(vetor_outro) if normalizar else vetor_outro

        sim = similaridade_cosseno(vetor_alvo_calc, vetor_outro_calc)

        if sim > 0:
            similaridades.append(sim)
            usuarios_ids.append(outro_user)

        similaridades = np.array(similaridades)
        usuarios_ids = np.array(usuarios_ids)

        if len(similaridades) == 0:
            return "Nenhum vizinho com gosto similar encontrado."

        k_real = min(k_vizinhos, len(similaridades))
        indices_top_k = np.argsort(similaridades)[-k_real:]

        top_k_sims = similaridades[indices_top_k]
        top_k_users = usuarios_ids[indices_top_k]

        filmes_nao_assistidos = matriz_notas.columns[vetor_alvo == 0]

        previsoes = []

        for filme_id in filmes_nao_assistidos:
            notas_vizinhos = matriz_notas.loc[top_k_users, filme_id].values

            mascara_assistiram = notas_vizinhos > 0

            if not np.any(mascara_assistiram):
                continue

            sims_validas = top_k_sims[mascara_assistiram]
            notas_validas = notas_vizinhos[mascara_assistiram]

            numerador = np.dot(sims_validas, notas_validas)
            denominador = np.sum(np.abs(sims_validas))

            if denominador > 0:
                nota_prevista = numerador / denominador
                previsoes.append((filme_id, nota_prevista))

        previsoes.sort(key=lambda x: x[1], reverse=True)
        top_n_filmes = previsoes[:n_recomendacoes]

        return top_n_filmes
