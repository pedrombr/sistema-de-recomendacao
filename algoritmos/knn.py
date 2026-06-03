import pandas as pd
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

def knn_user_based(matriz_treino, user_id, k_vizinhos=5, n_recomendacoes=5, normalizar=False, metrica="pearson"):

    # Verifica se o usuário existe no treino
    if user_id not in matriz_treino.index:
         return "Usuário não encontrado na base de treino"
    
    vetor_alvo = matriz_treino.loc[user_id].values
    vetor_alvo_calc = normalizacao_z_score(vetor_alvo) if normalizar else vetor_alvo

    similaridades = []
    usuarios_ids = []

    for outro_user in matriz_treino.index:
        if outro_user == user_id:
            continue

        vetor_outro = matriz_treino.loc[outro_user].values
        vetor_outro_calc = normalizacao_z_score(vetor_outro) if normalizar else vetor_outro

        valor_similaridade = 0.0

        if metrica == 'cosseno':
            valor_similaridade = similaridade_cosseno(vetor_alvo_calc, vetor_outro_calc)
        if metrica == 'pearson':
            valor_similaridade = correlacao_pearson(vetor_alvo_calc, vetor_outro_calc)

        if valor_similaridade > 0:
            similaridades.append(valor_similaridade)
            usuarios_ids.append(outro_user)

    similaridades = np.array(similaridades)
    usuarios_ids = np.array(usuarios_ids)

    if len(similaridades) == 0:
        return "Nenhum vizinho com gosto similar encontrado."

    k_real = min(k_vizinhos, len(similaridades))
    indices_top_k = np.argsort(similaridades)[-k_real:]

    top_k_sims = similaridades[indices_top_k]
    top_k_users = usuarios_ids[indices_top_k]

    filmes_nao_assistidos = matriz_treino.columns[vetor_alvo == 0]

    previsoes = []

    for filme_id in filmes_nao_assistidos:
        notas_vizinhos = matriz_treino.loc[top_k_users, filme_id].values

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


# previsão de notas para user-based e item-based

def prever_nota_user_based(matriz_treino, user_id, filme_id, k_vizinhos=5):
    
    # ve se o filme e o usuário existem na matriz de treino
    if filme_id not in matriz_treino.columns or user_id not in matriz_treino.index:
        return 0 

    # pega apenas os usuários que realmente avaliaram o filme alvo
    usuarios_avaliaram = matriz_treino[matriz_treino[filme_id] > 0]
    
    if usuarios_avaliaram.empty:
        return 0

    # calcula a similaridade (Correlação) entre o usuário alvo e os outros
    usuario_alvo = matriz_treino.loc[user_id]
    similaridades = usuarios_avaliaram.apply(
        lambda row: usuario_alvo.corr(row), axis=1
    ).fillna(0)

    # top K vizinhos com similaridade positiva
    top_k_vizinhos = similaridades[similaridades > 0].nlargest(k_vizinhos)

    if top_k_vizinhos.empty:
        return 0

    # calcula a média ponderada (Nota * Similaridade)
    soma_pesos = top_k_vizinhos.sum()
    soma_notas_pesadas = sum(top_k_vizinhos[vizinho] * matriz_treino.at[vizinho, filme_id] for vizinho in top_k_vizinhos.index)

    return soma_notas_pesadas / soma_pesos


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
