import pandas as pd
import numpy as np
from metricas import (similaridade_cosseno, correlacao_pearson, rmse)


class KNNUserBased:

    valores_k_para_testar = [5, 10, 15, 20]

    def __init__(
        self,
        k_vizinhos=5,
        n_recomendacoes=5,
        normalizar=True,
        metrica="pearson"
    ):

        self.k_vizinhos = k_vizinhos
        self.n_recomendacoes = n_recomendacoes
        self.normalizar = normalizar
        self.metrica = metrica

        self.matriz_similaridade = None


    def normalizacao_z_score(self, vetor_notas):

        notas_validas = vetor_notas > 0

        if np.sum(notas_validas) == 0:
            return vetor_notas

        media = np.mean(vetor_notas[notas_validas])
        desvio_padrao = np.std(vetor_notas[notas_validas])

        if desvio_padrao == 0:
            return vetor_notas

        vetor_normalizado = vetor_notas.copy().astype(float)

        vetor_normalizado[notas_validas] = ((vetor_notas[notas_validas] - media)/ desvio_padrao)

        return vetor_normalizado


    def fit(self, matriz_treino):

        self.matriz_treino = matriz_treino

        # pré-cálculo das similaridades
        self.matriz_similaridade = (matriz_treino.T.corr(method='pearson'))

        return self


    def knn_user_based(self, matriz_treino, user_id):

        # Verifica se o usuário existe no treino
        if user_id not in matriz_treino.index:
            return "Usuário não encontrado na base de treino"

        vetor_alvo = matriz_treino.loc[user_id].values

        filmes_nao_assistidos = (matriz_treino.columns[vetor_alvo == 0])

        similaridades_usuario = (self.matriz_similaridade[user_id].drop(user_id).fillna(0))

        similaridades_usuario = (similaridades_usuario[similaridades_usuario > 0])

        if similaridades_usuario.empty:
            return "Nenhum vizinho com gosto similar encontrado."

        top_k_vizinhos = similaridades_usuario.nlargest(
            self.k_vizinhos
        )

        previsoes = []

        for filme_id in filmes_nao_assistidos:

            notas_vizinhos = matriz_treino.loc[
                top_k_vizinhos.index,
                filme_id
            ]

            mascara_assistiram = notas_vizinhos > 0

            if not np.any(mascara_assistiram):
                continue

            sims_validas = top_k_vizinhos[mascara_assistiram]

            notas_validas = notas_vizinhos[mascara_assistiram]

            numerador = np.dot(sims_validas,notas_validas)

            denominador = np.sum(np.abs(sims_validas))

            if denominador > 0:

                nota_prevista = (numerador / denominador)

                previsoes.append((filme_id, nota_prevista))

        previsoes.sort(key=lambda x: x[1],reverse=True)

        top_n_filmes = (previsoes[:self.n_recomendacoes])

        return top_n_filmes


    def prever_nota_user_based(self,matriz_treino,user_id,filme_id):

        # verifica usuário e filme
        if (filme_id not in matriz_treino.columns or user_id not in matriz_treino.index):
            return 0

        # usuários que avaliaram o filme
        usuarios_avaliaram = matriz_treino[
            matriz_treino[filme_id] > 0
        ]

        if usuarios_avaliaram.empty:
            return 0

        # similaridades já pré-calculadas
        similaridades = (self.matriz_similaridade[user_id].loc[usuarios_avaliaram.index].fillna(0))

        # remove o próprio usuário
        if user_id in similaridades.index:
            similaridades = similaridades.drop(user_id)

        # mantém apenas positivas
        similaridades = (similaridades[similaridades > 0])

        if similaridades.empty:
            return 0

        # top K vizinhos
        top_k_vizinhos = similaridades.nlargest(self.k_vizinhos)

        soma_pesos = top_k_vizinhos.sum()

        soma_notas_pesadas = sum(
            top_k_vizinhos[vizinho] * matriz_treino.at[vizinho, filme_id]
            for vizinho in top_k_vizinhos.index
        )

        notas_previstas = (soma_notas_pesadas / soma_pesos)

        return notas_previstas


    def avaliar_rmse_user_based(self,matriz_treino,matriz_teste):

        y_pred = []
        y_real = []

        # percorre apenas avaliações reais
        usuarios, filmes = np.where(matriz_teste.values > 0)

        for u, f in zip(usuarios, filmes):

            user_id = matriz_teste.index[u]
            filme_id = matriz_teste.columns[f]

            nota_real = matriz_teste.at[user_id,filme_id]

            nota_prevista = (self.prever_nota_user_based(matriz_treino,user_id,filme_id))

            if nota_prevista > 0:

                y_pred.append(nota_prevista)
                y_real.append(nota_real)

        y_pred = np.array(y_pred)
        y_real = np.array(y_real)

        erro_rmse = rmse(y_pred,y_real)

        return erro_rmse
