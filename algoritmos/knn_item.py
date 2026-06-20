import pandas as pd
import numpy as np
from metricas import (similaridade_cosseno, cosseno_ajustado, correlacao_pearson, rmse)


class KNNItemBased:

    # lista de k para testar no KNN
    valores_k_para_testar = [10, 15, 20, 25, 27, 30, 35]

    def __init__(self, k_vizinhos=5, n_recomendacoes=5, normalizar=True, metrica="cosseno_ajustado"):

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


    def fit(self, matriz_treino, medias_treino):

            self.matriz_treino = matriz_treino

            self.matriz_np = (matriz_treino.values.astype(np.float32))

            self.medias_treino = (np.asarray(medias_treino,dtype=np.float32))

            self.user_to_idx = {user_id: idx for idx, user_id in enumerate(matriz_treino.index)}

            self.item_to_idx = {filme_id: idx for idx, filme_id in enumerate(matriz_treino.columns)}

            filmes = matriz_treino.columns
            n_filmes = len(filmes)

            self.matriz_similaridade = np.zeros((n_filmes, n_filmes),dtype=np.float32)

            for i in range(n_filmes):

                vetor_i = self.matriz_np[:, i]

                if self.normalizar:
                    vetor_i = self.normalizacao_z_score(vetor_i)

                for j in range(i, n_filmes):

                    if i == j:

                        self.matriz_similaridade[i, j] = 1.0
                        continue

                    vetor_j = self.matriz_np[:, j]

                    if self.normalizar:
                        vetor_j = (self.normalizacao_z_score(vetor_j))

                    correlacionados = ((vetor_i > 0) & (vetor_j > 0))

                    if np.sum(correlacionados) < 5:
                        continue

                    x = vetor_i[correlacionados]
                    y = vetor_j[correlacionados]

                    sim = 0.0

                    if self.metrica == "cosseno_ajustado":

                        medias = (self.medias_treino[correlacionados])
                        sim = cosseno_ajustado(x, y,medias)

                    elif self.metrica == "pearson":

                        sim = correlacao_pearson(x,y)

                    elif self.metrica == "cosseno":

                        sim = similaridade_cosseno(x, y)

                    # shrinkage
                    n = len(x)

                    shrinkage = n / (n + 10)

                    sim *= shrinkage

                    self.matriz_similaridade[i, j] = sim
                    self.matriz_similaridade[j, i] = sim

            return self


    def knn_item_based(self,matriz_treino,medias_treino,user_id):

            if user_id not in self.user_to_idx:
                return (
                    "Usuário não encontrado "
                    "na base de treino"
                )

            u_idx = self.user_to_idx[user_id]

            vetor_usuario = self.matriz_np[u_idx]

            filmes_assistidos = np.where(
                vetor_usuario > 0
            )[0]

            filmes_nao_assistidos = np.where(
                vetor_usuario == 0
            )[0]

            if len(filmes_assistidos) == 0:

                return ("O usuário não avaliou nenhum filme.")

            previsoes = []

            for filme_alvo_idx in filmes_nao_assistidos:

                similaridades = (self.matriz_similaridade[filme_alvo_idx,filmes_assistidos])

                mascara = similaridades > 0

                if not np.any(mascara):
                    continue

                sims_validas = similaridades[mascara]

                filmes_validos = filmes_assistidos[mascara]

                notas_usuario = vetor_usuario[filmes_validos]

                if len(sims_validas) > self.k_vizinhos:

                    top_k_idx = np.argpartition(sims_validas,-self.k_vizinhos)[-self.k_vizinhos:]

                    sims_validas = (sims_validas[top_k_idx])

                    notas_usuario = (notas_usuario[top_k_idx])

                denominador = np.sum(np.abs(sims_validas))

                if denominador > 0:

                    nota_prevista = (np.dot(sims_validas,notas_usuario) / denominador)

                    filme_id = (matriz_treino.columns[filme_alvo_idx])

                    previsoes.append((filme_id,float(nota_prevista)))

            previsoes.sort(key=lambda x: x[1],reverse=True)

            return previsoes[ :self.n_recomendacoes]


    def prever_nota_item_based(self, matriz_treino, medias_treino, user_id,filme_id):

            if (user_id not in self.user_to_idx or filme_id not in self.item_to_idx):
                return 0

            u_idx = self.user_to_idx[user_id]
            f_idx = self.item_to_idx[filme_id]

            vetor_usuario = self.matriz_np[u_idx]

            filmes_assistidos = np.where(vetor_usuario > 0)[0]

            if len(filmes_assistidos) == 0:
                return 0

            similaridades = (self.matriz_similaridade[f_idx,filmes_assistidos])

            mascara = similaridades > 0

            if not np.any(mascara):
                return 0

            sims_validas = similaridades[mascara]

            filmes_validos = filmes_assistidos[mascara]

            notas_usuario = vetor_usuario[filmes_validos]

            if len(sims_validas) > self.k_vizinhos:

                top_k_idx = np.argpartition(sims_validas,-self.k_vizinhos)[-self.k_vizinhos:]

                sims_validas = (sims_validas[top_k_idx])

                notas_usuario = (notas_usuario[top_k_idx])

            soma_pesos = np.sum(np.abs(sims_validas))

            if soma_pesos == 0:
                return 0

            nota_prevista = (np.dot(sims_validas,notas_usuario) / soma_pesos)

            return float(nota_prevista)


    def avaliar_rmse_item_based(self, matriz_treino, matriz_teste, medias_treino):

        y_pred = []
        y_real = []

        # percorre apenas avaliações reais
        usuarios, filmes = np.where(matriz_teste.values > 0)

        for u, f in zip(usuarios, filmes):

            user_id = matriz_teste.index[u]
            filme_id = matriz_teste.columns[f]

            nota_real = matriz_teste.at[user_id,filme_id]

            nota_prevista = (self.prever_nota_item_based(matriz_treino, medias_treino, user_id, filme_id))

            if nota_prevista > 0:

                y_pred.append(nota_prevista)
                y_real.append(nota_real)

        y_pred = np.array(y_pred)
        y_real = np.array(y_real)

        erro_rmse = rmse(y_pred, y_real)

        return erro_rmse
