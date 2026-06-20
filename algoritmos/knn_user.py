import pandas as pd
import numpy as np
from metricas import (similaridade_cosseno, correlacao_pearson, rmse)


class KNNUserBased:

    valores_k_para_testar = [10, 15, 20, 25, 27, 30, 35]

    def __init__(self, k_vizinhos=5, n_recomendacoes=5, normalizar=True, metrica="pearson"):

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
        self.matriz_np = matriz_treino.values.astype(np.float32)
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(matriz_treino.index)}
        self.item_to_idx = {filme_id: idx for idx, filme_id in enumerate(matriz_treino.columns)}

        if self.metrica == 'pearson':
            self.matriz_similaridade = (matriz_treino.T.corr(method='pearson').fillna(0).values.astype(np.float32))
        elif self.metrica == 'pearson_unha':
            n_users = self.matriz_np.shape[0]
            self.matriz_similaridade = np.zeros((n_users, n_users),dtype=np.float32)

            for i in range(n_users):
                vetor_i = self.matriz_np[i]
                for j in range(i, n_users):
                    if i == j:

                        self.matriz_similaridade[i, j] = 1.0
                        continue

                    vetor_j = self.matriz_np[j]

                    sim = correlacao_pearson(vetor_i,vetor_j)

                    self.matriz_similaridade[i, j] = sim
                    self.matriz_similaridade[j, i] = sim

        return self


    def knn_user_based(self, matriz_treino, user_id):

        # verifica usuário
        if user_id not in self.user_to_idx:
            return "Usuário não encontrado na base de treino"

        u_idx = self.user_to_idx[user_id]

        vetor_usuario = self.matriz_np[u_idx]

        filmes_nao_assistidos = np.where(vetor_usuario == 0)[0]

        similaridades_usuario = (self.matriz_similaridade[u_idx]).copy()

        similaridades_usuario[u_idx] = 0

        mascara_positiva = (similaridades_usuario > 0)

        if not np.any(mascara_positiva):
            return "Nenhum vizinho com gosto similar encontrado."

        previsoes = []

        for filme_idx in filmes_nao_assistidos:

            notas_filme = self.matriz_np[:, filme_idx]

            mascara_avaliaram = ((notas_filme > 0)& mascara_positiva)

            if not np.any(mascara_avaliaram):
                continue

            sims_validas = similaridades_usuario[
                mascara_avaliaram
            ]

            notas_validas = notas_filme[
                mascara_avaliaram
            ]

            # top K
            if len(sims_validas) > self.k_vizinhos:

                top_k_idx = np.argpartition(
                    sims_validas,
                    -self.k_vizinhos
                )[-self.k_vizinhos:]

                sims_validas = sims_validas[top_k_idx]
                notas_validas = notas_validas[top_k_idx]

            denominador = np.sum(np.abs(sims_validas))

            if denominador > 0:

                nota_prevista = (np.dot(sims_validas,notas_validas) / denominador)

                filme_id = matriz_treino.columns[
                    filme_idx
                ]

                previsoes.append((filme_id, float(nota_prevista)))

        previsoes.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return previsoes[:self.n_recomendacoes]


    def prever_nota_user_based(self, matriz_treino, user_id, filme_id):

        # verifica existência
        if (user_id not in self.user_to_idx or filme_id not in self.item_to_idx):
            return 0

        u_idx = self.user_to_idx[user_id]
        f_idx = self.item_to_idx[filme_id]

        # usuários que avaliaram o filme
        notas_filme = self.matriz_np[:, f_idx]

        usuarios_validos = notas_filme > 0

        if not np.any(usuarios_validos):
            return 0

        # similaridades do usuário alvo
        similaridades = self.matriz_similaridade[u_idx]

        # remove próprio usuário
        usuarios_validos[u_idx] = False

        # mantém apenas similares positivos
        mascara = (usuarios_validos & (similaridades > 0))

        if not np.any(mascara):
            return 0

        sims_validas = similaridades[mascara]
        notas_validas = notas_filme[mascara]

        # top K
        if len(sims_validas) > self.k_vizinhos:

            top_k_idx = np.argpartition(
                sims_validas,
                -self.k_vizinhos
            )[-self.k_vizinhos:]

            sims_validas = sims_validas[top_k_idx]
            notas_validas = notas_validas[top_k_idx]

        soma_pesos = np.sum(sims_validas)

        if soma_pesos == 0:
            return 0

        soma_notas_pesadas = np.dot(sims_validas,notas_validas)

        nota_prevista = (soma_notas_pesadas / soma_pesos)

        return float(nota_prevista)


    def avaliar_rmse_user_based(self, matriz_treino, matriz_teste):

        y_pred = []
        y_real = []

        # cache numpy
        teste_values = matriz_teste.values

        usuarios_idx, filmes_idx = np.where(teste_values > 0)

        usuarios_ids = matriz_teste.index.to_numpy()
        filmes_ids = matriz_teste.columns.to_numpy()

        for u, f in zip(usuarios_idx, filmes_idx):

            user_id = usuarios_ids[u]
            filme_id = filmes_ids[f]

            nota_real = teste_values[u, f]

            nota_prevista = (
                self.prever_nota_user_based(
                    matriz_treino,
                    user_id,
                    filme_id
                )
            )

            if nota_prevista > 0:

                y_pred.append(nota_prevista)
                y_real.append(nota_real)

        y_pred = np.asarray(y_pred, dtype=np.float32)
        y_real = np.asarray(y_real, dtype=np.float32)

        return rmse(y_pred, y_real)
