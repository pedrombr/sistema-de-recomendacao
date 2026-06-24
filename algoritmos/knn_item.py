import pandas as pd
import numpy as np
from metricas import (similaridade_cosseno, cosseno_ajustado ,correlacao_pearson, rmse)

class KNNItemBased:

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
        vetor_normalizado = (vetor_notas.copy().astype(float))

        if desvio_padrao == 0:
            vetor_normalizado[notas_validas] = 0.0
            return vetor_normalizado

        vetor_normalizado[notas_validas] = ((vetor_notas[notas_validas] - media) / desvio_padrao)

        return vetor_normalizado


    def fit(self, matriz_treino):

        self.matriz_treino = matriz_treino
        self.matriz_original = (matriz_treino.values.astype(np.float32))
        self.matriz_np = (self.matriz_original.copy())
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(matriz_treino.index)}
        self.item_to_idx = {filme_id: idx for idx, filme_id in enumerate(matriz_treino.columns)}

        n_items = self.matriz_np.shape[1]

        self.medias_itens = np.zeros(n_items, dtype=np.float32)

        self.desvios_itens = np.ones(n_items, dtype=np.float32)

        # NORMALIZAÇÃO POR ITEM
        if self.normalizar:

            for j in range(n_items):

                vetor_original = (self.matriz_np[:, j])
                notas_validas = (vetor_original > 0)

                if np.sum(notas_validas) == 0:
                    continue

                media = np.mean(vetor_original[notas_validas])
                desvio = np.std(vetor_original[notas_validas])

                if desvio == 0:
                    desvio = 1.0

                self.medias_itens[j] = media
                self.desvios_itens[j] = desvio

                self.matriz_np[:, j] = (self.normalizacao_z_score(vetor_original))

        # MATRIZ DE SIMILARIDADE ENTRE ITENS
        if self.metrica == 'cosseno_ajustado':
            medias_usuarios = np.zeros(self.matriz_np.shape[0],dtype=np.float32)

            for u in range(self.matriz_np.shape[0]):

                notas_usuario = self.matriz_original[u]
                notas_validas = notas_usuario > 0

                if np.sum(notas_validas) > 0:
                    medias_usuarios[u] = np.mean(notas_usuario[notas_validas])

            self.matriz_similaridade = np.zeros((n_items, n_items),dtype=np.float32)

            for i in range(n_items):

                vetor_i = self.matriz_original[:, i]

                for j in range(i, n_items):
                    if i == j:
                        self.matriz_similaridade[i, j] = 1.0
                        continue

                    vetor_j = self.matriz_original[:, j]
                    sim = cosseno_ajustado(vetor_i, vetor_j, medias_usuarios)

                    self.matriz_similaridade[i, j] = sim
                    self.matriz_similaridade[j, i] = sim

        elif self.metrica == "pearson":

            self.matriz_similaridade = (pd.DataFrame(self.matriz_np, index=matriz_treino.index, columns=matriz_treino.columns).corr(method="pearson").fillna(0).values.astype(np.float32))

        elif self.metrica == "pearson_unha":

            self.matriz_similaridade = np.zeros((n_items, n_items),dtype=np.float32)

            for i in range(n_items):
                vetor_i = self.matriz_np[:, i]
                for j in range(i, n_items):

                    if i == j:

                        self.matriz_similaridade[i, j] = 1.0
                        continue

                    vetor_j = self.matriz_np[:, j]

                    sim = correlacao_pearson(vetor_i,vetor_j)

                    self.matriz_similaridade[i, j] = sim
                    self.matriz_similaridade[j, i] = sim

        return self


    def prever_nota_item_based(self, matriz_treino, user_id, filme_id):

        if (user_id not in self.user_to_idx or filme_id not in self.item_to_idx):
            return 0

        u_idx = self.user_to_idx[user_id]
        f_idx = self.item_to_idx[filme_id]

        notas_usuario = (self.matriz_original[u_idx])
        filmes_avaliados = (notas_usuario > 0)

        # remove o próprio filme
        filmes_avaliados[f_idx] = False

        if not np.any(filmes_avaliados):
            return 0

        similaridades = (self.matriz_similaridade[f_idx])

        mascara = (filmes_avaliados & (similaridades > 0))

        if not np.any(mascara):
            return 0

        sims_validas = (similaridades[mascara])
        notas_validas = (self.matriz_np[u_idx, mascara])

        # TOP K
        if len(sims_validas) > self.k_vizinhos:
            top_k_idx = np.argpartition(sims_validas, -self.k_vizinhos)[-self.k_vizinhos:]
            sims_validas = (sims_validas[top_k_idx])
            notas_validas = (notas_validas[top_k_idx])

        denominador = np.sum(np.abs(sims_validas))

        if denominador == 0:
            return 0

        # previsão normalizada
        nota_prevista_normalizada = (np.dot(sims_validas, notas_validas) / denominador)

        # DESNORMALIZAÇÃO POR ITEM
        if self.normalizar:
            media_item = (self.medias_itens[f_idx])
            desvio_item = (self.desvios_itens[f_idx])
            nota_prevista = (media_item + (nota_prevista_normalizada * desvio_item))

        else:
            nota_prevista = (nota_prevista_normalizada)

        nota_prevista = np.clip(nota_prevista,1,5)

        return float(nota_prevista)


    def item_based(self, matriz_treino, user_id):

        if user_id not in self.user_to_idx:
            return "Usuário não encontrado"

        u_idx = self.user_to_idx[user_id]
        vetor_usuario = (self.matriz_original[u_idx])

        filmes_nao_assistidos = np.where(vetor_usuario == 0)[0]

        previsoes = []

        for filme_idx in filmes_nao_assistidos:

            filme_id = (matriz_treino.columns[filme_idx])

            nota_prevista = (self.prever_nota_item_based(matriz_treino, user_id, filme_id))

            if nota_prevista > 0:
                previsoes.append((filme_id, float(nota_prevista)))

        previsoes.sort(key=lambda x: x[1], reverse=True)

        return previsoes[:self.n_recomendacoes]


    def avaliar_rmse_item_based(self, matriz_treino, matriz_teste):

        y_pred = []
        y_real = []

        teste_values = matriz_teste.values
        usuarios_idx, filmes_idx = np.where(teste_values > 0)
        usuarios_ids = (matriz_teste.index.to_numpy())
        filmes_ids = (matriz_teste.columns.to_numpy())

        for u, f in zip(usuarios_idx, filmes_idx):

            user_id = usuarios_ids[u]
            filme_id = filmes_ids[f]
            nota_real = teste_values[u, f]
            nota_prevista = (self.prever_nota_item_based(matriz_treino, user_id, filme_id))

            if nota_prevista > 0:

                y_pred.append(nota_prevista)
                y_real.append(nota_real)

        y_pred = np.asarray(y_pred, dtype=np.float32)
        y_real = np.asarray(y_real, dtype=np.float32)

        return rmse(y_pred, y_real)
