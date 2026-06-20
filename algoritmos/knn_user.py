import pandas as pd
import numpy as np
from metricas import (similaridade_cosseno, correlacao_pearson, rmse)


class KNNUserBased:

    valores_k_para_testar = [10, 15, 20, 25, 27, 30, 35, 40]

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
        vetor_normalizado = vetor_notas.copy().astype(float)

        if desvio_padrao == 0:
            vetor_normalizado[notas_validas] = 0.0
            return vetor_notas

        vetor_normalizado[notas_validas] = ((vetor_notas[notas_validas] - media)/desvio_padrao)

        return vetor_normalizado


    def fit(self, matriz_treino):

        self.matriz_treino = matriz_treino
        #self.matriz_np = matriz_treino.values.astype(np.float32)
        self.matriz_original = (matriz_treino.values.astype(np.float32))
        self.matriz_np = (self.matriz_original.copy())
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(matriz_treino.index)}
        self.item_to_idx = {filme_id: idx for idx, filme_id in enumerate(matriz_treino.columns)}
        n_users = self.matriz_np.shape[0]


        self.medias_usuarios = np.zeros(n_users,dtype=np.float32)
        self.desvios_usuarios = np.ones(n_users,dtype=np.float32)

        #normalização dos dados
        if self.normalizar:

            for i in range(n_users):
                vetor_original = (self.matriz_np[i])

                notas_validas = (vetor_original > 0)

                if np.sum(notas_validas) == 0:
                    continue

                media = np.mean(vetor_original[notas_validas])

                desvio = np.std(vetor_original[notas_validas])

                if desvio == 0:
                    desvio = 1.0

                self.medias_usuarios[i] = media
                self.desvios_usuarios[i] = desvio

                self.matriz_np[i] = (self.normalizacao_z_score(vetor_original))

        if self.metrica == 'pearson':
            self.matriz_similaridade = (pd.DataFrame(
                            self.matriz_np,
                            index=matriz_treino.index,
                            columns=matriz_treino.columns
                        ).T.corr(method='pearson').fillna(0).values.astype(np.float32))
        elif self.metrica == 'pearson_unha':

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

        # usuário ORIGINAL
        vetor_usuario_original = (self.matriz_original[u_idx])

        # filmes não assistidos
        filmes_nao_assistidos = np.where(vetor_usuario_original == 0)[0]

        # similaridades do usuário
        similaridades_usuario = (self.matriz_similaridade[u_idx]).copy()

        # remove próprio usuário
        similaridades_usuario[u_idx] = 0

        # apenas positivas
        mascara_positiva = (similaridades_usuario > 0)

        if not np.any(mascara_positiva):
            return "Nenhum vizinho com gosto similar encontrado."

        previsoes = []

        for filme_idx in filmes_nao_assistidos:

            # notas NORMALIZADAS
            notas_filme_normalizadas = (self.matriz_np[:, filme_idx])

            # quem avaliou de verdade
            mascara_avaliaram = ((self.matriz_original[:, filme_idx] > 0) & mascara_positiva)

            if not np.any(mascara_avaliaram):
                continue

            sims_validas = (similaridades_usuario[mascara_avaliaram])

            notas_validas = (notas_filme_normalizadas[mascara_avaliaram])

            # TOP K
            if len(sims_validas) > self.k_vizinhos:

                top_k_idx = np.argpartition(sims_validas,-self.k_vizinhos)[-self.k_vizinhos:]

                sims_validas = (sims_validas[top_k_idx])

                notas_validas = (notas_validas[top_k_idx])

            denominador = np.sum(np.abs(sims_validas))

            if denominador == 0:
                continue

            # previsão no espaço normalizado
            nota_prevista_normalizada = (np.dot(sims_validas,notas_validas) / denominador)

            # DESNORMALIZAÇÃO
            if self.normalizar:

                media_usuario = (self.medias_usuarios[u_idx])

                desvio_usuario = (self.desvios_usuarios[u_idx])

                nota_prevista = (media_usuario + (nota_prevista_normalizada * desvio_usuario))

            else:
                nota_prevista = (nota_prevista_normalizada)
                #clipping opcional
            nota_prevista = np.clip(nota_prevista,1,5)
            filme_id = (matriz_treino.columns[filme_idx])

            previsoes.append((filme_id,float(nota_prevista)))

        previsoes.sort(key=lambda x: x[1],reverse=True)

        return previsoes[:self.n_recomendacoes]


    def prever_nota_user_based(self,matriz_treino,user_id,filme_id):

        # verifica existência
        if (user_id not in self.user_to_idx or filme_id not in self.item_to_idx):
            return 0

        u_idx = self.user_to_idx[user_id]
        f_idx = self.item_to_idx[filme_id]

        # notas NORMALIZADAS
        notas_filme_normalizadas = (self.matriz_np[:, f_idx])

        # usuários que avaliaram
        usuarios_validos = (self.matriz_original[:, f_idx] > 0)

        if not np.any(usuarios_validos):
            return 0

        # similaridades
        similaridades = (self.matriz_similaridade[u_idx])

        # remove próprio usuário
        usuarios_validos[u_idx] = False

        # máscara final
        mascara = (usuarios_validos & (similaridades > 0))

        if not np.any(mascara):
            return 0

        sims_validas = (similaridades[mascara])

        notas_validas = (notas_filme_normalizadas[mascara])

        # TOP K
        if len(sims_validas) > self.k_vizinhos:
            top_k_idx = np.argpartition(sims_validas,-self.k_vizinhos)[-self.k_vizinhos:]
            sims_validas = (sims_validas[top_k_idx])
            notas_validas = (notas_validas[top_k_idx])

        denominador = np.sum(np.abs(sims_validas))

        if denominador == 0:
            return 0

        # previsão normalizada
        nota_prevista_normalizada = (np.dot(sims_validas,notas_validas) / denominador)

        # DESNORMALIZAÇÃO
        if self.normalizar:
            media_usuario = (self.medias_usuarios[u_idx])
            desvio_usuario = (self.desvios_usuarios[u_idx])
            nota_prevista = (media_usuario + (nota_prevista_normalizada * desvio_usuario))

        else:
            nota_prevista = (nota_prevista_normalizada)

        # clipping opcional
        nota_prevista = np.clip(nota_prevista,1,5)

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

            nota_prevista = (self.prever_nota_user_based(matriz_treino,user_id,filme_id))

            if nota_prevista > 0:

                y_pred.append(nota_prevista)
                y_real.append(nota_real)

        y_pred = np.asarray(y_pred, dtype=np.float32)
        y_real = np.asarray(y_real, dtype=np.float32)

        return rmse(y_pred, y_real)
