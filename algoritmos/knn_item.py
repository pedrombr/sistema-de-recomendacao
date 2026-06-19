import pandas as pd
import numpy as np

from metricas import (
    similaridade_cosseno,
    cosseno_ajustado,
    correlacao_pearson,
    rmse
)


class KNNItemBased:

    # lista de k para testar no KNN
    valores_k_para_testar = [5, 10, 15, 20]

    def __init__(
        self,
        k_vizinhos=5,
        n_recomendacoes=5,
        normalizar=True,
        metrica="cosseno_ajustado"
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

        vetor_normalizado[notas_validas] = (
            (vetor_notas[notas_validas] - media)
            / desvio_padrao
        )

        return vetor_normalizado


    def fit(self, matriz_treino, medias_treino):

        self.matriz_treino = matriz_treino
        self.medias_treino = medias_treino

        filmes = matriz_treino.columns
        n_filmes = len(filmes)

        # matriz numpy para acelerar acesso
        matriz_np = matriz_treino.values.astype(float)

        # matriz final de similaridade
        self.matriz_similaridade = pd.DataFrame(
            np.zeros((n_filmes, n_filmes)),
            index=filmes,
            columns=filmes
        )

        for i in range(n_filmes):

            filme_i = filmes[i]

            vetor_i = matriz_np[:, i]

            if self.normalizar:
                vetor_i = self.normalizacao_z_score(vetor_i)

            for j in range(i, n_filmes):

                filme_j = filmes[j]

                # diagonal principal
                if i == j:

                    self.matriz_similaridade.iat[i, j] = 1.0
                    continue

                vetor_j = matriz_np[:, j]

                if self.normalizar:
                    vetor_j = self.normalizacao_z_score(vetor_j)

                sim = 0.0

                # máscara vetorizada
                correlacionados = (
                    (vetor_i > 0)
                    & (vetor_j > 0)
                )

                if np.sum(correlacionados) < 2:
                    continue

                x = vetor_i[correlacionados]
                y = vetor_j[correlacionados]

                if self.metrica == 'cosseno_ajustado':

                    medias = medias_treino[correlacionados]

                    x = x - medias
                    y = y - medias

                    numerador = np.dot(x, y)

                    denominador = (
                        np.linalg.norm(x)
                        * np.linalg.norm(y)
                    )

                    if denominador > 0:
                        sim = numerador / denominador

                elif self.metrica == 'pearson':

                    x_media = np.mean(x)
                    y_media = np.mean(y)

                    x_centralizado = x - x_media
                    y_centralizado = y - y_media

                    numerador = np.dot(
                        x_centralizado,
                        y_centralizado
                    )

                    denominador = (
                        np.linalg.norm(x_centralizado)
                        * np.linalg.norm(y_centralizado)
                    )

                    if denominador > 0:
                        sim = numerador / denominador

                elif self.metrica == 'cosseno':

                    numerador = np.dot(x, y)

                    denominador = (
                        np.linalg.norm(x)
                        * np.linalg.norm(y)
                    )

                    if denominador > 0:
                        sim = numerador / denominador

                # preenche os dois lados da matriz
                self.matriz_similaridade.iat[i, j] = sim
                self.matriz_similaridade.iat[j, i] = sim

        return self


    def knn_item_based(
        self,
        matriz_treino,
        medias_treino,
        user_id
    ):

        if user_id not in matriz_treino.index:
            return "Usuário não encontrado na base de treino"

        vetor_usuario = matriz_treino.loc[user_id].values

        filmes_assistidos = (
            matriz_treino.columns[vetor_usuario > 0]
        )

        filmes_nao_assistidos = (
            matriz_treino.columns[vetor_usuario == 0]
        )

        if len(filmes_assistidos) == 0:

            return (
                "O usuário não avaliou nenhum filme "
                "para basearmos a recomendação."
            )

        previsoes = []

        for filme_alvo in filmes_nao_assistidos:

            similaridades = (
                self.matriz_similaridade.loc[
                    filme_alvo,
                    filmes_assistidos
                ]
                .fillna(0)
            )

            similaridades = (
                similaridades[similaridades > 0]
            )

            if similaridades.empty:
                continue

            top_k_similares = similaridades.nlargest(
                self.k_vizinhos
            )

            notas_do_usuario = np.array([

                matriz_treino.at[user_id, filme]

                for filme in top_k_similares.index
            ])

            sims_validas = top_k_similares.values

            numerador = np.dot(
                sims_validas,
                notas_do_usuario
            )

            denominador = np.sum(
                np.abs(sims_validas)
            )

            if denominador > 0:

                nota_prevista = (
                    numerador / denominador
                )

                previsoes.append(
                    (filme_alvo, nota_prevista)
                )

        previsoes.sort(
            key=lambda x: x[1],
            reverse=True
        )

        top_n_filmes = (
            previsoes[:self.n_recomendacoes]
        )

        return top_n_filmes


    def prever_nota_item_based(
        self,
        matriz_treino,
        medias_treino,
        user_id,
        filme_id
    ):

        if (
            filme_id not in matriz_treino.columns
            or user_id not in matriz_treino.index
        ):
            return 0

        # filmes já assistidos
        notas_do_usuario = matriz_treino.loc[user_id]

        filmes_assistidos = (
            notas_do_usuario[
                notas_do_usuario > 0
            ].index
        )

        if len(filmes_assistidos) == 0:
            return 0

        similaridades = (
            self.matriz_similaridade.loc[
                filme_id,
                filmes_assistidos
            ]
            .fillna(0)
        )

        similaridades = (
            similaridades[similaridades > 0]
        )

        if similaridades.empty:
            return 0

        # top K filmes similares
        top_k_similares = similaridades.nlargest(
            self.k_vizinhos
        )

        soma_pesos = top_k_similares.sum()

        soma_notas_pesadas = sum(

            top_k_similares[filme]
            * matriz_treino.at[user_id, filme]

            for filme in top_k_similares.index
        )

        nota_prevista = (
            soma_notas_pesadas / soma_pesos
        )

        return nota_prevista


    def avaliar_rmse_item_based(
        self,
        matriz_treino,
        matriz_teste,
        medias_treino
    ):

        y_pred = []
        y_real = []

        # percorre apenas avaliações reais
        usuarios, filmes = np.where(
            matriz_teste.values > 0
        )

        for u, f in zip(usuarios, filmes):

            user_id = matriz_teste.index[u]
            filme_id = matriz_teste.columns[f]

            nota_real = matriz_teste.at[
                user_id,
                filme_id
            ]

            nota_prevista = (
                self.prever_nota_item_based(
                    matriz_treino,
                    medias_treino,
                    user_id,
                    filme_id
                )
            )

            if nota_prevista > 0:

                y_pred.append(nota_prevista)
                y_real.append(nota_real)

        y_pred = np.array(y_pred)
        y_real = np.array(y_real)

        erro_rmse = rmse(
            y_pred,
            y_real
        )

        return erro_rmse
