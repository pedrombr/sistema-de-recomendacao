import numpy as np
from metricas import rmse

class SVD:
    valores_reg_para_testar = [0.01, 0.02, 0.05, 0.1]

    def __init__(self, k_factors=20, learning_rate=0.01, reg=0.02, epochs=20, random_state=1):
        self.k_factors = k_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = epochs
        self.random_state = random_state


    def train(self, rating):
        np.random.seed(self.random_state)

        n_users = rating["UserID"].max()
        n_itens = rating["MovieID"].max()

        self.P = np.random.normal(0, 0.1, (n_users, self.k_factors))
        self.Q = np.random.normal(0, 0.1, (n_itens, self.k_factors))
        self.media_global = rating["Rating"].mean()
        self.u_bias = np.zeros(n_users)
        self.i_bias = np.zeros(n_itens)

        for epoch in range(self.n_epochs):

            y_pred = []
            y_real = []

            for _, row in rating.iterrows():
                u = row["UserID"] - 1
                i = row["MovieID"] - 1
                r = row["Rating"]

                predict = self.media_global + self.u_bias[u] + self.i_bias[i] + np.dot(self.P[u], self.Q[i])

                error = r - predict

                y_pred.append(predict)
                y_real.append(r)

                self.u_bias[u] += (self.learning_rate * (error - self.reg * self.u_bias[u]))
                self.i_bias[i] += (self.learning_rate * (error - self.reg * self.i_bias[i]))

                p_old = self.P[u].copy()

                self.P[u] += (self.learning_rate * (error * self.Q[i] - self.reg * self.P[u]))
                self.Q[i] += (self.learning_rate * (error * p_old - self.reg * self.Q[i]))

            epoch_rmse = rmse(np.array(y_pred), np.array(y_real))
            print(f"Epoch {epoch+1}/{self.n_epochs}")
            print(f"RMSE={epoch_rmse:.4f}")
        return self

    def predict(self, user_id, item_id):
        u = user_id - 1
        i = item_id - 1

        return (self.media_global + self.u_bias[u] + self.i_bias[i] + np.dot(self.P[u], self.Q[i]))

    def avaliar_rmse_svd(self, base_df):
        lista_prev = []
        lista_real = []

        for _, linha in base_df.iterrows():
            user_id = int(linha['UserID'])
            filme_id = int(linha['MovieID'])
            nota_real = linha['Rating']

            try:
                prev = self.predict(user_id, filme_id)
                lista_prev.append(prev)
                lista_real.append(nota_real)
            except IndexError:
                pass

        return rmse(np.array(lista_prev), np.array(lista_real))
