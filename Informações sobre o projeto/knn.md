* Notação e Definições
* Usuários: $u \in {1, \dots, U}$
* Itens: $i \in {1, \dots, I}$
* Conjunto de treinamento $T$, contendo preferências observadas $r_{ui}$ para algumas combinações de usuário-item $(u, i)$
* $r_{ui}$: preferência observada (ex.: compra, avaliação, número de cliques, etc.)
* $R$: quantidade total de avaliações
* $U$: quantidade total de usuários
* $I$: quantidade total de itens
* $\mu$: média global das avaliações
* $U_i$: conjunto de usuários que avaliaram o item $i$
* $I_u$: conjunto de itens avaliados pelo usuário $u$
* $U_{ij} = U_i \cap U_j$: usuários que avaliaram os itens $i$ e $j$
* $I_{uv} = I_u \cap I_v$: itens avaliados pelos usuários $u$ e $v$
* $N_i(u)$: conjunto de vizinhos do usuário $u$ que avaliaram o item $i$

Predição com KNN

A predição da avaliação do usuário $u$ para o item $i$ é dada por:

$$\tilde{r}_{ui} = \frac{\sum_{v \in N_i(u)} w_{uv} \cdot r_{vi}}{\sum_{v \in N_i(u)} |w_{uv}|}$$

* $w_{uv}$: similaridade entre os usuários $u$ e $v$
* $N_i(u)$: conjunto de vizinhos de $u$ que avaliaram o item $i$
