import pandas as pd
import numpy as np
from importdataset import carregar_dataset_rating

dataset_notas = carregar_dataset_rating()
matriz_notas = np.array(dataset_notas)

def knn_user_based():
    notas = matriz_notas[:, 2]
    usuarios = matriz_notas[:, 0]

    mascara_notas = (notas > 0)

    usuarios_com_notas = usuarios[mascara_notas]

    return len(usuarios_com_notas)
