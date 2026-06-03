import pandas as pd

def dividir_treino_val_teste(dataset, perc_treino=0.8, perc_val=0.1):

    # embaralha o dataset
    df_embaralhado = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # índices de corte baseados no tamanho total do dataset
    tamanho_total = len(df_embaralhado)
    corte_treino = int(tamanho_total * perc_treino)
    corte_val = corte_treino + int(tamanho_total * perc_val)
    
    # fatiamento do dataframe
    treino = df_embaralhado.iloc[:corte_treino]
    validacao = df_embaralhado.iloc[corte_treino:corte_val]
    teste = df_embaralhado.iloc[corte_val:]
    
    return treino, validacao, teste

