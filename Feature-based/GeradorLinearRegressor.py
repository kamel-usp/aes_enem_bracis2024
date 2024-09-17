import os
import pandas as pd
from sklearn.linear_model import LinearRegression

def get_dados():
    path_atual = os.getcwd()
    path_dados = os.path.join(path_atual, "dataset")
    train = pd.read_csv(os.path.join(path_dados, "Treinamento_split.csv"), encoding='utf-8')
    validation = pd.read_csv(os.path.join(path_dados, "Validacao_split.csv"), encoding='utf-8')
    test = pd.read_csv(os.path.join(path_dados, "Teste_split.csv"), encoding='utf-8')
    dic = {}
    dic['treinamento'] = pd.concat([train, validation])
    dic['teste'] = test
    return dic

def selecionar_comp(comp):
    return lambda row: eval(row['grade'])[comp-1]

def get_dados_tratados(comp):
    dados = get_dados()
    dados['treinamento']['competencia'] = dados['treinamento'].apply(selecionar_comp(comp), axis=1)
    dados['treinamento'] = dados['treinamento'].drop(['grade', 'index_text'], axis=1)
    dados['teste']['competencia'] = dados['teste'].apply(selecionar_comp(comp), axis=1)
    dados['teste'] = dados['teste'].drop(['grade', 'index_text'], axis=1)
    return dados

def get_linear_regressor(comp):
    dados = get_dados()
    treinamento = dados['treinamento']
    #print(treinamento)
    treinamento['competencia'] = treinamento.apply(selecionar_comp(comp), axis=1)
    treinamento = treinamento.drop(['grade', 'index_text'], axis=1)
    model = LinearRegression()
    y = treinamento['competencia']
    x = treinamento.drop("competencia", axis=1).values
    model.fit(x,y)
    return model