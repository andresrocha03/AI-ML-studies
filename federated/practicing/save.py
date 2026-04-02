import pickle
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np

with open('data/processed/loan_status.pkl','rb') as f:
    x_treino,y_treino,x_teste,y_teste = pickle.load(f)


with open('data/processed/x_treino.parquet','rb') as f:
    x_treino = pd.read_parquet(f)
with open('data/processed/y_treino.parquet','rb') as f:
    y_treino = pd.read_parquet(f)
with open('data/processed/x_teste.parquet','rb') as f:
    x_teste = pd.read_parquet(f)
with open('data/processed/x_teste.parquet','rb') as f:
    y_teste = pd.read_parquet(f)

"""
df = pd.DataFrame(x_treino)
df.to_parquet("x_treino.parquet")
df = pd.DataFrame(x_teste)
df.to_parquet("x_teste.parquet")
df = pd.DataFrame(y_treino)
df.to_parquet("y_treino.parquet")
df = pd.DataFrame(y_teste)
df.to_parquet("y_teste.parquet")
"""

"""
x_treino = pd.DataFrame(x_treino)
y_treino = pd.DataFrame(y_treino)
x_teste = pd.DataFrame(x_teste)
y_teste = pd.DataFrame(y_teste)
"""

model = tree.DecisionTreeClassifier()
model.fit(x_treino,y_treino)



print(accuracy_score(y_teste,model.predict(x_teste)))
