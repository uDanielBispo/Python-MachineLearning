import numpy as np
import pandas as pd

# Lê o arquivo csv separado por ";" e informa o enconding para que ele entenda os caracteres
db = pd.read_csv('svbr.csv', delimiter=';', encoding='latin1')

# selecionando todos os dados da planilha e armazenando em X somente os valores
X = db.iloc[:,:].values

from sklearn.impute import SimpleImputer

# indicando os dados faltantes
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')

imputer = imputer.fit(X[:,1:3])
X = imputer.transform(X[:,1:3]).astype(str)
X = np.insert(X, 0, db.iloc[:,0].values, axis=1)
print(X)