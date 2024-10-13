import numpy as np
import pandas as pd

db = pd.read_csv('admission.csv', delimiter=';', encoding='latin1')

# indica para pegar todas as linhas e colunas (exceto a ultima coluna -1)
X = db.iloc[:,:-1].values

# o -1 indica que começa a pegar os dados a partir da ultima coluna
y = db.iloc[:, -1].values

# Serve para introduzir os dados 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')

# A função fit retira parametros da base de dados, parametros como média, desvio padrão etc...
# A função transform vai aplicar os parametros em transformações do conjunto de dados
# Indica a partir da 1ª coluna (ja que a primeira coluna só possui nomes [string])
imputer = imputer.fit_transform(X[:,1:])

# Muda strings para numeros, tranforma em rotulos
# PROBLEMA: Valores decimais afetam as decisões do machine learning
# Solução: utilizar valores binarios para condificar as strings
from sklearn.preprocessing import LabelEncoder
labelenconder_X = LabelEncoder()
X[:, 0] = labelenconder_X.fit_transform(X[:, 0])

# SOLUÇÂO:
X = X[:, 1:]
D = pd.get_dummies(X[:,0])
X = np.insert(X, 0, D.values, axis=1)


# Dividindo em conjunto de treino e de teste
from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)

print(XTrain)









