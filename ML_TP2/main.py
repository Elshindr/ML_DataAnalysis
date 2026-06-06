import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

df=pd.read_csv('carprices.csv')
df.head()

#Suppression de la colonne
df = df.drop('Unnamed: 0', axis = 1)

#Passage de la colonne en indice
#df.set_index(['Unnamed: 0'], inplace=True)

df['cylindernumber'].value_counts()

df['doornumber'].value_counts()

# normalisation post-préparation
def replace_integer(x):
    if x == 'two':
        return 2
    if x == 'three':
        return 3
    if x == 'four':
        return 4
    if x == 'five':
        return 5
    if x == 'six':
        return 6
    if x == 'eight':
        return 8
    if x == 'twelve':
        return 12


df['doornumber'] = df['doornumber'].apply(replace_integer)
df['cylindernumber'] = df['cylindernumber'].apply(replace_integer)
feats = df.drop('price', axis=1)
target=df['price']

X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state=42)


# ONEHOTENCODER normalisation catégorie
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop="first", sparse=False)

cat= ['fueltype', 'aspiration']
X_train[cat] = ohe.fit_transform(X_train[cat])
X_test[cat] = ohe.transform(X_test[cat])


# Standarisation sur les numeriques
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

cols=['doornumber', 'cylindernumber', 'wheelbase', 'carlength','carwidth', 'carheight', 'curbweight']
X_train[cols] = scaler.fit_transform(X_train[cols])
X_test[cols] = scaler.fit(X_test)


# Regression lineaire
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

## coef et intercept_
coeffs = list(regressor.coef_)
coeffs.insert(0, regressor.intercept_)

feats2 = list(feats.columns)
feats2.insert(0, 'intercept')

pd.DataFrame({'valeur estimée': coeffs}, index=feats2)

#Affichage de la droite de regression et analyse
pred_test = lr.predict(X_test)
plt.scatter(pred_test, y_test)
plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), c="r")
plt.show()


