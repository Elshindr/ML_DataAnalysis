

import pandas as pd
df = pd.read_csv("known.csv")
df.head()

df.duplicated() # il n'y pas apparement pas de doublon dans le jeu de données

cols_na = df.columns[df.isna().sum() > 0]
nb_na = df[cols_na].isna().sum()
for i, v in nb_na.items():
    print("Taux de valeurs manquete pour "+ i, round(v*100/len(df), 2))
    
tab = df.drop('PassengerId', axis=1)

for c in tab.columns:
    print(tab[c].value_counts())
    
]:

df = df.drop(["Name", "PassengerId" , "Cabin"], axis = 1)
df.head()

from sklearn.model_selection import train_test_split

X = df.drop("Transported", axis=1)
y = df['Transported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = ['Age','RoomService', 'FoodCourt','ShoppingMall','Spa','VRDeck']
num_train = X_train[num_cols]
num_test = X_test[num_cols]

cat_cols = ['HomePlanet','CryoSleep','Destination', 'VIP']
cat_train = X_train[cat_cols]
cat_test = X_test[cat_cols]


from sklearn.impute import SimpleImputer

si_num = SimpleImputer(strategy='median')
X_train.loc[:, num_cols] = si_num.fit_transform(X_train[num_cols])
X_test.loc[:, num_cols] = si_num.transform(X_test[num_cols])


si_cat = SimpleImputer(strategy='most_frequent')
X_train.loc[:, cat_cols] = si_cat.fit_transform(X_train[cat_cols])
X_test.loc[:, cat_cols] = si_cat.transform(X_test[cat_cols])

print(X_test.loc[:, cat_cols].columns, len(X_test.loc[:, cat_cols].columns))
print(X_test[cat_cols].columns, len(X_test[cat_cols].columns))


from sklearn.preprocessing import OneHotEncoder


ohe = OneHotEncoder(drop='first', sparse_output=False)
X_train.loc[:, cat_cols] = ohe.fit_transform(X_train[cat_cols])
X_test.loc[:, cat_cols] = ohe.transform(X_test[cat_cols])

#print(X_test.loc[:, cat_cols].columns, len(X_test.loc[:, cat_cols].columns))
#print(X_test[cat_cols].columns, len(X_test[cat_cols].columns))

from sklearn.linar_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)

from skleanr.metrics import classification_report
y_pred_lr = lr.predict(X_test)
y_pred_dtc = dtc.predict(X_test)

print(pd.crosstab(y_test, y_pred_lr, rownames=['Realite'], colnames=['Prediction']))
print(classification_report(y_test, y_pred_lr))

print(pd.crosstab(y_test, y_pred_dtc, rownames=['Realite'], colnames=['Prediction']))
print(classification_report(y_test, y_pred_dtc))


## Entrainement complet
df_k= pd.read_csv("known.csv")
df_k = df_k.drop(["Name", "PassengerId" , "Cabin"], axis = 1)
df_k.head()

X = df_k.drop("Transported", axis=1)
y = df_k['Transported']


lr = LinearRegression()
lr.fit(X_train, y_train)



print(X_test.loc[:, cat_cols].columns, len(X_test.loc[:, cat_cols].columns))
print(X_test[cat_cols].columns, len(X_test[cat_cols].columns))

## Prediction missing
df_m = pd.read_csv("missing.csv")
df_m.head()

num_cols = ['Age','RoomService', 'FoodCourt','ShoppingMall','Spa','VRDeck']
num_train = X_train[num_cols]
num_test = X_test[num_cols]

cat_cols = ['HomePlanet','CryoSleep','Destination', 'VIP']
cat_train = X_train[cat_cols]
cat_test = X_test[cat_cols]

si_num = SimpleImputer(strategy='median')
X_train.loc[:, num_cols] = si_num.fit_transform(X_train[num_cols])
X_test.loc[:, num_cols] = si_num.transform(X_test[num_cols])


si_cat = SimpleImputer(strategy='most_frequent')
X_train.loc[:, cat_cols] = si_cat.fit_transform(X_train[cat_cols])
X_test.loc[:, cat_cols] = si_cat.transform(X_test[cat_cols])

ohe = OneHotEncoder(drop='first', sparse_output=False)
X_train.loc[:, cat_cols] = ohe.fit_transform(X_train[cat_cols])
X_test.loc[:, cat_cols] = ohe.transform(X_test[cat_cols])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=42)

lr.fit()
y_pred_ms = lr.predict(X_test)




from sklearn.pipeline import Pipeline,ColumnTransformer

num_pipe = ColumnTransformer(steps=[('imputer', SimpleImputer(strategy='median'))])
                    
cat_pipe = ColumnTransformer([('imputer', SimpleImputer(strategy='most_frequent'))])
                    
 = ColumnTransformer( transformer=[('num'= num_trans)])
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', LogisticRegression())
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=42)
pipe.fit(X_train, y_train)
print("score du model:", pipe.score(X_test, y_test))

 