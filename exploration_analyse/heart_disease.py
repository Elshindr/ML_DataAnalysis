import pandas as pd

df = pd.read_csv("heart_disease.csv", sep=";", index_col=0)


df.describe()
df.head()


age = df[df["age"] == 37]
age.head()
# les deux individus de 37 ans sont apparement cardiaques

max = df['age'].idxmax()
print(df.loc[max])
# il ne semble pas cardiaque

pd.crosstab(df["sex"], df["target"])



print(df.groupby('target')['age'].mean())



df['sex'] = df['sex'].replace({'Male':0, 'Female':1})
df.head()

def corr_data(data):
    if data < 50.0 or data > 250.0 :
        data = df['thalach'].median()
    return data

df['thalach'] = df['thalach'].apply(corr_data)
df.head()
:

df.isna().sum()



df = df.dropna(axis=0, how='all', subset=['target'])
df.isna().sum()

:

df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
df['exang'] = df['exang'].fillna(df['exang'].mode()[0])

df.isna().sum()

df['trestbps'] = df['trestbps'].fillna(df['trestbps'].median())
df['chol'] = df['chol'].fillna(df['chol'].median())
df['thalach'] = df['thalach'].fillna(df['thalach'].median())

df.isna().sum()

X = df.drop(['target'], axis = 1)
y = df['target']

X_norm = pd.DataFrame(data = X, columns=X.columns)

for col_i in range(0, len(X.columns)):
    print(col_i, X.columns[col_i])
    
    for lig_i in range(len(X)):
        x_new = 2* ((lig_i - X[col_i].min()) / (X[col_i].max() - X[col_i].min())) - 1
        X_norm.loc[lig_i, col_i] = x_new
        
X_norm.head()

df = pd.read_csv("heart_disease.csv", index_col=0, sep=";")
df.head()

]:

def arg_is_df(df):
    def arg_is_decorator(function):
        def arg_is_fn(*args, **kwargs):
            if isinstance(df, pd.DataFrame): 
                print(test)
                return function(*args, **kwargs)
            else:
                return "Le parametre doit etre du type DataFrame"
        return arg_is_fn
    return arg_is_decorator

@arg_is_df
def preprocess_data(dfc: pd.DataFrame) -> pd.DataFrame :
    
    dfc['sex'] = dfc['sex'].replace({'Male':0, 'Female':1})
    dfc['thalach'] = dfc['thalach'].apply(corr_data)
    dfc = dfc.dropna(axis=0, how='all', subset=['target'])
    
    dfc['ca'] = dfc['ca'].fillna(dfc['ca'].mode()[0])
    dfc['exang'] = dfc['exang'].fillna(dfc['exang'].mode()[0])

    
    dfc['trestbps'] = dfc['trestbps'].fillna(dfc['trestbps'].median())
    dfc['chol'] = dfc['chol'].fillna(df['chol'].median())
    dfc['thalach'] = dfc['thalach'].fillna(dfc['thalach'].median())
    return dfc
    
df = preprocess_data(df)
df.head()

