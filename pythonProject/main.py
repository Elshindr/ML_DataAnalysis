# Scikit-Learn
# Librairie Python construite à partir de NumPy, Scipy et matplotlib
# = entrainer des modèles de Machine Learning

# Instanciation du modèle choisi
# Pour créer un modèle, on génère un objet de la classe correspondante à cet objet
# LinearRegression est un algorithme de régression
# On peut spécifier certains paramètres de l'algorithme entre les parenthèses



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#t = np.arange(0, 5, 0.2)
#t2 = [i**2 for i in t]
#t3 = [i**3 for i in t]
#plt.plot(t, t, ':r', t, t2,'-g', t,t3, '--b')
#plt.plot(t,t,t,t**2,t,t**3)
#plt.plot(t,t,'hy')
#plt.plot(t,t**2,'g-', linewidth=5)
#plt.plot(t,t**3,'b', marker='D')

#plt.ylim(0, 50)

#plt.grid(True)
#plt.plot([50,100,150,200], [2,3,7,10], "b-*", linewidth=0.8, label="Trajet 1")
#plt.plot([50,100,150,200], [2,7,9,10], "g-+", linewidth=0.8, label="Trajet 2")
#plt.xlabel('Vitesse')
#plt.ylabel('Temps')
#plt.legend();



# BAR
"""
plt.bar(range(4), [2, 3, 4, 5] , color = 'y', width = 0.6)

plt.bar([1,3,5,7,9],[5,2,7,8,2], color='b', label = 'Exemple 1');
plt.bar([2,4,6,8,10],[8,6,2,5,6], color='g', label = 'Exemple 2', bottom=[5,2,7,8,2]);

plt.xlabel('Nombre')
plt.ylabel('Hauteur')
plt.title('Mon graphique en barres');
plt.legend()




barWidth = 0.4
x1 = range(12)
x2 = [i+barWidth for i in x1]

df12 = df.head(12)

plt.bar(x1, df12.Product1, width= barWidth, label = "Produit1")
plt.bar(x2, df12.Product2, width= barWidth, label = "Produit2")
plt.xticks([0, 2, 4,6, 8, 11],['Janvier', 'Mars', 'Mai', 'Juillet', 'Septembre', 'Decembre'])
plt.legend()

df.head(6).plot.bar(x = 'Month', y=['Product1', 'Product2', 'Returns'],stacked=True,
                     rot=0)
plt.xticks(range(6), ["Janvier","Février","Mars","Avril","Mai","Juin"])
plt.legend();"""



# histogramme
#plt.scatter(range(0,7), [8,7,6,5,6,7,8], color='red', marker= '*', s=40)
#plt.hist(np.random.choice(11, 40), bins=7, color = 'orange', rwidth=0.8, density = True,orientation = 'horizontal')

x= [1, 2, 2, 3, 4, 4, 4, 4, 4, 5, 5]
plt.hist(x, range= (1, 6), bins=5, rwidth= 0.6, color = '#EE3459', density=True, orientation = 'horizontal')

plt.xlabel('Probabilité')
plt.ylabel('Valeurs')
plt.title('Histogramme orizontale')



"""df = pd.read_csv('sales_data.csv')
plt.hist([df.Product1, df.Product2], bins = 6, color = ['#f27750', '#f7bf59'], label = ['Product1', 'Product2'])
plt.xlabel('Ventes')
plt.ylabel('Effectifs')
plt.title('Histogrammme 2 series')
plt.legend();


df = pd.read_csv('sales_data.csv')
plt.hist([df.Product1, df.Product2], histtype='barstacked', bins = [100,200,280,325,450,600,800], rwidth = 0.8, color = ['#0086ad', '#97ebdb'], label = ['Product1', 'Product2'])
plt.xlabel('Ventes')
plt.ylabel('Effectifs')
plt.title('2 series superposées')
plt.legend();

#avec pandas et un dataFrame
df.plot.hist(y=['Product1', 'Product2'], bins = 7, rwidth = 0.8 , color= ['#0c4c83', '#830c4c'], alpha=0.5);
df.plot.hist(y=['Product1', 'Product2'], bins = 7, subplots=True, rwidth = 0.8 , color= ['#0c4c83', '#830c4c'], alpha=0.5);
df.hist(column=['Product1', 'Product2'], bins = 7, rwidth = 0.8 , color= ['#0c4c83']);
"""

# boite a moustaches
"""plt.show();
df['Mois']= df.Month.apply(lambda x : x[3:])

l=list()
for i in df.Mois.unique():
    l.append(df[df['Mois'] == i]['Turnover'])
plt.boxplot(l)
plt.xticks(range(1,13),df.Mois.unique())
#sous panda
df.boxplot(column= 'Turnover', by='Mois', figsize= (7,7));
"""


#pie chart
plt.figure(figsize=(6,6))
x=[1, 2, 3, 4, 10]
plt.pie(x, labels=['A', 'B', 'C', 'D', 'E'])
plt.legend()

plt.show()

plt.figure(figsize=(7,7))

plt.pie(x=df.head(6).Turnover, labels= ['Janv', 'Fev', 'Mars', 'Avril', 'Mai', 'Juin'],
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple'],
       explode=[0, 0, 0,0.2, 0, 0],
       autopct=lambda x: str(round(x, 2)) + '%',
       pctdistance=0.7,
       labeldistance=1.2,
       shadow=True)
plt.legend();



#composition de graph
"""df = pd.read_csv( 'sales_data.csv')

plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plt.bar(range(len(df.Product1)),df.Product1, label="bar")
plt.legend()

plt.subplot(2,2,2)
plt.scatter(df.Product1, df.Product2, label='nuage')
plt.legend()


plt.subplot(2,2,3)
plt.plot(df.Returns, label = "Returns")
plt.legend()

plt.subplot(2,2,4)
plt.hist(df.Turnover, label = "Turnover")
plt.legend();


# Panda permet directement de créer plus graph en 1 fois
df.plot(y = ['Product1', 'Product2', 'Returns', 'Turnover'], subplots=True, layout= (2,2),
        style = ['b--', 'm:p', 'g-.', 'c-d'], figsize=(7,7));"""
plt.figure(figsize=(7,7))
plt.boxplot(df.Turnover)

plt.axes([0.65, 0.65, 0.2, 0.15], facecolor='#ffe5c1')
plt.hist(df.Turnover, color="#FFC575")
plt.xticks([])
plt.yticks([])
plt.xlabel("Distribution")


#texte et annotation
x = np.arange(0.0, 2.0, 0.01)
plt.plot(x, np.cos(2*np.pi*x) )
plt.ylim(-2,2)
plt.text(0.8, 1.1, 'maximun local')
plt.show()


x = np.arange(0, 5, 0.1)
plt.plot(x, np.sin(2*np.pi*x) * np.exp(-x), '-m' , x, np.exp(-x), "y", x, -np.exp(-x), 'r')
plt.grid(True)


plt.annotate('Asymptote: exp(-x)', xytext=(2, 0.5), xy=(1.5, 0.25), arrowprops={'facecolor':'green'})
plt.annotate('Asymptote: - exp(-x)', xytext=(2, -0.5), xy=(1.5, -0.25), arrowprops={"facecolor": 'red'})