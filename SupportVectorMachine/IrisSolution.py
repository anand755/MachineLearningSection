import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import Image,display

# setosaUrl = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
# setosaImg = Image(setosaUrl,width=300,height=300)
# display(setosaImg)
#
# versicolorUrl = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
# versicolorImg = Image(versicolorUrl,width=300,height=300)
# display(versicolorImg)
#
#
# virginicaUrl = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
# virginicaImg = Image(virginicaUrl,width=300, height=300)
# display(virginicaImg)



iris = sns.load_dataset('iris')
print(iris.info())

#sns.pairplot(iris, hue='species', palette='Dark2')


setosa = iris[iris['species']=='setosa']
#sns.kdeplot(setosa['sepal_width'], setosa['sepal_length'], cmap = 'plasma', shade=True, shade_lowest=False)

from sklearn.model_selection import train_test_split
X = iris.drop('species', axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train,y_train)
prediction = svc_model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))


#Grid  search practice

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)


grid_prediction = grid.predict(X_test)
print(confusion_matrix(y_test, grid_prediction))
print(classification_report(y_test, grid_prediction))











#plt.show()









