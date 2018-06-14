import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ad_data = pd.read_csv("Data/advertising.csv")
print(ad_data.info())
#print(ad_data.head())
#print(ad_data.describe())

# sns.set_style('whitegrid')
# ad_data['Age'].hist(bins=30)
# plt.xlabel('Age')


#sns.jointplot(x='Age', y='Area Income', data=ad_data)
#sns.jointplot(x='Age', y='Daily Time Spent on Site', kind='kde', color='red', data=ad_data)

#sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ad_data, color='green')
#sns.pairplot(data=ad_data,hue='Clicked on Ad', palette='bwr')

from sklearn.model_selection import train_test_split
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=42)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

prediction = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))












plt.show()

