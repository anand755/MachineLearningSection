import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

loans = pd.read_csv('Data/loan_data.csv')
#print(loans.info())

#print(loans.describe())
#print(loans.head())

# plt.figure(figsize=(10,6))
# loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5, color='blue', bins=30, label='Credit.Policy = 1')
# loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5, color='red', bins=30, label='Credit.Policy = 0')
# plt.legend()
# plt.xlabel('FICO')






# plt.figure(figsize=(10,6))
# loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5, color='blue', bins=30, label='not.fully.paid = 1')
# loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5, color='red', bins=30, label='not.fully.paid = 0')
# plt.legend()
# plt.xlabel('FICO')



# plt.figure(figsize=(11,7))
# sns.countplot(x='purpose', hue='not.fully.paid', data=loans, palette='Set1')


#sns.jointplot(x='fico', y='int.rate', data=loans, color='purple')


# plt.figure(figsize=(11,7))
# sns.lmplot(y='int.rate', x='fico', data=loans, hue='credit.policy', col='not.fully.paid', palette='Set1')



cat_feat = ['purpose']

final_data = pd.get_dummies(data=loans,columns=cat_feat, drop_first=True)
#print(final_data.info())


from sklearn.model_selection import train_test_split

X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)



from sklearn.tree import  DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
prediction =  dtree.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))



from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train, y_train)
predictionrfc = rfc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictionrfc))
print(confusion_matrix(y_test,predictionrfc))









plt.show()










