import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv("Data/Ecommerce Customers")

#print(customers.head())
#print(customers.describe())
print(customers.info())

sns.set_palette("GnBu_d")
sns.set_style("whitegrid")

#sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)
#sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers)
#sns.jointplot(x='Time on App', y='Length of Membership', data=customers, kind='hex')
#sns.pairplot(customers)
#sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)

y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print('Coefficient is : \n ', lm.coef_)
prediction = lm.predict(X_test)

# plt.scatter(y_test,prediction)
# plt.xlabel('Y Test')
# plt.ylabel('Predicted Y')


from sklearn import metrics
print('MAE: ', metrics.mean_absolute_error(y_test,prediction))
print('MSE: ', metrics.mean_squared_error(y_test,prediction))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test,prediction)))

#sns.distplot((y_test-prediction),bins=30)

#plt.show()


coefficient = pd.DataFrame(lm.coef_,X.columns)
coefficient.columns = ['Coefficient']
print(coefficient)