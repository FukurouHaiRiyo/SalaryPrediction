import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns


def line(x, a, b):
    return x * a + b

df = pd.read_csv('Cursuri Machine Learning/Curs2/PrezicereSalariu/Salary.csv', delimiter=',')
df.info()

X = df['YearsExperience'].values
X = X.reshape(-1, 1)

y = df['Salary'].values

# y_mean = y.mean()
# y_std = y.std()
#
# y = (y - y_mean) / y_std

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# X_train, X_test = X[:2], X[2:]
# y_train, y_test = y[:2], y[2:]
print(X_train.shape)
plt.scatter(X_train, y_train, c='orange')
plt.scatter(X_test, y_test)
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
#plt.show()

correlation_matrix = df.corr()
print(correlation_matrix)
#sns.heatmap(correlation_matrix, annot=True)
#plt.show()

model = LinearRegression()
# ridge_model = Ridge(alpha=0.1)
#
model.fit(X_train, y_train)
# ridge_model.fit(X_train, y_train)
#
y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)
#
# y_pred_train_r = ridge_model.predict(X_train)
# y_pred_r = ridge_model.predict(X_test)

print("R square for train {}".format(model.score(X_train, y_train)))
# print("MSE for train: {}".format(mean_squared_error(y_train, y_pred_train)))
print("R squared for test: {}".format(model.score(X_test, y_test)))
# print("MSE for test: {}".format(mean_squared_error(y_test, y_pred)))
#
# # print("R square for train Ridge {}".format(ridge_model.score(X_train, y_train)))
# # print("MSE for train Ridge: {}".format(mean_squared_error(y_train, y_pred_train_r)))
# # print("R squared for test Ridge: {}".format(ridge_model.score(X_test, y_test)))
# # print("MSE for test Ridge: {}".format(mean_squared_error(y_test, y_pred_r)))
#
a = model.coef_[0]
b = model.intercept_
#
# a_ridge = ridge_model.coef_[0]
# b_ridge = ridge_model.intercept_
#
points = np.array([X.min(), X.max()])

plt.plot(points, line(points, a, b), c='red')
#plt.plot(points, line(points, a_ridge, b_ridge))
plt.show()
