import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r'C:\Users\Neha\Downloads\AI\1.POLYNOMIAL REGRESSION\emp_sal.csv')


x=dataset.iloc[:, 1:2].values
y=dataset.iloc[:,2].values

# SVR Model
from sklearn.svm import SVR
regressor = SVR(kernel = "poly", degree = 4, gamma = 'auto' , C = 5.0)
regressor.fit(x,y)

y_pred_svr = regressor.predict([[6.5]])
print(y_pred_svr)


# KNN Model
from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(n_neighbors=4,weights='distance', algorithm='brute')
knn_reg.fit(x,y)

y_pred_knn = knn_reg.predict([[6.5]])
print(y_pred_knn)

#Decission Tree
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(criterion='poisson',max_depth=3, random_state=0)
dt_reg.fit(x,y)

dt_pred = dt_reg.predict([[6.5]])
print(dt_pred)

#Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=20, random_state=43)
rf_reg.fit(x,y)

rf_pred = rf_reg.predict([[6.5]])
print(rf_pred)


