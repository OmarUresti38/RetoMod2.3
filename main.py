import sys
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sklearn'])
import numpy as np
import math
import operator
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics

data = pd.read_csv('Estatura-peso_HyM2.csv')

X = data['H_peso'].values
Y = data['H_estat'].values

# calculate mean of x & y using an inbuilt numpy method mean()
mean_x = np.mean(X)
mean_y = np.mean(Y)

m = len(X)

#Sklearn Linear Regression model
Xsl = X.reshape(m, 1)
reg = LinearRegression()
reg = reg.fit(Xsl, Y)

Y_pred = reg.predict(Xsl)
r2_square = reg.score(Xsl, Y)
print(f'r2sl = {r2_square}')

plt.plot(Xsl, Y_pred, color='k', label='Sklearn RL')
plt.scatter(X, Y, c='#ef5423', label='data points')

plt.xlabel('Peso')
plt.ylabel('Estatura')
plt.xlim([50, 100])
plt.legend()
plt.show()
