from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path="C:\\Users\\HEMANTH\\OneDrive\\Desktop\\Book1.csv"
data=pd.read_csv(path)
#print(data)
x=data.level.values.reshape(-1,1)
y=data.salary.values.reshape(-1,1)
poly=PolynomialFeatures(degree=6)
x_poly=poly.fit_transform(x)
reg=LinearRegression()
reg.fit(x_poly,y)
print(reg.score(x_poly,y))
print(reg.predict(poly.fit_transform([[10]])))
plt.scatter(x,y,color="red")
plt.plot(x,reg.predict(poly.fit_transform(x)),color="yellow")
plt.title("Machine Learning algo...")
plt.show()