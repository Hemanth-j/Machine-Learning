#Simple Linear Regression using Custom Dataset

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path="C:\\Users\\HEMANTH\\Downloads\\1.01. Simple linear regression.csv"
dataa=pd.read_csv(path)
print(dataa.shape)
#x=dataa["SAT"]
#y=dataa["GPA"]
X=np.array(dataa["SAT"]).reshape(-1,1)
Y=np.array(dataa["GPA"]).reshape(-1,1)
lnr=LinearRegression()
#x_train,x_test,y_tain,y_test=train_test_split(X,Y,test_size=0.25)
lnr.fit(X,Y)
y_pr=lnr.predict(X)
print(lnr.score(X,Y))
plt.scatter(X,Y)
plt.plot(X,y_pr,color="r")
plt.title("simple ML algorithm")
plt.show()