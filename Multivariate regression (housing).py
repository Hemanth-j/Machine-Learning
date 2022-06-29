import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.linear_model import LinearRegression
path="C:\\Users\\HEMANTH\\Downloads\\housing.csv\\housing.csv"
data=pd.read_csv(path)
#print(data)
x=data.iloc[:,[3,4,5]]
y=data.iloc[:,[-2]].head()
#print(x)
#print(x["total_rooms"])
import math
cx=math.floor(x.median())
x=x.fillna(cx)
reg=LinearRegression()
reg.fit(x,y)