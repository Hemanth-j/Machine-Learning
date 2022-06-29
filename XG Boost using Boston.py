# XG BOOST Regressor implementation
#we tried with linear regression which gave 58% accuracy where , XGBOOSt gave 76%
import matplotlib.pyplot as plt
import pandas as pd
import xgboost
from sklearn.datasets import load_boston
data=load_boston()
df=pd.DataFrame(data=data.data,columns=data.feature_names)
df["target"]=data.target
x=df.drop("target",axis="columns")
y=df.iloc[:,[-1]]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,train_size=0.8)
from xgboost import XGBRegressor
x_reg=XGBRegressor(learning_rate=0.3,gamma=150,n_estimators=100,max_depth=12,random_state=42,booster="gbtree",n_jobs=2,objectvie='reg:squarederror')
x_reg.fit(x_train,y_train)
print(x_reg.score(x_test,y_test))
