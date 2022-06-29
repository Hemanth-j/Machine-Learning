#Gradient Boosting Regressor using Boston housing Dataset
from sklearn.datasets import load_boston
data=load_boston()
import pandas as pd
df=pd.DataFrame(data=data.data,columns=data.feature_names)
df["target"]=data.target
x=df.drop("target",axis="columns")
y=df.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
from sklearn.ensemble import GradientBoostingRegressor
reg=GradientBoostingRegressor(n_estimators=3000,learning_rate=0.2,random_state=0)
reg.fit(x_train,y_train)
print("Boosting regression score :",reg.score(x_test,y_test))
from sklearn.svm import SVR
s_reg=SVR(kernel="poly")
s_reg.fit(x_train,y_train)
print("Normal svm regression score :",s_reg.score(x_test,y_test))