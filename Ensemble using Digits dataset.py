#2 examples for random forest i) using load digits dataset   ii) using IRIS dataset
"""import pandas as pd
from  sklearn.datasets import load_digits
data=load_digits()
df=pd.DataFrame(data=data.data,columns=data.feature_names)
df["target"]=data.target
x=df.drop(["target"],axis="columns")
y=df.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42,stratify=y)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
r_clf=RandomForestClassifier()
#hyper parameter tuning
param={"n_estimators":[161,162,163,164,165,166,167,168,169,170],"criterion":["gini","entropy"]}
cv=GridSearchCV(r_clf,param,cv=5)
cv.fit(x_train,y_train)
print(cv.best_params_,cv.best_score_)
print(cv.score(x_test,y_test))"""

#using IRIS dataset
import pandas as pd
from  sklearn.datasets import load_iris
data=load_iris()
df=pd.DataFrame(data=data.data,columns=data.feature_names)
df["target"]=data.target
x=df.drop(["target"],axis="columns")
y=df.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42,stratify=y)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
r_clf=RandomForestClassifier(criterion="entropy",n_estimators=80,random_state=0)
r_clf.fit(x_train,y_train)
print(r_clf.get_params())
print(r_clf.score(x_test,y_test))
