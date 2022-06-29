#Random forest implementation uusing diabetes datset
#tip: Random foreest=Decision tree with bagging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df=pd.read_csv("C:\\Users\\HEMANTH\\Downloads\\diabetes.csv")
x=df.drop("Outcome",axis="columns")
y=df.Outcome
from sklearn.preprocessing import StandardScaler
scl=StandardScaler()
x_scl=scl.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scl,y,train_size=0.8,stratify=y,random_state=0)
from sklearn.ensemble import RandomForestClassifier
r_clf=RandomForestClassifier(random_state=42)
r_clf.fit(x_train,y_train)
print(r_clf.score(x_train,y_train))
print(r_clf.score(x_test,y_test))
from sklearn.model_selection import cross_val_score
cv=cross_val_score(r_clf,x_scl,y,cv=5)
print("cross validation score",cv.mean())
from sklearn import tree
plt.figure(figsize=(5,5))
_=tree.plot_tree(r_clf.estimators_[0],feature_names=x.columns,filled=True,rounded=True)
plt.show()