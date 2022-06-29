# Iris project on Bagging
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
data=load_iris()
df=pd.DataFrame(data=data.data,columns=data.feature_names)
df["target_names"]=data.target
x=df.drop("target_names",axis="columns")
y=df.target_names
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,train_size=0.8,random_state=0)
#print(y.shape)
#print(y.value_counts())
#clf.fit(x_train,y_train)
#print(clf.score(x_train,y_train))
#print(clf.score(x_test,y_test))
from sklearn import tree
#Hyper Parameter Tuning
"""parms={"criterion":["gini","entropy"],"max_depth":["None",3,4,5,6,7,8,9,10],"max_features":["None",1,2,3,4],"splitter":["best"],"max_leaf_nodes":[2,3,4,5,6,7,8,9,10],"random_state":["None",0,42],"ccp_alpha":["None",0,1,2,3,4,5,6,7],"class_weight":["None",1,2,3,4,5,6,7,8],}
#tree.plot_tree(clf,fontsize=7,rounded=True,feature_names=data.feature_names,class_names=data.target_names,filled=True)
#plt.show()
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(clf,parms,cv=7,return_train_score=False)
grid.fit(x_train,y_train)
print(grid.best_score_,print(grid.best_params_))"""
#print(clf.get_params())   #best params
from sklearn.model_selection import cross_val_score
"""cv=cross_val_score(clf,x_train,y_train,cv=5)
print("*******")
print(cv.mean())
print("*******")
cvv=cross_val_score(clf,x_test,y_test,cv=5)
print(cvv.mean())"""
from sklearn.ensemble import BaggingClassifier
bagg=BaggingClassifier(clf,n_estimators=100,max_samples=0.8,oob_score=True,random_state=0)
bagg.fit(x_train,y_train)
print(clf.get_params())
print("oob score : ",bagg.oob_score_)
print("normal base model score:")
print("train:",bagg.score(x_train,y_train))
print("testing:",bagg.score(x_test,y_test))
train_score=cross_val_score(bagg, x_train, y_train, cv=10)
print("Training cross val score: ", train_score.mean())
test_score=cross_val_score(bagg, x_test, y_test, cv=10)
print("Testing cross val score: ", test_score.mean())
