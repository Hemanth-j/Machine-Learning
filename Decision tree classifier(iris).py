from sklearn.datasets import load_iris
import pandas as pd
data=load_iris()
df=pd.DataFrame(data.data,columns=data.feature_names)
#y=pd.DataFrame(data.target,columns=data.target_names)
df["target"]=data.target
df["target"]=df.target.map({0:"setosa",1:"versicolor",2:"virginica"})
x=df.drop("target",axis="columns")
y=df.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.85,random_state=0)
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(max_depth=25)
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
print(clf.predict([[5.8,2.8,5.1,2.4]]))
import matplotlib.pyplot as plt
print(x_train)
#plt.scatter(x_train["petal width (cm)"],y_train,marker="+",color="Green")
#plt.scatter(x_train["petal length (cm)"],y_train,marker="+",color="red")
from sklearn import tree
tree.plot_tree(clf,feature_names=data.feature_names,class_names=data.target_names,filled=True)
plt.title("Decision Graph")
plt.show()