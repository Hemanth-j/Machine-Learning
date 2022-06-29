#Decision tree illustration

import  pandas as pd
df=pd.read_csv("C:\\Users\\HEMANTH\\Downloads\\Titanic.csv")
import seaborn as sns
import matplotlib.pyplot as plt
df["Age"]=df.iloc[:,5].fillna(df.iloc[:,5].mean())
#df["Age"]=df["Age"].interpolate(method="pad")
#print(df.iloc[:,5].mean())
#print(df["Age"])
#sns.heatmap(df.isnull())
#plt.show()
y=df["Survived"]
c=[False,False,True,False,True,True,False,False,False,True,False,False]
x=df.iloc[:,c]
x["sex"]=df.Sex.map({"female":2,"male":1})
x=x.drop("Sex",axis="columns")
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y[:714],train_size=0.8,random_state=0)
from sklearn import tree
clf=tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
plt.scatter(x["Fare"],x["Age"],color="red",marker="+")
plt.scatter(x["Pclass"],x["Age"],color="yellow",marker="+")
plt.show()
