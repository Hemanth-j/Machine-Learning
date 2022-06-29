#adaboost implementation
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
data=load_breast_cancer()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
clf=SVC(kernel="rbf",probability=True)
#clf.fit(x_train,y_train)
#print(clf.score(x_test,y_test))
from sklearn.ensemble import AdaBoostClassifier
#here if we provide default base estimator , we get accuracy around 98%,but when we use svc as base estimator we get around 93%
ada=AdaBoostClassifier(clf,n_estimators=50,learning_rate=1,random_state=0)
ada.fit(x_train,y_train)
print("adaboost score : ",ada.score(x_test,y_test))
y_pred=ada.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
import seaborn as sns
sns.heatmap(cm,annot=True)
print("Plain classifier :")
clf.fit(x_train,y_train)
print("plain classy : ",clf.score(x_test,y_test))
#plt.show()
