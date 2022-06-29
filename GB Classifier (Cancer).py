#Gradient Boosting Classification Implementation
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
data=load_breast_cancer()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
from sklearn.ensemble import GradientBoostingClassifier
g_clf=GradientBoostingClassifier(n_estimators=150,learning_rate=0.15,random_state=0)
g_clf.fit(x_train,y_train)
print(g_clf.score(x_test,y_test))
y_pred=g_clf.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
import seaborn as sns
sns.heatmap(cm,annot=True)
plt.show()