#XG Boost classification using titanic dataset
#and comparision with different classification models
import seaborn as sns
df=sns.load_dataset("titanic")
f=df[["survived","pclass","sex","age"]]
f=f.dropna()
x=f.drop("survived",axis="columns")
mapp={"male":0,"female":1}
x["sex_map"]=x["sex"].map(mapp)
x=x.drop("sex",axis="columns")
y=f.survived
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
def modelll(model,x_tr,y_tr,x_te,y_te):
    model.fit(x_tr,y_tr)
    print(model," : ",model.score(x_te,y_te))
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
modelll(LogisticRegression(random_state=0),x_train,y_train,x_test,y_test)
modelll(DecisionTreeClassifier(random_state=0),x_train,y_train,x_test,y_test)
modelll(SVC(kernel="linear",random_state=0),x_train,y_train,x_test,y_test)
modelll(SVC(kernel="poly",random_state=0),x_train,y_train,x_test,y_test)
modelll(SVC(kernel="rbf",random_state=0),x_train,y_train,x_test,y_test)
modelll(SGDClassifier(random_state=0),x_train,y_train,x_test,y_test)
modelll(RandomForestClassifier(random_state=0),x_train,y_train,x_test,y_test)
modelll(XGBClassifier(n_estimators=90,random_state=0,learning_rate=0.06,max_depth=7,n_jobs=-1),x_train,y_train,x_test,y_test)
modelll(GradientBoostingClassifier(random_state=0),x_train,y_train,x_test,y_test)