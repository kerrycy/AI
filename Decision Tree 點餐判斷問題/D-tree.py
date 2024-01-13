import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline  

from sklearn import preprocessing, tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
clf = tree.DecisionTreeClassifier(random_state=0)

#讀入新建立之.csv檔
titanic = pd.read_csv("/content/train1.csv")

#將資料轉為數字
titanic["Alt"].replace(['Yes','No'],[0,1],inplace=True)
titanic["Bar"].replace(['Yes','No'],[0,1],inplace=True)
titanic["Fri"].replace(['Yes','No'],[0,1],inplace=True)
titanic["Hun"].replace(['Yes','No'],[0,1],inplace=True)
titanic["Pat"].replace(['None','Some','Full'],[0,1,2],inplace=True)
titanic["Price"].replace(['$$$','$$','$'],[0,1,2],inplace=True)
titanic["Rain"].replace(['Yes','No'],[0,1],inplace=True)
titanic["Res"].replace(['Yes','No'],[0,1],inplace=True)
titanic["Type"].replace(['French','Thai','Burger','Italian'],[0,1,2,3],inplace=True)
titanic["GoalWillWait"].replace(['Yes','No'],[0,1],inplace=True)


X= pd.DataFrame([titanic["Alt"], titanic["Bar"]]).T
X.columns=["Alt", "Bar"]
y = titanic["GoalWillWait"]



Xtrain, XTest, yTrain, yTest = \
train_test_split(X, y, test_size=0.9, random_state=1)
dtree =tree.DecisionTreeClassifier()
dtree.fit(Xtrain, yTrain)
print("準確率 :", dtree.score(XTest, yTest))
preds= dtree.predict_proba(X=XTest)
print(pd.crosstab(preds[:,0], columns=[X["Alt"],XTest["Bar"]]))

preds= dtree.predict_proba(X=XTest)
print(pd.crosstab(preds[:,0], columns=[X["Alt"],XTest["Bar"]]))
clf = clf.fit(XTest, yTest)
plt.figure()
tree.plot_tree(clf)
plt.show()