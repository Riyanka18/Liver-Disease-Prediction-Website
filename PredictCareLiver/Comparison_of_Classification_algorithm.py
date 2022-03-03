
d={}
d2={}

from sklearn.metrics import precision_score, recall_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv('indian_liver_patient.csv')

#####_____Logistic Regression

from sklearn.preprocessing import LabelEncoder

labelencoder_Y= LabelEncoder()
dataset.iloc[:,1]=labelencoder_Y.fit_transform(dataset.iloc[:,1].values)
#1 because dignosis is at index 1
#so we exclude the first column and take x values from 2nd column
X = dataset.iloc[:,2:31].values
y = dataset.iloc[:,1].values

dataset["Albumin_and_Globulin_Ratio"] = dataset.Albumin_and_Globulin_Ratio.fillna(dataset['Albumin_and_Globulin_Ratio'].mean())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)
print()
pn=precision_score(y_pred_train,y_train)
rn=recall_score(y_pred_train,y_train)

d["Precision LRegression"]=pn
d2["Recall LRegression"]=rn
'''
print('Accuracy of test set for Logistic Regression = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))
print("Precision of test set for Logistic Regression = ",pn)
print("Recall of test set for Logistic Regression = ",rn)

###########################################################################################################################################
#Random forest
X = dataset.iloc[:,2:32].values
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
c= RandomForestClassifier(n_estimators = 10)
c.fit(X_train, y_train)

# Predicting the Test set results
y_p= c.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_p, y_test)

y_pred_train = c.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

pn=precision_score(y_pred_train,y_train)
rn=recall_score(y_pred_train,y_train)

d["Precision Random"]=pn
d2["Recall Random"]=rn

print()
print('Accuracy for test set for Random Forest = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))
print("Precision of test set for Random Forest = ",pn)
print("Recall of test set for Random Forest = ",rn)

##################################################
#DECISION TREE

X = dataset.iloc[:,2:32].values
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)
pn=precision_score(y_pred_train,y_train)
rn=recall_score(y_pred_train,y_train)

d["Precision DTree"]=pn
d2["Recall DTree"]=rn

print()
print('Accuracy for test set for Decision Tree = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))
print("Precision of test set for Decision Tree = ",pn)
print("Recall of test set for Decision Tree = ",rn)
plt.show()
#########################################################################################################
# Plot
ind=np.arange(len(d))
palette = sns.color_palette("husl", len(d))

f=plt.figure()
plt.plot(ind, list(d.values()), color="blue")
plt.xticks(ind, list(d.keys()))
plt.xlabel("Classification Algorithms")
plt.ylabel("Precision")
plt.show()
plt.plot(ind, list(d2.values()), color="blue")
plt.xticks(ind, list(d2.keys()))
plt.xlabel("Classification Algorithms")
plt.ylabel("Recall")
plt.show()
'''