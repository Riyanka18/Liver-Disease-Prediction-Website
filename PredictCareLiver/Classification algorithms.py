#The testing of Random Forest Classifier for detection of Malignant or Benign tumor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv('Breast Cancer data.csv')

dataset['diagnosis'].value_counts()


get_ipython().magic('matplotlib inline')
sns.countplot(dataset['diagnosis'],label='count');

dataset.dtypes.head()

#The values of M and B must be encoded I will try to encode it to 0's and 1'
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y= LabelEncoder()
dataset.iloc[:,1]=labelencoder_Y.fit_transform(dataset.iloc[:,1].values)
#1 because dignosis is at index 1

#higher values indicate that the column has greater influence on prediction
#negative values show negative influence and 0 values show no influence

#The dataset set from column 2 onwards is independent(X) 
#And column 1 is dependent(Y)

X=dataset.iloc[:,2:31].values
Y=dataset.iloc[:,1].values


# Split the data into training and testing set
# 80% training 20% testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size = 0.2,random_state= 0)

from sklearn.preprocessing import StandardScaler
#SCale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#various algorithms used
#Now we will classify using various Models
def Classify(X_train,Y_train):
    
    #logistic regression
    log=LogisticRegression(random_state=0)
    log.fit(X_train,Y_train)
    
    #Decision Trees
    tree=DecisionTreeClassifier(criterion='entropy',random_state=0)
    tree.fit(X_train,Y_train)
    #Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
    clf.fit(X_train, Y_train)

    print("Accuracy of logistic regression",log.score(X_train, Y_train))
    print("Accuracy of Decision Tree",tree.score(X_train, Y_train))
    print("Accuracy of Random Forest",clf.score(X_train, Y_train))
    return log,tree,clf
    
model=Classify(X_train,Y_train)

#Checking accuracy of confusion matrix

from sklearn.metrics import confusion_matrix
for i in range(len(model)):
    print("Model i",i)
    cm=confusion_matrix(Y_test,model[0].predict(X_test))
    true_pos=cm[0][0]
    true_neg=cm[1][1]
    false_pos=cm[0][1]
    false_neg=cm[1][0]
    print(cm)
    print("Testing accuracy: ",(true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg))
