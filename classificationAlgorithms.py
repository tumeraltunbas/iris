import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
df = pd.read_excel('Iris.xls')

y = df.iloc[:,4:].values
x = df.iloc[:,:4].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)


#Logistic Regression - Needs scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(random_state=0)
lg.fit(x_train,y_train)
y_predict = lg.predict(x_test)
print('LG')
print(confusion_matrix(y_test,y_predict))

#KNN - Needs scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10,metric='cityblock') #Cityblock gives less error rate than others.
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
print('KNN')
print(confusion_matrix(y_test,y_pred))

#Support Vector Machine - needs scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
print('SVM')
print(confusion_matrix(y_test,y_pred))

#Naive Bayes - needs scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
from sklearn.naive_bayes import GaussianNB
gb = GaussianNB()
gb.fit(x_train,y_train)
y_pred = gb.predict(x_test)
print('GNBayes')
print(confusion_matrix(y_test,y_pred))

#Decision Tree - no needs scaling
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
print('DT')
print(confusion_matrix(y_test,y_pred))

#Random Forest - no needs scaling
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10, criterion='entropy')
rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)
print('RF')
print(confusion_matrix(y_test,y_pred))