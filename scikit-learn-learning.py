import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

wine = pd.read_csv('winequality-red.csv', sep=';')
# print(wine.head())
# print(wine.info())

# preprocessing
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names)
wine['quality'].unique()
label_quality = LabelEncoder
wine['quality'] = label_quality.fit_transform(wine['quality'],wine['quality'])
#print(wine.head(10))
print(wine['quality'].value_counts())
sns.countplot(wine['quality'])
plt.show()

#seperate the dataset
x = wine.drop('quality', axis=1)
y = wine['quality']

#spliting train and test dataset
X_train, X_test, y_train, y_test =train_test_split(x,y, test_size=0.2, random_state=42)

#standard scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Randomforest classifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

'''
#model performance
print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test,pred_rfc))
'''

#svm classifier

clf = svm.SVC()
clf.fit(X_train, y_train)
pred_clf = clf.predict(X_test)

print(classification_report(y_test, pred_clf))
print(confusion_matrix(y_test,pred_clf))