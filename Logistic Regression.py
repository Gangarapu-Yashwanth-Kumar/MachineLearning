#lets import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#lets read dataset
dataset=pd.read_csv(r"E:\Datasets\logit classification.csv")

# DIVIDE THE DATASET INTO INDEPENDENT AND DEPENDENT VARIABLES
X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
'''
#Feature Scaling
from sklearn.preprocessing import Normalizer
sc=Normalizer()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
'''
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)

from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print(cr)

bias=classifier.score(X_train,y_train)
print(bias)

variance=classifier.score(X_test,y_test)
print(variance)

from sklearn.metrics import roc_curve, roc_auc_score

y_prob = classifier.predict_proba(X_test) [:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

# Print AUC
print("AUC Score:" , auc_score)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color = 'blue', label = 'ROC curve (area=%0.2f)' % auc_score)
plt.plot([0,1],[0,1], color='gray', linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Reciever operating charactersistic (ROC)')
plt.legend(loc="lower right")
plt.grid()
plt.show()

#Future Prediction
dataset1=pd.read_csv(r"E:\Datasets\final1.csv")
d2=dataset1.copy()
dataset1=dataset1.iloc[:, [3,4]].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
M=sc.fit_transform(dataset1)

y_pred1=pd.DataFrame()

d2['y_pred1']=classifier.predict(M)

d2.to_csv('final1.csv')


