import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"C:\Users\rohit\Desktop\Datasets\iris.csv")

X = data.drop(['Id', 'Species'], axis = 1)
y  = data['Species']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=2)

#Model Build

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
#y_pred = logreg.predict(X_test)
#print(metrics.accuracy_score(y_test, y_pred))


pickle.dump(logreg, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

