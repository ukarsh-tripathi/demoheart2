import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('heart.csv') 
from sklearn.model_selection import train_test_split
X = df.drop('target', axis = 1).values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train,y_train)
pickle.dump(lr,open('model.pkl','wb'))
