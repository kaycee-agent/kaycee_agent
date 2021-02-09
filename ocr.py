from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


mnist = datasets.load_digits()
X = mnist['data']
y = mnist['target']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2 , random_state = 3116 )
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
#Initialize Neural Network, set up parameters for your grid search and implement a Random Search procedure
ann = MLPClassifier()
grid_parameters = {'hidden_layer_sizes': list(range(100,450,50)), 'activation':['relu','identity', 'logistic','tanh'], 'learning_rate':['constant','adaptive','invscaling']}
ann_grid_search = RandomizedSearchCV(ann, grid_parameters, cv = 5, n_iter = 10)
ann_grid_search.fit(X_train,y_train)
#Accuracy score
y_pred = ann_grid_search.predict(X_test)
print('Accuracy Score:', accuracy_score(y_test,y_pred))
#Best hyperparameters
print(ann_grid_search.best_estimator_)
#First 10 predictions concatenated with the actual targets
print(np.concatenate((y_test[:10].reshape(-1,1),y_pred[:10].reshape(-1,1)),axis = 1))
 

   
