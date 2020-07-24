# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Installing dependencies
# %pip install numpy
# %pip install pandas
# %pip install matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

# + _uuid="3977359aa7f65cc9d96fdf0ede4e99116fb314c6"
#Loading the dataset
diabetes_data = pd.read_csv('./diabetes.csv')

#Print the first 5 rows of the dataframe.
diabetes_data.head()

# + _uuid="8322614f5de4888d713c0468a43ae2a3eb1b8862"
## observing the shape of the data --> (#_rows, #_cols)
diabetes_data.shape

# + _uuid="9d9bd9ecd9612fb32f0629a8a3c0a85a14b034cf"
X = diabetes_data.drop("Outcome",axis = 1)
y = diabetes_data.Outcome

# + _uuid="e10c79fe13861fb0fbe714c977f93bb8ed90a5fd"
X.head()

# + _uuid="f14c88ae0a061de30566d336608d195ba89993d9"
# %pip install sklearn

#importing train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)

# 'test_size' — This parameter decides the size of the data that has to be split as the test dataset. This is given as a fraction
# 'random_state' - an integer, which will act as the seed for the random number generator during the split. Setting the random_state is desirable for reproducibility
# 'stratify' - makes a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to parameter stratify. For example, if variable y is a binary categorical variable with values 0 and 1 and there are 25% of zeros and 75% of ones, stratify=y will make sure that your random split has 25% of 0's and 75% of 1's.
# -

X_train.shape

X_test.shape

# +
# StandardScaler() will normalize the features i.e. each column of X, INDIVIDUALLY (!!!) so that each column/feature/variable will have μ = 0 and σ = 1.
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  

# Compute the mean and std to be used for later scaling.
scaler.fit(X_train)

# Perform standardization by centering and scaling
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 
# -

# # K-NN

# + _uuid="a7081050c51df07b8af1cd18c9be61f041a97fb8"
from sklearn.neighbors import KNeighborsClassifier

test_scores = []
train_scores = []

for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))

# + _uuid="ee126a72ca24e54ee78bfac94a21dfac1a3edee1"
## score that comes from testing on the same datapoints that were used for training
max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))

# + _uuid="8bbfda9d066c354f974dcb1180c3348aaa915c4e"
## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))

# + [markdown] _uuid="fe08768381ea8011d90ae58149c8e41b0a707da2"
# ### K-NN Result Visualisation

# + _uuid="2a5c0b4fde15148a049fa340a58f5b4fa421e614"
plt.figure(figsize=(12,5))
plt.plot(range(1,15),train_scores,marker='*',label='Train Score')
plt.plot(range(1,15),test_scores,marker='o',label='Test Score')
# -

# This one compares...

# + [markdown] _uuid="1db31455aba31edc524091fa0914743a284034c5"
# #### The best result is captured at k = 11 hence 11 is used for the final model

# + _uuid="277c1bb9c48cca13536ac8ba71604818d323fae0"
#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(11)

knn.fit(X_train,y_train)
knn.score(X_test,y_test)

# + [markdown] _uuid="ab1e49d83f39a6ddc780c394d3a052b49508c6ac"
# ## Model Performance Analysis

# + _uuid="d09044f60af8405e7334c2062404336d0849e871"
#import confusion_matrix
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
y_pred = knn.predict(X_test)
#confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

# Better ways to display Confusion Matrix

# + _uuid="6ac998149c1f0dd304b807707f0dc44dd2b2ffb3"
#import classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
# -



# + _uuid="20b2083d2eaf2fca599eb6f2ef8803be0b1ac5d7"
from sklearn.metrics import roc_curve
y_pred_proba = knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# + _uuid="379eefad0181f1f57ffbb3634ab6d132af17464f"
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=11) ROC curve')
plt.show()

# + _uuid="6c92773e49532f6133b23d511058202bb77ff2cd"
#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba)

# +
#### Export the scaler and model ####

import pickle
# Export the scaler
with open('./diabetes-scaler.pkl', 'wb') as scaler_pkl:
  pickle.dump(scaler, scaler_pkl)

# Export the model
with open('./diabetes-knn-model.pkl', 'wb') as model_pkl:
  pickle.dump(knn, model_pkl)

# +
#### Export the scaler and model ####

import pickle

# Import all the packages you need for your model below
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load the scaler 
with open('./diabetes-scaler.pkl', 'rb') as scaler_pkl:
    scaler_load = pickle.load(scaler_pkl)
    
# Load the model
with open('./diabetes-knn-model.pkl', 'rb') as model_pkl:
    knn_load = pickle.load(model_pkl)

# +
# Unseen data (create a new observation for testing)
# Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
# 6	148	72	35	0	33.6	0.627	50	1
# 1	85	66	29	0	26.6	0.351	31	0
# 8	183	64	0	0	23.3	0.672	32	1

X_unseen = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50], [1, 85, 66, 29, 0, 26.6, 0.351, 31], [8, 183, 64, 0, 0, 23.3, 0.672, 32]])
# X_unseen = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

# Apply the scale
X_unseen_scale = scaler_load.transform(X_unseen)

# Get the result
result = knn_load.predict(X_unseen_scale)

# Print result to the console
print('Predicted result for observation ' + str(X_unseen) + ' is: ' + str(result))
# -


