# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# Installing dependencies
# %pip install numpy
# %pip install pandas
# %pip install matplotlib

# abc
# This is a comment

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

# %% _uuid="3977359aa7f65cc9d96fdf0ede4e99116fb314c6"
#Loading the dataset
diabetes_data = pd.read_csv('./diabetes.csv')

#Print the first 5 rows of the dataframe.
diabetes_data.head()

# %% _uuid="8322614f5de4888d713c0468a43ae2a3eb1b8862"
## observing the shape of the data --> (#_rows, #_cols)
diabetes_data.shape

# %% _uuid="9d9bd9ecd9612fb32f0629a8a3c0a85a14b034cf"
X = diabetes_data.drop("Outcome",axis = 1)
y = diabetes_data.Outcome

# %% _uuid="e10c79fe13861fb0fbe714c977f93bb8ed90a5fd"
X.head()

# %% _uuid="f14c88ae0a061de30566d336608d195ba89993d9"
# %pip install sklearn

#importing train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)

# 'test_size' — This parameter decides the size of the data that has to be split as the test dataset. This is given as a fraction
# 'random_state' - an integer, which will act as the seed for the random number generator during the split. Setting the random_state is desirable for reproducibility
# 'stratify' - makes a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to parameter stratify. For example, if variable y is a binary categorical variable with values 0 and 1 and there are 25% of zeros and 75% of ones, stratify=y will make sure that your random split has 25% of 0's and 75% of 1's.

# %%
X_train.shape

# %%
X_test.shape

# %%
# StandardScaler() will normalize the features i.e. each column of X, INDIVIDUALLY (!!!) so that each column/feature/variable will have μ = 0 and σ = 1.
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  

# Compute the mean and std to be used for later scaling.
scaler.fit(X_train)

# Perform standardization by centering and scaling
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 

# %% [markdown]
# # LOGISTIC REGRESSION

# %%
# Logistic regression essentially adapts the linear regression formula to allow it to act as a classifier
# Ref: https://towardsdatascience.com/the-basics-logistic-regression-and-regularization-828b0d2d206c

# If the logistic regression model used for addressing the binary classification kind of problems it’s known as the binary logistic regression classifier. Whereas the logistic regression model used for multiclassification kind of problems, it’s called the multinomial logistic regression classifier.
# - Sigmoid function: used in the logistic regression model for binary classification.
# - Softmax function: used in the logistic regression model for multiclassification.

# Ref: https://dataaspirant.com/2017/03/14/multinomial-logistic-regression-model-works-machine-learning/

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=0.01).fit(X_train, y_train)

# C: Regularization. Regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error.
# Given how Scikit cites it as being: C = 1/λ. The relationship, would be that lowering C - would strengthen the Lambda regulator.

print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

# %%
logreg.intercept_.T

# %%
logreg.coef_.T

# %% [markdown]
# ### Confusion Matrix

# %%
from sklearn.metrics import classification_report, confusion_matrix  
y_pred = logreg.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# Format of sklearn confusion_matrix:
#                  Class Predicted by the model
#                  0       1
#  Actual  0       150     17
#  Class   1       53      36

# %% [markdown]
# ### ROC

# %%
from sklearn import metrics


print("Accuracy=", metrics.accuracy_score(y_test, y_pred))
 
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="logreg, auc="+str(auc))
plt.plot([0,1],[0,1],'k--')
plt.legend(loc=4)
plt.show()

# %% [markdown]
# ### Gains / Lift Chart

# %%
# %pip install scikit-plot

# Ref: https://www.listendata.com/2014/08/excel-template-gain-and-lift-charts.html

import matplotlib.pyplot as plt
import scikitplot as skplt
y_pred_probas = logreg.predict_proba(X_test)

# %%
skplt.metrics.plot_cumulative_gain(y_test, y_pred_probas)
plt.show()

# %%
skplt.metrics.plot_lift_curve(y_test, y_pred_probas)
plt.show()

# %% [markdown]
# # K-NN

# %% _uuid="a7081050c51df07b8af1cd18c9be61f041a97fb8"
from sklearn.neighbors import KNeighborsClassifier

test_scores = []
train_scores = []

for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))

# %% _uuid="ee126a72ca24e54ee78bfac94a21dfac1a3edee1"
## score that comes from testing on the same datapoints that were used for training
max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))

# %% _uuid="8bbfda9d066c354f974dcb1180c3348aaa915c4e"
## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))

# %% [markdown] _uuid="fe08768381ea8011d90ae58149c8e41b0a707da2"
# ### K-NN Result Visualisation

# %% _uuid="2a5c0b4fde15148a049fa340a58f5b4fa421e614"
plt.figure(figsize=(12,5))
plt.plot(range(1,15),train_scores,marker='*',label='Train Score')
plt.plot(range(1,15),test_scores,marker='o',label='Test Score')

# %% [markdown]
# This one compares...

# %% [markdown] _uuid="1db31455aba31edc524091fa0914743a284034c5"
# #### The best result is captured at k = 11 hence 11 is used for the final model

# %% _uuid="277c1bb9c48cca13536ac8ba71604818d323fae0"
#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(11)

knn.fit(X_train,y_train)
knn.score(X_test,y_test)

# %% [markdown] _uuid="ab1e49d83f39a6ddc780c394d3a052b49508c6ac"
# ## Model Performance Analysis

# %% _uuid="d09044f60af8405e7334c2062404336d0849e871"
#import confusion_matrix
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
y_pred = knn.predict(X_test)
#confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

# Better ways to display Confusion Matrix

# %% _uuid="6ac998149c1f0dd304b807707f0dc44dd2b2ffb3"
#import classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

# %%

# %% _uuid="20b2083d2eaf2fca599eb6f2ef8803be0b1ac5d7"
from sklearn.metrics import roc_curve
y_pred_proba = knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# %% _uuid="379eefad0181f1f57ffbb3634ab6d132af17464f"
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=11) ROC curve')
plt.show()

# %% _uuid="6c92773e49532f6133b23d511058202bb77ff2cd"
#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba)

# %% _uuid="e369c794253b71d3daa1444fb7d11872fb8a110c"
#import GridSearchCV
from sklearn.model_selection import GridSearchCV
#In case of classifier like knn the parameter to be tuned is n_neighbors
param_grid = {'n_neighbors':np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X_train,y_train)

print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))

# %%

# %% [markdown]
# # Naiive Bayes

# %%
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# %%
# Initiating the Gaussian Classifier
mod = GaussianNB()

# %%
# Training your model 
mod.fit(X_train, y_train)

# %%
# Predicting Outcome 
predicted = mod.predict(X_test)

# %%
mod.score(X_test,y_test)

# %%
# Confusion Matrix
y_pred = mod.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# %%

# %%
# You can compare the performance of multiple models in one ROC chart. Wrtie your own codes in the cells below.

# %%
from sklearn import metrics

# Logreg
logreg_y_pred_proba = logreg.predict_proba(X_test)[::,1]
logreg_fpr, logreg_tpr, _ = metrics.roc_curve(y_test,  logreg_y_pred_proba)
plt.plot(logreg_fpr,logreg_tpr,label="logreg")
plt.plot([0,1],[0,1],'k--')
plt.legend(loc=4)
plt.xlabel('fpr')
plt.ylabel('tpr')


# KNN
knn = KNeighborsClassifier(19)
knn.fit(X_train,y_train)
knn_y_pred_proba = knn.predict_proba(X_test)[:,1]
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_y_pred_proba)
plt.plot(knn_fpr,knn_tpr,label='Knn')
plt.legend(loc=4)
plt.title('Different ROC curves')


# Naive Bayes
mod_y_pred = mod.predict(X_test)
mod_fpr, mod_tpr, _ = roc_curve(y_test, mod_y_pred)
plt.plot(mod_fpr,mod_tpr,label='Naive Bayes')
plt.legend(loc=4)
plt.show()

# %%
