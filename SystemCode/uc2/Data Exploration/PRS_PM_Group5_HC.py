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

# # Parkway Project

# ### Load Data

# #### Import packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
pd.set_option('display.max_rows', 500)
import warnings
warnings.filterwarnings(action='once')

# #### Read data from file

#filePath = '.\data\ParkwaySampleDataForProject_05.xlsx'
#filePath = '.\data\ParkwaySampleDataForProject_06.xlsx'
filePath = '.\ParkwaySampleDataForProject_12_WithLabelNoFormula.xlsx'
df = pd.read_excel(filePath)
display(df.shape)

# ### Exploratory Data Analysis

# #### Explore data

# +
info = []
for col in df.columns:
    non_null  = len(df) - np.sum(pd.isna(df[col]))
    num_unique = df[col].nunique()
    col_type = str(df[col].dtype)

    info.append([col, non_null, num_unique, col_type])

features_info = pd.DataFrame(info, columns = ['colName','non-null values', 'unique', 'dtype'])

display(features_info)
#features_info.to_csv('Info_List.csv')

# +
# drop columns
col_drop = ['PAYER_NAME_1', 'PAYER_NAME_2', 'PAYER_NAME_3', 'PAYER_NAME_4', 'PAYER_NAME_5', 'DISCHARGE_TYPE_DESC', 'DOCTOR_NAME', 'SPECIALTY_DESC',
            'TOSP_STRING', 'TOSP_DESC1', 'TOSP_DESC2', 'TOSP_DESC3', 'TOSP_DESC4', 'DRG_DESC', 'PAYER_CODE1_V', 'PAYER_NAME1_V', 'PAYER_CODE2_V',
            'PAYER_NAME2_V', 'PAYER_CODE3_V', 'PAYER_NAME3_V', 'PAYER_CODE4_V', 'PAYER_NAME4_V', 'PACKAGE_DESC', 'PACKAGE_DESC1', 'PACKAGE_DESC2',
            'ICDCODE_STRING', 'PACKAGE_CODE', 'PACKAGE_PRICE', 'PACKAGE_EXCL', 'PACKAGE_ADJ', 'PACKAGE_CODE1', 'PACKAGE_CODE2','WRITE_OFF',
            'PCT_WRITE_OFF','PROF_FEE','TOTAL_FEES','TOTAL_PAID_AMT','PAYER_1_PAID_AMT','PAYER_2_PAID_AMT','PAYER_3_PAID_AMT',
            'PAYER_4_PAID_AMT','PAYER_5_PAID_AMT','PATIENT_SID','PATIENT_NUMBER']
df1 =  df.drop(col_drop, axis=1)

# convert dates
col_dt = df1.select_dtypes(include=np.datetime64).columns
for col in col_dt:
    df1[col+'_year'] = df1[col].dt.year
    df1[col+'_month'] = df1[col].dt.month
    df1[col+'_day'] = df1[col].dt.day
df1 =  df1.drop(col_dt, axis=1)

# convert objects to factors
col_obj = df1.select_dtypes(include=np.object).columns
for col in col_obj:
    df1[col] = pd.factorize(df1[col])[0] +1

print(df1.info())
# -

print(df['CASE_NUMBER'].nunique())

print(df[df.duplicated(subset=['CASE_NUMBER'], keep=False)])

df.hist(figsize=(15,20), layout=(-1,5))
plt.tight_layout()
plt.show()

print(df1)

# ### Feature Engineering

# Aggregate 'Admission_Age' using 'ADMISSION_DTE', 'DOB'
print(df['DOB'].head())
df['Admission_Age'] = df['ADMISSION_DTE'].dt.year-df['DOB'].dt.year
print(df[['ADMISSION_DTE', 'DOB','Admission_Age']])

# Aggregate 'Admission_Age' using 'ADMISSION_DTE', 'DOB'
print(df1['DOB_year'].head())
df1['Admission_Age'] = df1['ADMISSION_DTE_year']-df1['DOB_year']
print(df1[['ADMISSION_DTE_year', 'DOB_year','Admission_Age']])

# ### Split Data to Training and Test

type(df1)

df1.shape

df1.index = df1.CASE_NUMBER
df1 =  df1.drop(['CASE_NUMBER'], axis=1)

# Fill NAN with zeros
df1 = df1.fillna(0)

print(df1)

df1.to_csv('PARKWAY_PROCESSED_2_NAN.csv')

X = df1.drop("WRITE_OFF_LABEL",axis = 1)
y = df1.WRITE_OFF_LABEL

import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline
#importing train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)

X.head()

y.head()

y_train.head()

print('X_train.shape is ', X_train.shape)
print('X_test.shape is ', X_test.shape)
print('y_train.shape is ', y_train.shape)
print('y_test.shape is ', y_test.shape)

# +
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 
# -

print("Below is X_train")
X_train

#type(X_train)
np.where(np.isnan(X_train))

print("Below is X_test")
X_test

# ## LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=0.01).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

logreg.intercept_.T

logreg.coef_.T

# ### Confusion Matrix for Logistic Regression

from sklearn.metrics import classification_report, confusion_matrix  
y_pred = logreg.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# ### ROC from Logistic Regression

# +
from sklearn import metrics


print("Accuracy=", metrics.accuracy_score(y_test, y_pred))
 
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="logreg, auc="+str(auc))
plt.plot([0,1],[0,1],'k--')
plt.legend(loc=4)
plt.show()
# -

# # Decision Tree

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy',random_state=0)
dt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(dt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(dt.score(X_test, y_test)))

y_pred = dt.predict(X_test)

# +
from sklearn.metrics import classification_report, confusion_matrix  

print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# +
#DT visualizatin method 1

from sklearn.tree import export_graphviz

dotfile = open("dt2.dot", 'w')

export_graphviz(dt, out_file=dotfile,feature_names = X.columns,class_names=['0','1'])
dotfile.close()
# Copying the contents of the created file ('dt2.dot' ) to a graphviz rendering agent at http://webgraphviz.com/
# check out https://www.kdnuggets.com/2017/05/simplifying-decision-tree-interpretation-decision-rules-python.html

# +
#DT visualizatin method 2
# need to install Graphviz first https://graphviz.gitlab.io/_pages/Download/Download_windows.html
from sklearn.tree import export_graphviz
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

export_graphviz(dt, out_file='tree.dot', feature_names=X.columns,class_names=['0','1'])
# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'ParkwayWriteOff_tree.png', '-Gdpi=600'])

# Display in jupyter notebook
#from IPython.display import Image
#Image(filename = 'ParkwayWriteOff_tree.png')

# -

from sklearn import metrics
y_pred_dt = dt.predict(X_test)
print("Accuracy Score for DT =", metrics.accuracy_score(y_test, y_pred_dt))
fpr_dt, tpr_dt, _ = metrics.roc_curve(y_test,  y_pred_dt)
auc_dt = metrics.roc_auc_score(y_test, y_pred_dt)
#HC: Plot ROC for Decision Tree dt
plt.plot(fpr_dt,tpr_dt,label="Decision Tree, auc="+str(auc_dt))
plt.plot([0,1],[0,1],'k--')
plt.legend(loc=4)
plt.show()

# +
# Method 1: with bar chart
from matplotlib import pyplot
# get importance
importance = dt.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, %s, Score: %.5f' % (i,df1.columns[i],v))
# plot feature importance
print()
print('Below is the barchart for each feature\'s value')
pyplot.bar([x for x in range(len(importance))], importance)
#pyplot.bar([diabetes_data.columns[x] for x in range(len(importance))], importance)
pyplot.show()

#Method 2: Simply showing numbers
feature_importances = dict(zip(df1.columns, dt.feature_importances_))
feature_importances
# -

# ## NN MLP

# +
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  
# -

from sklearn.neural_network import MLPClassifier  
mlp = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000,verbose=2)  
mlp.fit(X_train, y_train)  

# +
predictions = mlp.predict(X_test)  

print("Accuracy", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

print("Accuracy on training set: {:.3f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test, y_test)))

import matplotlib.pyplot as plt
# %matplotlib inline
import matplotlib 
import pandas as pd
import numpy as np

plt.plot(mlp.loss_curve_)
plt.title("NN Loss Curve")
plt.xlabel("number of steps")
plt.ylabel("loss function")
plt.show()

from sklearn.neural_network import MLPClassifier  
mlp_enhanced = MLPClassifier(hidden_layer_sizes=(32,32), max_iter=1000,verbose=2)  
mlp_enhanced.fit(X_train, y_train)  

# +
predictions = mlp_enhanced.predict(X_test)  

print("Accuracy for the enhanced model: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  

print("Accuracy on training set: {:.3f}".format(mlp_enhanced.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp_enhanced.score(X_test, y_test)))
# -

# ## With SMOTE (Synthetic Minority Over Sampling Technique)

# Created 2 instances of smote here
# sm = normal smote with default sampling strategy = 1.0 -> this will create synthetic data for minority class such that the
#   number of samples for minority class equals with samples for majority class
# sm_ss = smote with sampling strategy = 0.1  -> this will create synthetic data for minority class such that the 
#   number of samples for minority class = 0.1 or 10% of the number of samples from majority class
import imblearn
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=55)
sm_ss = SMOTE(random_state=55,sampling_strategy=0.1)
X_train_rspl, y_train_rspl = sm.fit_sample (X_train,y_train)
X_train_rspl_ss, y_train_rspl_ss = sm_ss.fit_sample (X_train,y_train)
print(imblearn.__version__)


print('X_train.shape is ', X_train.shape)
print('X_train_rspl.shape is ', X_train_rspl.shape)
print('X_train_rspl_ss.shape is ', X_train_rspl_ss.shape)
print('X_test.shape is ', X_test.shape)
print('y_train.shape is ', y_train.shape)
print('y_train_rspl.shape is ', y_train_rspl.shape)
print('y_train_rspl_ss.shape is ', y_train_rspl_ss.shape)
print('y_test.shape is ', y_test.shape)


unique, count = np.unique(y_train, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
y_train_dict_value_count

unique, count = np.unique(y_train_rspl, return_counts=True)
y_train_smote_value_count = {k:v for (k,v) in zip(unique,count)}
y_train_smote_value_count

unique, count = np.unique(y_train_rspl_ss, return_counts=True)
y_train_smote_ss_value_count = {k:v for (k,v) in zip(unique,count)}
y_train_smote_ss_value_count

unique, count = np.unique(y_test, return_counts=True)
y_test_dict_value_count = {k:v for (k,v) in zip(unique,count)}
y_test_dict_value_count

# ### NN with SMOTE

# +
scaler = StandardScaler()  
scaler.fit(X_train_rspl)

X_train_rspl = scaler.transform(X_train_rspl)  
X_test = scaler.transform(X_test)  
# -

from sklearn.neural_network import MLPClassifier  
mlp_enhanced = MLPClassifier(hidden_layer_sizes=(32,32), max_iter=1000,verbose=2)  
mlp_enhanced.fit(X_train_rspl, y_train_rspl)  

from sklearn.neural_network import MLPClassifier  
mlp_enhanced_ss = MLPClassifier(hidden_layer_sizes=(32,32), max_iter=1000,verbose=2)  
mlp_enhanced_ss.fit(X_train_rspl_ss, y_train_rspl_ss)  

# +
# SMOTE with default sampling strategy = 1.0
#from sklearn.metrics import classification_report, confusion_matrix 
predictions = mlp_enhanced.predict(X_test)  

print("Accuracy for the NN with smote default sampling_strategy 1.0: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  

print("Accuracy on training set: {:.3f}".format(mlp_enhanced.score(X_train_rspl, y_train_rspl)))
print("Accuracy on test set: {:.3f}".format(mlp_enhanced.score(X_test, y_test)))

# +
# SMOTE with sampling strategy = 0.1
#from sklearn.metrics import classification_report, confusion_matrix 
predictions = mlp_enhanced_ss.predict(X_test)  

print("Accuracy for the enhanced model: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  

print("Accuracy on training set: {:.3f}".format(mlp_enhanced.score(X_train_rspl_ss, y_train_rspl_ss)))
print("Accuracy on test set: {:.3f}".format(mlp_enhanced.score(X_test, y_test)))
# -

# ### Decision Tree with SMOTE

# +
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy',random_state=0)
dt_ss = DecisionTreeClassifier(criterion='entropy',random_state=0)
dt.fit(X_train_rspl, y_train_rspl)
dt_ss.fit(X_train_rspl_ss, y_train_rspl_ss)

#below is with smote on default sampling strategy
print("Accuracy on training set for Decision Tree with smote on default sampling strategy: {:.3f}".format(dt.score(X_train_rspl, y_train_rspl)))
print("Accuracy on test set for Decison Tree with smote on default sampling strategy: {:.3f}".format(dt.score(X_test, y_test)))

#below is with smote on default sampling strategy
print("Accuracy on training set for Decision Tree with smote on 0.1 sampling strategy: {:.3f}".format(dt_ss.score(X_train_rspl_ss, y_train_rspl_ss)))
print("Accuracy on test set for Decison Tree with smote on 0.1 sampling strategy: {:.3f}".format(dt_ss.score(X_test, y_test)))


# -

y_pred = dt.predict(X_test)
y_pred_ss = dt_ss.predict(X_test)

# +
# Confusion Matrix on smote with default sampling strategy
from sklearn.metrics import classification_report, confusion_matrix  

print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
# +
# Confusion Matrix on smote with sampling strategy = 0.1
from sklearn.metrics import classification_report, confusion_matrix  

print(confusion_matrix(y_test, y_pred_ss))  
print(classification_report(y_test, y_pred_ss)) 
# -

# ## Logistic Regression with SMOTE and tuned ratio

# this is on smote with default sampling strategy = 1.0
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=0.01).fit(X_train_rspl, y_train_rspl)
print("Training set score: {:.3f}".format(logreg.score(X_train_rspl, y_train_rspl)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

# this is on smote with sampling strategy = 0.1
from sklearn.linear_model import LogisticRegression
logreg_ss = LogisticRegression(C=0.01).fit(X_train_rspl_ss, y_train_rspl_ss)
print("Training set score: {:.3f}".format(logreg_ss.score(X_train_rspl_ss, y_train_rspl_ss)))
print("Test set score: {:.3f}".format(logreg_ss.score(X_test, y_test)))

y_pred = logreg.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

y_pred_ss = logreg_ss.predict(X_test)
print(confusion_matrix(y_test, y_pred_ss))  
print(classification_report(y_test, y_pred_ss)) 

# +
# this is to search for the best ratio/sampling_strategy to be applied to SMOTE for better result
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(
    SMOTE(),
    LogisticRegression()
)

weights = np.linspace(0.005, 0.25, 10)

gsc = GridSearchCV(
    estimator=pipe,
    param_grid={
        'sampling_strategy': weights
    },
    scoring='f1',
    cv=3
)
grid_result = gsc.fit(X_train_rspl, y_train_rspl)

print("Best parameters : %s" % grid_result.best_params_)
weight_f1_score_df = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],
                                   'weight': weights })
weight_f1_score_df.plot(x='weight')
# -



