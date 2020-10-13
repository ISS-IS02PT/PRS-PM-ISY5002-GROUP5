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

# # Parkway Project Use Case 1: Write Off Cases Prediction

# ## DATA PREPARATION

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
# drop unnecessary columns
col_drop = ['PAYER_NAME_1', 'PAYER_NAME_2', 'PAYER_NAME_3', 'PAYER_NAME_4', 'PAYER_NAME_5', 'CASE_TYPE',
            'DISCHARGE_TYPE_DESC', 'DOCTOR_NAME', 'SPECIALTY_DESC','TOSP_STRING', 'TOSP_DESC1', 'TOSP_DESC2',
            'TOSP_DESC3', 'TOSP_DESC4', 'DRG_DESC', 'PAYER_CODE1_V', 'PAYER_NAME1_V', 'PAYER_CODE2_V',
            'PAYER_NAME2_V', 'PAYER_CODE3_V', 'PAYER_NAME3_V', 'PAYER_CODE4_V', 'PAYER_NAME4_V',
            'PACKAGE_DESC', 'PACKAGE_DESC1', 'PACKAGE_DESC2','ICDCODE_STRING', 'PACKAGE_CODE',
            'PACKAGE_PRICE', 'PACKAGE_EXCL', 'PACKAGE_ADJ', 'PACKAGE_CODE1', 'PACKAGE_CODE2','WRITE_OFF',
            'PCT_WRITE_OFF','PROF_FEE','TOTAL_FEES','TOTAL_PAID_AMT','PAYER_1_PAID_AMT','PAYER_2_PAID_AMT',
            'PAYER_3_PAID_AMT','PAYER_4_PAID_AMT','PAYER_5_PAID_AMT','PATIENT_SID','PATIENT_NUMBER','LANGUAGE','DRG_CODE']
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


# Checking df1 columns
info = []
for col in df1.columns:
    non_null  = len(df1) - np.sum(pd.isna(df1[col]))
    num_unique = df1[col].nunique()
    col_type = str(df1[col].dtype)

    info.append([col, non_null, num_unique, col_type])

features_info = pd.DataFrame(info, columns = ['colName','non-null values', 'unique', 'dtype'])

display(features_info)
# -

print(df['CASE_NUMBER'].nunique())

print(df[df.duplicated(subset=['CASE_NUMBER'], keep=False)])

df1.hist(figsize=(15,20), layout=(-1,5))
plt.tight_layout()
plt.show()

print(df1)

# ### Feature Engineering

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

df1.to_csv('PARKWAY_PROCESSED_4_NAN.csv')

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

X_train_scaled = scaler.transform(X_train)  
X_test_scaled = scaler.transform(X_test) 
# -

#type(X_train)
np.where(np.isnan(X_train))

# ## DATA PREPARATION WITH SMOTE (Synthetic Minority Over Sampling Technique)
# ## UNSCALED FOR DECISION TREE MODELS

# +
# Created few instances of smote here
# sm_ss_10 = normal smote with default sampling strategy = 1.0 -> this will create synthetic data for minority class such that the
#   number of samples for minority class equals with samples for majority class
# sm_ss_01 = smote with sampling strategy = 0.1
# sm_ss_02 = smote with sampling strategy = 0.2
# sm_ss_03 = smote with sampling strategy = 0.3 etc
# this will create synthetic data for minority class such that the number of samples for minority class = 0.1,0.2, 0.3 or 10%,20%,30% of the number of samples from majority class
import imblearn
from imblearn.over_sampling import SMOTE
print(imblearn.__version__)

sm_ss_01 = SMOTE(random_state=55,sampling_strategy=0.1)
sm_ss_02 = SMOTE(random_state=55,sampling_strategy=0.2)
sm_ss_03 = SMOTE(random_state=55,sampling_strategy=0.3)
sm_ss_04 = SMOTE(random_state=55,sampling_strategy=0.4)
sm_ss_05 = SMOTE(random_state=55,sampling_strategy=0.5)
sm_ss_06 = SMOTE(random_state=55,sampling_strategy=0.6)
sm_ss_07 = SMOTE(random_state=55,sampling_strategy=0.7)
sm_ss_08 = SMOTE(random_state=55,sampling_strategy=0.8)
sm_ss_09 = SMOTE(random_state=55,sampling_strategy=0.9)
sm_ss_10 = SMOTE(random_state=55) # default sampling strategy is 1.0


# Preparing Unscaled Training Data with Smote
X_train_unscaled_ss_01, y_train_unscaled_ss_01 = sm_ss_01.fit_sample(X_train,y_train)
X_train_unscaled_ss_02, y_train_unscaled_ss_02 = sm_ss_02.fit_sample(X_train,y_train)
X_train_unscaled_ss_03, y_train_unscaled_ss_03 = sm_ss_03.fit_sample(X_train,y_train)
X_train_unscaled_ss_04, y_train_unscaled_ss_04 = sm_ss_04.fit_sample(X_train,y_train)
X_train_unscaled_ss_05, y_train_unscaled_ss_05 = sm_ss_05.fit_sample(X_train,y_train)
X_train_unscaled_ss_06, y_train_unscaled_ss_06 = sm_ss_06.fit_sample(X_train,y_train)
X_train_unscaled_ss_07, y_train_unscaled_ss_07 = sm_ss_07.fit_sample(X_train,y_train)
X_train_unscaled_ss_08, y_train_unscaled_ss_08 = sm_ss_08.fit_sample(X_train,y_train)
X_train_unscaled_ss_09, y_train_unscaled_ss_09 = sm_ss_09.fit_sample(X_train,y_train)
X_train_unscaled_ss_10, y_train_unscaled_ss_10 = sm_ss_10.fit_sample(X_train,y_train)

# Checking Data Shape
print('X_train.shape is ', X_train.shape)
print('X_train_unscaled_ss_01.shape is ', X_train_unscaled_ss_01.shape)
print('y_train_unscaled_ss_01.shape is ', y_train_unscaled_ss_01.shape)
print('X_train_unscaled_ss_02.shape is ', X_train_unscaled_ss_02.shape)
print('y_train_unscaled_ss_02.shape is ', y_train_unscaled_ss_02.shape)
print('X_train_unscaled_ss_03.shape is ', X_train_unscaled_ss_03.shape)
print('y_train_unscaled_ss_03.shape is ', y_train_unscaled_ss_03.shape)
print('X_train_unscaled_ss_04.shape is ', X_train_unscaled_ss_04.shape)
print('y_train_unscaled_ss_04.shape is ', y_train_unscaled_ss_04.shape)
print('X_train_unscaled_ss_05.shape is ', X_train_unscaled_ss_05.shape)
print('y_train_unscaled_ss_05.shape is ', y_train_unscaled_ss_05.shape)
print('X_train_unscaled_ss_06.shape is ', X_train_unscaled_ss_06.shape)
print('y_train_unscaled_ss_06.shape is ', y_train_unscaled_ss_06.shape)
print('X_train_unscaled_ss_07.shape is ', X_train_unscaled_ss_07.shape)
print('y_train_unscaled_ss_07.shape is ', y_train_unscaled_ss_07.shape)
print('X_train_unscaled_ss_08.shape is ', X_train_unscaled_ss_08.shape)
print('y_train_unscaled_ss_08.shape is ', y_train_unscaled_ss_08.shape)
print('X_train_unscaled_ss_09.shape is ', X_train_unscaled_ss_09.shape)
print('y_train_unscaled_ss_09.shape is ', y_train_unscaled_ss_09.shape)
print('X_train_unscaled_ss_10.shape is ', X_train_unscaled_ss_10.shape)
print('y_train_unscaled_ss_10.shape is ', y_train_unscaled_ss_10.shape)

print('X_test.shape is ', X_test.shape)
print('y_test.shape is ', y_test.shape)



# +
# Checking Unique Values in y_train
unique, count = np.unique(y_train, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train unique values ',y_train_dict_value_count)

unique, count = np.unique(y_train_unscaled_ss_01, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train_unscaled_ss_01 unique values ',y_train_dict_value_count)

unique, count = np.unique(y_train_unscaled_ss_02, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train_unscaled_ss_02 unique values ',y_train_dict_value_count)

unique, count = np.unique(y_train_unscaled_ss_03, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train_unscaled_ss_03 unique values ',y_train_dict_value_count)

unique, count = np.unique(y_train_unscaled_ss_04, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train_unscaled_ss_04 unique values ',y_train_dict_value_count)

unique, count = np.unique(y_train_unscaled_ss_05, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train_unscaled_ss_05 unique values ',y_train_dict_value_count)

unique, count = np.unique(y_train_unscaled_ss_06, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train_unscaled_ss_06 unique values ',y_train_dict_value_count)

unique, count = np.unique(y_train_unscaled_ss_07, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train_unscaled_ss_07 unique values ',y_train_dict_value_count)

unique, count = np.unique(y_train_unscaled_ss_08, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train_unscaled_ss_08 unique values ',y_train_dict_value_count)

unique, count = np.unique(y_train_unscaled_ss_09, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train_unscaled_ss_09 unique values ',y_train_dict_value_count)

unique, count = np.unique(y_train_unscaled_ss_10, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train_unscaled_ss_10 unique values ',y_train_dict_value_count)

# -

# ## DECISION TREE

# ### Decision Tree WITHOUT SMOTE

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

dotfile = open("dt_unscaled_NoSMOTE.dot", 'w')

export_graphviz(dt, out_file=dotfile,feature_names = X.columns,class_names=['0','1'])
dotfile.close()

# +
#DT visualizatin method 2
# need to install Graphviz first https://graphviz.gitlab.io/_pages/Download/Download_windows.html
from sklearn.tree import export_graphviz
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

export_graphviz(dt, out_file='tree.dot', feature_names=X.columns,class_names=['0','1'])
# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'ParkwayWriteOff_Dectree_Unscaled_NoSmote.png', '-Gdpi=600'])

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

# ### Decision Tree WITH SMOTE

# #### Decision Tree WITH SMOTE SS = 0.1

dt.fit(X_train_unscaled_ss_01, y_train_unscaled_ss_01)
print("Accuracy on training set: {:.3f}".format(dt.score(X_train_unscaled_ss_01, y_train_unscaled_ss_01)))
print("Accuracy on test set: {:.3f}".format(dt.score(X_test, y_test)))
y_pred = dt.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# #### Decision Tree WITH SMOTE SS = 0.2

dt.fit(X_train_unscaled_ss_02, y_train_unscaled_ss_02)
print("Accuracy on training set: {:.3f}".format(dt.score(X_train_unscaled_ss_02, y_train_unscaled_ss_02)))
print("Accuracy on test set: {:.3f}".format(dt.score(X_test, y_test)))
y_pred = dt.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# #### Decision Tree WITH SMOTE SS = 0.3

dt.fit(X_train_unscaled_ss_03, y_train_unscaled_ss_03)
print("Accuracy on training set: {:.3f}".format(dt.score(X_train_unscaled_ss_03, y_train_unscaled_ss_03)))
print("Accuracy on test set: {:.3f}".format(dt.score(X_test, y_test)))
y_pred = dt.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# #### Decision Tree WITH SMOTE SS = 0.4

dt.fit(X_train_unscaled_ss_04, y_train_unscaled_ss_04)
print("Accuracy on training set: {:.3f}".format(dt.score(X_train_unscaled_ss_04, y_train_unscaled_ss_04)))
print("Accuracy on test set: {:.3f}".format(dt.score(X_test, y_test)))
y_pred = dt.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# #### Decision Tree WITH SMOTE SS = 0.5

dt.fit(X_train_unscaled_ss_05, y_train_unscaled_ss_05)
print("Accuracy on training set: {:.3f}".format(dt.score(X_train_unscaled_ss_05, y_train_unscaled_ss_05)))
print("Accuracy on test set: {:.3f}".format(dt.score(X_test, y_test)))
y_pred = dt.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# #### Decision Tree WITH SMOTE SS = 0.6

dt.fit(X_train_unscaled_ss_06, y_train_unscaled_ss_06)
print("Accuracy on training set: {:.3f}".format(dt.score(X_train_unscaled_ss_06, y_train_unscaled_ss_06)))
print("Accuracy on test set: {:.3f}".format(dt.score(X_test, y_test)))
y_pred = dt.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# #### Decision Tree WITH SMOTE SS = 0.7

dt.fit(X_train_unscaled_ss_07, y_train_unscaled_ss_07)
print("Accuracy on training set: {:.3f}".format(dt.score(X_train_unscaled_ss_07, y_train_unscaled_ss_07)))
print("Accuracy on test set: {:.3f}".format(dt.score(X_test, y_test)))
y_pred = dt.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# #### Decision Tree WITH SMOTE SS = 0.8

dt.fit(X_train_unscaled_ss_08, y_train_unscaled_ss_08)
print("Accuracy on training set: {:.3f}".format(dt.score(X_train_unscaled_ss_08, y_train_unscaled_ss_08)))
print("Accuracy on test set: {:.3f}".format(dt.score(X_test, y_test)))
y_pred = dt.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# #### Decision Tree WITH SMOTE SS = 0.9

dt.fit(X_train_unscaled_ss_09, y_train_unscaled_ss_09)
print("Accuracy on training set: {:.3f}".format(dt.score(X_train_unscaled_ss_09, y_train_unscaled_ss_09)))
print("Accuracy on test set: {:.3f}".format(dt.score(X_test, y_test)))
y_pred = dt.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# #### Decision Tree WITH SMOTE SS = 1.0

dt.fit(X_train_unscaled_ss_10, y_train_unscaled_ss_10)
print("Accuracy on training set: {:.3f}".format(dt.score(X_train_unscaled_ss_10, y_train_unscaled_ss_10)))
print("Accuracy on test set: {:.3f}".format(dt.score(X_test, y_test)))
y_pred = dt.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# ## DATA PREPARATION WITH SMOTE (Synthetic Minority Over Sampling Technique)
# ## SCALED FOR OTHER MODELS

# +
sm_ss_01 = SMOTE(random_state=55,sampling_strategy=0.1)
sm_ss_02 = SMOTE(random_state=55,sampling_strategy=0.2)
sm_ss_03 = SMOTE(random_state=55,sampling_strategy=0.3)
sm_ss_04 = SMOTE(random_state=55,sampling_strategy=0.4)
sm_ss_05 = SMOTE(random_state=55,sampling_strategy=0.5)
sm_ss_06 = SMOTE(random_state=55,sampling_strategy=0.6)
sm_ss_07 = SMOTE(random_state=55,sampling_strategy=0.7)
sm_ss_08 = SMOTE(random_state=55,sampling_strategy=0.8)
sm_ss_09 = SMOTE(random_state=55,sampling_strategy=0.9)
sm_ss_10 = SMOTE(random_state=55) # default sampling strategy is 1.0


# Preparing Scaled Training Data with Smote
X_train_scaled_ss_01, y_train_scaled_ss_01 = sm_ss_01.fit_sample(X_train_scaled,y_train)
X_train_scaled_ss_02, y_train_scaled_ss_02 = sm_ss_02.fit_sample(X_train_scaled,y_train)
X_train_scaled_ss_03, y_train_scaled_ss_03 = sm_ss_03.fit_sample(X_train_scaled,y_train)
X_train_scaled_ss_04, y_train_scaled_ss_04 = sm_ss_04.fit_sample(X_train_scaled,y_train)
X_train_scaled_ss_05, y_train_scaled_ss_05 = sm_ss_05.fit_sample(X_train_scaled,y_train)
X_train_scaled_ss_06, y_train_scaled_ss_06 = sm_ss_06.fit_sample(X_train_scaled,y_train)
X_train_scaled_ss_07, y_train_scaled_ss_07 = sm_ss_07.fit_sample(X_train_scaled,y_train)
X_train_scaled_ss_08, y_train_scaled_ss_08 = sm_ss_08.fit_sample(X_train_scaled,y_train)
X_train_scaled_ss_09, y_train_scaled_ss_09 = sm_ss_09.fit_sample(X_train_scaled,y_train)
X_train_scaled_ss_10, y_train_scaled_ss_10 = sm_ss_10.fit_sample(X_train_scaled,y_train)

# Checking Data Shape
print('X_train.shape is ', X_train.shape)
print('X_train_scaled_ss_01.shape is ', X_train_scaled_ss_01.shape)
print('y_train_scaled_ss_01.shape is ', y_train_scaled_ss_01.shape)
print('X_train_scaled_ss_02.shape is ', X_train_scaled_ss_02.shape)
print('y_train_scaled_ss_02.shape is ', y_train_scaled_ss_02.shape)
print('X_train_scaled_ss_03.shape is ', X_train_scaled_ss_03.shape)
print('y_train_scaled_ss_03.shape is ', y_train_scaled_ss_03.shape)
print('X_train_scaled_ss_04.shape is ', X_train_scaled_ss_04.shape)
print('y_train_scaled_ss_04.shape is ', y_train_scaled_ss_04.shape)
print('X_train_scaled_ss_05.shape is ', X_train_scaled_ss_05.shape)
print('y_train_scaled_ss_05.shape is ', y_train_scaled_ss_05.shape)
print('X_train_scaled_ss_06.shape is ', X_train_scaled_ss_06.shape)
print('y_train_scaled_ss_06.shape is ', y_train_scaled_ss_06.shape)
print('X_train_scaled_ss_07.shape is ', X_train_scaled_ss_07.shape)
print('y_train_scaled_ss_07.shape is ', y_train_scaled_ss_07.shape)
print('X_train_scaled_ss_08.shape is ', X_train_scaled_ss_08.shape)
print('y_train_scaled_ss_08.shape is ', y_train_scaled_ss_08.shape)
print('X_train_scaled_ss_09.shape is ', X_train_scaled_ss_09.shape)
print('y_train_scaled_ss_09.shape is ', y_train_scaled_ss_09.shape)
print('X_train_scaled_ss_10.shape is ', X_train_scaled_ss_10.shape)
print('y_train_scaled_ss_10.shape is ', y_train_scaled_ss_10.shape)

print('X_test.shape is ', X_test.shape)
print('y_test.shape is ', y_test.shape)


# +
# Checking Unique Values in y_train
unique, count = np.unique(y_train, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train unique values ',y_train_dict_value_count)

unique, count = np.unique(y_train_scaled_ss_01, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train_scaled_ss_01 unique values ',y_train_dict_value_count)

unique, count = np.unique(y_train_scaled_ss_02, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train_scaled_ss_02 unique values ',y_train_dict_value_count)

unique, count = np.unique(y_train_scaled_ss_03, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train_scaled_ss_03 unique values ',y_train_dict_value_count)

unique, count = np.unique(y_train_scaled_ss_04, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train_scaled_ss_04 unique values ',y_train_dict_value_count)

unique, count = np.unique(y_train_scaled_ss_05, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train_scaled_ss_05 unique values ',y_train_dict_value_count)

unique, count = np.unique(y_train_scaled_ss_06, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train_scaled_ss_06 unique values ',y_train_dict_value_count)

unique, count = np.unique(y_train_scaled_ss_07, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train_scaled_ss_07 unique values ',y_train_dict_value_count)

unique, count = np.unique(y_train_scaled_ss_08, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train_scaled_ss_08 unique values ',y_train_dict_value_count)

unique, count = np.unique(y_train_scaled_ss_09, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train_scaled_ss_09 unique values ',y_train_dict_value_count)

unique, count = np.unique(y_train_scaled_ss_10, return_counts=True)
y_train_dict_value_count = {k:v for (k,v) in zip(unique,count)}
print('y_train_scaled_ss_10 unique values ',y_train_dict_value_count)
# -

# ## LOGISTIC REGRESSION

# ### Logistic Regression Without SMOTE

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=0.01).fit(X_train_scaled, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test_scaled, y_test)))
y_pred = logreg.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# ### Logistic Regression With SMOTE

# #### Logistic Regression With SMOTE ss=0.1

logreg = LogisticRegression(C=0.01).fit(X_train_scaled_ss_01, y_train_scaled_ss_01)
print("Training set score: {:.3f}".format(logreg.score(X_train_scaled_ss_01, y_train_scaled_ss_01)))
print("Test set score: {:.3f}".format(logreg.score(X_test_scaled, y_test)))
y_pred = logreg.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# #### Logistic Regression With SMOTE ss=0.2

logreg = LogisticRegression(C=0.01).fit(X_train_scaled_ss_02, y_train_scaled_ss_02)
print("Training set score: {:.3f}".format(logreg.score(X_train_scaled_ss_02, y_train_scaled_ss_02)))
print("Test set score: {:.3f}".format(logreg.score(X_test_scaled, y_test)))
y_pred = logreg.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# #### Logistic Regression With SMOTE ss=0.3

logreg = LogisticRegression(C=0.01).fit(X_train_scaled_ss_03, y_train_scaled_ss_03)
print("Training set score: {:.3f}".format(logreg.score(X_train_scaled_ss_03, y_train_scaled_ss_03)))
print("Test set score: {:.3f}".format(logreg.score(X_test_scaled, y_test)))
y_pred = logreg.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# #### Logistic Regression With SMOTE ss=0.4

logreg = LogisticRegression(C=0.01).fit(X_train_scaled_ss_04, y_train_scaled_ss_04)
print("Training set score: {:.3f}".format(logreg.score(X_train_scaled_ss_04, y_train_scaled_ss_04)))
print("Test set score: {:.3f}".format(logreg.score(X_test_scaled, y_test)))
y_pred = logreg.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# #### Logistic Regression With SMOTE ss=0.5

logreg = LogisticRegression(C=0.01).fit(X_train_scaled_ss_05, y_train_scaled_ss_05)
print("Training set score: {:.3f}".format(logreg.score(X_train_scaled_ss_05, y_train_scaled_ss_05)))
print("Test set score: {:.3f}".format(logreg.score(X_test_scaled, y_test)))
y_pred = logreg.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# #### Logistic Regression With SMOTE ss=0.6

logreg = LogisticRegression(C=0.01).fit(X_train_scaled_ss_06, y_train_scaled_ss_06)
print("Training set score: {:.3f}".format(logreg.score(X_train_scaled_ss_06, y_train_scaled_ss_06)))
print("Test set score: {:.3f}".format(logreg.score(X_test_scaled, y_test)))
y_pred = logreg.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# #### Logistic Regression With SMOTE ss=0.7

logreg = LogisticRegression(C=0.01).fit(X_train_scaled_ss_07, y_train_scaled_ss_07)
print("Training set score: {:.3f}".format(logreg.score(X_train_scaled_ss_07, y_train_scaled_ss_07)))
print("Test set score: {:.3f}".format(logreg.score(X_test_scaled, y_test)))
y_pred = logreg.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# #### Logistic Regression With SMOTE ss=0.8

logreg = LogisticRegression(C=0.01).fit(X_train_scaled_ss_08, y_train_scaled_ss_08)
print("Training set score: {:.3f}".format(logreg.score(X_train_scaled_ss_08, y_train_scaled_ss_08)))
print("Test set score: {:.3f}".format(logreg.score(X_test_scaled, y_test)))
y_pred = logreg.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# #### Logistic Regression With SMOTE ss=0.9

logreg = LogisticRegression(C=0.01).fit(X_train_scaled_ss_09, y_train_scaled_ss_09)
print("Training set score: {:.3f}".format(logreg.score(X_train_scaled_ss_09, y_train_scaled_ss_09)))
print("Test set score: {:.3f}".format(logreg.score(X_test_scaled, y_test)))
y_pred = logreg.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# #### Logistic Regression With SMOTE ss=1.0

logreg = LogisticRegression(C=0.01).fit(X_train_scaled_ss_10, y_train_scaled_ss_10)
print("Training set score: {:.3f}".format(logreg.score(X_train_scaled_ss_10, y_train_scaled_ss_10)))
print("Test set score: {:.3f}".format(logreg.score(X_test_scaled, y_test)))
y_pred = logreg.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# ## NAIVE BAYES

# ### Naive Bayes Without SMOTE

# +
from sklearn.naive_bayes import GaussianNB
# Initiating the Gaussian Classifier
mod = GaussianNB()

# Training your model 
mod.fit(X_train_scaled, y_train)

# Predicting Outcome 
predicted = mod.predict(X_test_scaled)

# Score
print("Training set score: {:.3f}".format(mod.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}".format(mod.score(X_test_scaled, y_test)))

# Confusion Matrix
y_pred = mod.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
# -

# ### Naive Bayes With SMOTE

# #### Naive Bayes Without SMOTE ss=0.1

# +
# Training your model 
mod.fit(X_train_scaled_ss_01, y_train_scaled_ss_01)

# Predicting Outcome 
predicted = mod.predict(X_test_scaled)

# Score
print("Training set score: {:.3f}".format(mod.score(X_train_scaled_ss_01, y_train_scaled_ss_01)))
print("Test set score: {:.3f}".format(mod.score(X_test_scaled, y_test)))

# Confusion Matrix
y_pred = mod.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
# -

# #### Naive Bayes Without SMOTE ss=0.2

# +
# Training your model 
mod.fit(X_train_scaled_ss_02, y_train_scaled_ss_02)

# Predicting Outcome 
predicted = mod.predict(X_test_scaled)

# Score
print("Training set score: {:.3f}".format(mod.score(X_train_scaled_ss_02, y_train_scaled_ss_02)))
print("Test set score: {:.3f}".format(mod.score(X_test_scaled, y_test)))

# Confusion Matrix
y_pred = mod.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
# -

# #### Naive Bayes Without SMOTE ss=0.3

# +
# Training your model 
mod.fit(X_train_scaled_ss_03, y_train_scaled_ss_03)

# Predicting Outcome 
predicted = mod.predict(X_test_scaled)

# Score
print("Training set score: {:.3f}".format(mod.score(X_train_scaled_ss_03, y_train_scaled_ss_03)))
print("Test set score: {:.3f}".format(mod.score(X_test_scaled, y_test)))

# Confusion Matrix
y_pred = mod.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
# -

# #### Naive Bayes Without SMOTE ss=0.4

# +
# Training your model 
mod.fit(X_train_scaled_ss_04, y_train_scaled_ss_04)

# Predicting Outcome 
predicted = mod.predict(X_test_scaled)

# Score
print("Training set score: {:.3f}".format(mod.score(X_train_scaled_ss_04, y_train_scaled_ss_04)))
print("Test set score: {:.3f}".format(mod.score(X_test_scaled, y_test)))

# Confusion Matrix
y_pred = mod.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
# -

# #### Naive Bayes Without SMOTE ss=0.5

# +
# Training your model 
mod.fit(X_train_scaled_ss_05, y_train_scaled_ss_05)

# Predicting Outcome 
predicted = mod.predict(X_test_scaled)

# Score
print("Training set score: {:.3f}".format(mod.score(X_train_scaled_ss_05, y_train_scaled_ss_05)))
print("Test set score: {:.3f}".format(mod.score(X_test_scaled, y_test)))

# Confusion Matrix
y_pred = mod.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
# -

# #### Naive Bayes Without SMOTE ss=0.6

# +
# Training your model 
mod.fit(X_train_scaled_ss_06, y_train_scaled_ss_06)

# Predicting Outcome 
predicted = mod.predict(X_test_scaled)

# Score
print("Training set score: {:.3f}".format(mod.score(X_train_scaled_ss_06, y_train_scaled_ss_06)))
print("Test set score: {:.3f}".format(mod.score(X_test_scaled, y_test)))

# Confusion Matrix
y_pred = mod.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
# -

# #### Naive Bayes Without SMOTE ss=0.7

# +
# Training your model 
mod.fit(X_train_scaled_ss_07, y_train_scaled_ss_07)

# Predicting Outcome 
predicted = mod.predict(X_test_scaled)

# Score
print("Training set score: {:.3f}".format(mod.score(X_train_scaled_ss_07, y_train_scaled_ss_07)))
print("Test set score: {:.3f}".format(mod.score(X_test_scaled, y_test)))

# Confusion Matrix
y_pred = mod.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
# -

# #### Naive Bayes Without SMOTE ss=0.8

# +
# Training your model 
mod.fit(X_train_scaled_ss_08, y_train_scaled_ss_08)

# Predicting Outcome 
predicted = mod.predict(X_test_scaled)

# Score
print("Training set score: {:.3f}".format(mod.score(X_train_scaled_ss_08, y_train_scaled_ss_08)))
print("Test set score: {:.3f}".format(mod.score(X_test_scaled, y_test)))

# Confusion Matrix
y_pred = mod.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
# -

# #### Naive Bayes Without SMOTE ss=0.9

# +
# Training your model 
mod.fit(X_train_scaled_ss_09, y_train_scaled_ss_09)

# Predicting Outcome 
predicted = mod.predict(X_test_scaled)

# Score
print("Training set score: {:.3f}".format(mod.score(X_train_scaled_ss_09, y_train_scaled_ss_09)))
print("Test set score: {:.3f}".format(mod.score(X_test_scaled, y_test)))

# Confusion Matrix
y_pred = mod.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
# -

# #### Naive Bayes Without SMOTE ss=1.0

# +
# Training your model 
mod.fit(X_train_scaled_ss_10, y_train_scaled_ss_10)

# Predicting Outcome 
predicted = mod.predict(X_test_scaled)

# Score
print("Training set score: {:.3f}".format(mod.score(X_train_scaled_ss_10, y_train_scaled_ss_10)))
print("Test set score: {:.3f}".format(mod.score(X_test_scaled, y_test)))

# Confusion Matrix
y_pred = mod.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# + [markdown] slideshow={"slide_type": "-"}
# ## NEURAL NET MLP
# -

# ### Neural Net MLP Without SMOTE, 2 Layers

# #### Neural Net MLPClassifier without SMOTE, hidden layer (16,16)

from sklearn.neural_network import MLPClassifier  
from sklearn import metrics
mlp = MLPClassifier(hidden_layer_sizes=(16,16), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled, y_train)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# #### Neural Net MLPClassifier without SMOTE, hidden layer (32,32)

mlp = MLPClassifier(hidden_layer_sizes=(32,32), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled, y_train)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# #### Neural Net MLPClassifier without SMOTE, hidden layer (64,64)

mlp = MLPClassifier(hidden_layer_sizes=(64,64), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled, y_train)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# #### Neural Net MLPClassifier without SMOTE, hidden layer (128,128)

mlp = MLPClassifier(hidden_layer_sizes=(128,128), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled, y_train)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# #### Neural Net MLPClassifier without SMOTE, hidden layer (256,256)

mlp = MLPClassifier(hidden_layer_sizes=(256,256), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled, y_train)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# ### Neural Net MLP Without SMOTE, 3 Layers

# #### Neural Net MLPClassifier without SMOTE, hidden layer (16,16,16)

mlp = MLPClassifier(hidden_layer_sizes=(16,16,16), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled, y_train)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# #### Neural Net MLPClassifier without SMOTE, hidden layer (32,32,32)

mlp = MLPClassifier(hidden_layer_sizes=(32,32,32), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled, y_train)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# #### Neural Net MLPClassifier without SMOTE, hidden layer (64,64,64)

mlp = MLPClassifier(hidden_layer_sizes=(64,64,64), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled, y_train)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# ### Neural Net MLP Without SMOTE, 4 Layers

# #### Neural Net MLPClassifier without SMOTE, hidden layer (16,16,16,16)

mlp = MLPClassifier(hidden_layer_sizes=(16,16,16,16), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled, y_train)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# #### Neural Net MLPClassifier without SMOTE, hidden layer (32,32,32,32)

mlp = MLPClassifier(hidden_layer_sizes=(32,32,32,32), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled, y_train)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# ### Neural Net With SMOTE

# #### Neural Net 2 Layers (128,128) With SMOTE ss=0.1

mlp = MLPClassifier(hidden_layer_sizes=(128,128), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled_ss_01, y_train_scaled_ss_01)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled_ss_01, y_train_scaled_ss_01)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# #### Neural Net 2 Layers (128,128) With SMOTE ss=0.2

mlp = MLPClassifier(hidden_layer_sizes=(128,128), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled_ss_02, y_train_scaled_ss_02)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled_ss_02, y_train_scaled_ss_02)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# #### Neural Net 2 Layers (128,128) With SMOTE ss=0.3

mlp = MLPClassifier(hidden_layer_sizes=(128,128), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled_ss_03, y_train_scaled_ss_03)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled_ss_03, y_train_scaled_ss_03)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# #### Neural Net 3 Layers (16,16,16) With SMOTE ss=0.1

mlp = MLPClassifier(hidden_layer_sizes=(16,16,16), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled_ss_01, y_train_scaled_ss_01)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled_ss_01, y_train_scaled_ss_01)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# #### Neural Net 3 Layers (16,16,16) With SMOTE ss=0.2

mlp = MLPClassifier(hidden_layer_sizes=(16,16,16), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled_ss_02, y_train_scaled_ss_02)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled_ss_02, y_train_scaled_ss_02)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# #### Neural Net 3 Layers (32,32,32) With SMOTE ss=0.1

mlp = MLPClassifier(hidden_layer_sizes=(32,32,32), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled_ss_01, y_train_scaled_ss_01)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled_ss_01, y_train_scaled_ss_01)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# #### Neural Net 3 Layers (32,32,32) With SMOTE ss=0.2

mlp = MLPClassifier(hidden_layer_sizes=(32,32,32), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled_ss_02, y_train_scaled_ss_02)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled_ss_02, y_train_scaled_ss_02)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# #### Neural Net With 4 Layers (16,16,16,16) SMOTE ss=0.1

mlp = MLPClassifier(hidden_layer_sizes=(16,16,16,16), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled_ss_01, y_train_scaled_ss_01)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled_ss_01, y_train_scaled_ss_01)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# #### Neural Net With 4 Layers (16,16,16,16) SMOTE ss=0.2

mlp = MLPClassifier(hidden_layer_sizes=(16,16,16,16), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled_ss_02, y_train_scaled_ss_02)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled_ss_02, y_train_scaled_ss_02)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# #### Neural Net With 4 Layers (32,32,32,32) SMOTE ss=0.1

mlp = MLPClassifier(hidden_layer_sizes=(32,32,32,32), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled_ss_01, y_train_scaled_ss_01)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled_ss_01, y_train_scaled_ss_01)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# #### Neural Net With SMOTE ss=1.0

# #### Neural Net With 4 Layers (32,32,32,32) SMOTE ss=0.2

mlp = MLPClassifier(hidden_layer_sizes=(32,32,32,32), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled_ss_02, y_train_scaled_ss_02)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled_ss_02, y_train_scaled_ss_02)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# #### Neural Net With 4 Layers (32,32,32,32) SMOTE ss=0.5

mlp = MLPClassifier(hidden_layer_sizes=(32,32,32,32), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled_ss_05, y_train_scaled_ss_05)  

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled_ss_05, y_train_scaled_ss_05)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# #### Neural Net With 4 Layers (32,32,32,32) SMOTE ss=1.0

mlp = MLPClassifier(hidden_layer_sizes=(32,32,32,32), max_iter=1000,verbose=2)  
mlp.fit(X_train_scaled_ss_10, y_train_scaled_ss_10) 

# +
predictions = mlp.predict(X_test_scaled) 

# Score
print("Training set score: {:.3f}".format(mlp.score(X_train_scaled_ss_10, y_train_scaled_ss_10)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
# -

# ## KNN

# +
# Be careful when running this, about 2 full days are needed to run this!!!
from sklearn.neighbors import KNeighborsClassifier

test_scores = []
train_scores = []

for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train_scaled,y_train)
    
    train_scores.append(knn.score(X_train_scaled,y_train))
    test_scores.append(knn.score(X_test_scaled,y_test))
# -

## score that comes from testing on the same datapoints that were used for training
max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))

## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))

plt.figure(figsize=(12,5))
plt.plot(range(1,15),train_scores,marker='*',label='Train Score')
plt.plot(range(1,15),test_scores,marker='o',label='Test Score')

from sklearn.neighbors import KNeighborsClassifier
# From the above, the best KNN is at k=8
#Setup a knn classifier with k neighbors
best_knn = KNeighborsClassifier(8)

# ### KNN without SMOTE

# +
# Fit the KNN model with training data and score the test data

best_knn.fit(X_train_scaled,y_train)
best_knn.score(X_test_scaled,y_test)
# -

import pickle
# save the model to disk
filename = 'best_knn_without_smote.sav'
pickle.dump(best_knn, open(filename, 'wb'))

# load the model from disk, NO NEED TO RUN IF NOT NECESSARY
knn_loaded_model = pickle.load(open(filename, 'rb'))
#result = knn_loaded_model.score(X_test, y_test)
#print(result)

# +
y_pred = best_knn.predict(X_test_scaled)
# Score for Best KNN, No SMOTE
print("Training set score: {:.3f}".format(best_knn.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}".format(best_knn.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
# -

# ### KNN With SMOTE

# #### KNN With SMOTE ss=0.1

# Fit the KNN model with training data and score the test data
best_knn.fit(X_train_scaled_ss_01,y_train_scaled_ss_01)
#best_knn.score(X_test_scaled,y_test)

# save the model to disk
filename = 'best_knn_with_smote_ss_01.sav'
pickle.dump(best_knn, open(filename, 'wb'))

# load the model from disk
knn_loaded_model = pickle.load(open(filename, 'rb'))
#result = knn_loaded_model.score(X_test, y_test)
#print(result)

# +
y_pred = best_knn.predict(X_test_scaled)
# Score for Best KNN, with SMOTE
print("Training set score: {:.3f}".format(best_knn.score(X_train_scaled_ss_01, y_train_scaled_ss_01)))
print("Test set score: {:.3f}".format(best_knn.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred)) 
# -

# #### KNN With SMOTE ss=0.5

# Fit the KNN model with training data and score the test data
best_knn.fit(X_train_scaled_ss_05,y_train_scaled_ss_05)
#best_knn.score(X_test_scaled,y_test)

# save the model to disk
filename = 'best_knn_with_smote_ss_05.sav'
pickle.dump(best_knn, open(filename, 'wb'))

# load the model from disk
knn_loaded_model = pickle.load(open(filename, 'rb'))
#result = knn_loaded_model.score(X_test, y_test)
#print(result)

# +
y_pred = best_knn.predict(X_test_scaled)
# Score for Best KNN, with SMOTE
print("Training set score: {:.3f}".format(best_knn.score(X_train_scaled_ss_05, y_train_scaled_ss_05)))
print("Test set score: {:.3f}".format(best_knn.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred)) 
# -

# #### KNN With SMOTE ss=1.0

# Fit the KNN model with training data and score the test data
best_knn.fit(X_train_scaled_ss_10,y_train_scaled_ss_10)
#best_knn.score(X_test_scaled,y_test)

# save the model to disk
filename = 'best_knn_with_smote_ss_10.sav'
pickle.dump(best_knn, open(filename, 'wb'))

# load the model from disk
knn_loaded_model = pickle.load(open(filename, 'rb'))
#result = knn_loaded_model.score(X_test, y_test)
#print(result)

# +
y_pred = best_knn.predict(X_test_scaled)
# Score for Best KNN, with SMOTE
print("Training set score: {:.3f}".format(best_knn.score(X_train_scaled_ss_10, y_train_scaled_ss_10)))
print("Test set score: {:.3f}".format(best_knn.score(X_test_scaled, y_test)))

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred)) 
# -


