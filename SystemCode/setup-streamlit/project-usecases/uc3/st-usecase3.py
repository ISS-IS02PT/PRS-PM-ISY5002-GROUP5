import streamlit as st
import SessionState
import pickle
import numpy as np
import pandas as pd
import time
from datetime import datetime,date

import constants_uc3
from datapipeline_uc3 import Datapipeline
from tensorflow.keras.models import load_model

def rerun():
  # print(str(datetime.now()) + " Re-running")
  raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))

custom_column_names = {
  'ACTUAL_CASE_NUMBER': 'CASE_NUMBER',
  'ACTUAL_TREATMENT_CATEGORY': 'TREATMENT_CATEGORY',
  'ACTUAL_PATIENT_NUMBER': 'PATIENT_NUMBER',
  'ACTUAL_PRIMARY_DIAGNOSIS_SID': 'PRIMARY_DIAGNOSIS_SID',
  'ACTUAL_CASE_INSTITUTION': 'CASE_INSTITUTION',
  'ACTUAL_ADMISSION_DTE': 'ADMISSION_DTE',
  'ACTUAL_LOS': 'LENGTH_OF_STAYS',
  'ACTUAL_ICU_HDU_LOS' : 'ICU_HDU_LOS',
  'ACTUAL_SURGICAL_CODE_1': 'SURGICAL_CODE_1',
  'ACTUAL_SURGICAL_CODE_2': 'SURGICAL_CODE_2',
  'ACTUAL_SURGICAL_CODE_3': 'SURGICAL_CODE_3',
  'ACTUAL_DIAGNOSIS_CODE': 'DIAGNOSIS_CODE',
  'ADM_DATE': 'ADM_DATE',
  'INST': 'INST',
  'CASE_NO': 'CASE_NO',
  'ETBS_LOS': 'ETBS_LOS',
  'ETBS_ICU_HDU_LOS': 'ETBS_ICU_HDU_LOS',
  'ETBS_MOVE_TYPE': 'ETBS_MOVE_TYPE',
  'ETBS_TOSP_1': 'ETBS_TOSP_1',
  'ETBS_TOSP_2': 'ETBS_TOSP_2',
  'ETBS_TOSP_3': 'ETBS_TOSP_3',
  'ETBS_ICD10_1': 'ETBS_ICD10_1',
  'ETBS_ICD10_2': 'ETBS_ICD10_2',
  'ETBS_ICD10_3': 'ETBS_ICD10_3',
  'CASE_TYPE': 'CASE_TYPE',
  'PATIENT_TYPE': 'PATIENT_TYPE',
  'ADMISSION_TYPE': 'ADMISSION_TYPE',
  'TREATMENT_CATEGORY': 'TREATMENT_CAT',
  'REFERRAL_TYPE': 'REFERRAL_TYPE',
  'DEPT_OU': 'DEPT_OU',
  'ADMITTING_SMC_NUMBER': 'ADMITTING_SMC_NUMBER',
  'ATTENDING_SMC_NUMBER': 'ATTENDING_SMC_NUMBER',
  'REFERRAL_SMC_NUMBER': 'REFERRAL_SMC_NUMBER',
  'ADM_CLASS_DISC': 'ADM_CLASS_DISC',
  'PATIENT_NUMBER': 'PATIENT_NUMBER',
  'GENDER': 'GENDER',
  'DOB': 'DOB',
  'MARITAL_STATUS': 'MARITAL_STATUS',
  'RELIGION': 'RELIGION',
  'NATIONALITY': 'NATIONALITY',
  'RESID_CTY': 'RESID_CTY',
  'RESID_POSTALCODE': 'RESID_POSTALCODE',
  'OCCUPATION': 'OCCUPATION',
  'RESID_GEOAREA': 'RESID_GEOAREA',
  'NONRESID_FLAG': 'NONRESID_FLAG',
  'IDENT_TYPE': 'IDENT_TYPE',
  'CONT_POSTAL': 'CONT_POSTAL',
  'CONT_RELATION': 'CONT_RELATION',
  'TOTAL_HOSP': 'TOTAL_HOSP',
}

col_dtypes = {
  'ACTUAL_CASE_NUMBER': np.int64,
  'ACTUAL_TREATMENT_CATEGORY': np.object,
  'ACTUAL_PATIENT_NUMBER': np.int64,
  'ACTUAL_PRIMARY_DIAGNOSIS_SID': np.int64,
  'ACTUAL_CASE_INSTITUTION':  np.object,
  'ACTUAL_ADMISSION_DTE':  np.datetime64,
  'ACTUAL_LOS':  np.int64,
  'ACTUAL_ICU_HDU_LOS' :  np.float64,
  'ACTUAL_SURGICAL_CODE_1':  np.object,
  'ACTUAL_SURGICAL_CODE_2':  np.object,
  'ACTUAL_SURGICAL_CODE_3':  np.object,
  'ACTUAL_DIAGNOSIS_CODE':  np.object,
  'ADM_DATE':  np.datetime64,
  'INST':  np.object,
  'CASE_NO':  np.float64,
  'ETBS_LOS':  np.object,
  'ETBS_ICU_HDU_LOS':  np.object,
  'ETBS_MOVE_TYPE':  np.object,
  'ETBS_TOSP_1':  np.object,
  'ETBS_TOSP_2':  np.object,
  'ETBS_TOSP_3':  np.object,
  'ETBS_ICD10_1':  np.object,
  'ETBS_ICD10_2':  np.object,
  'ETBS_ICD10_3':  np.object,
  'CASE_TYPE':  np.object,
  'PATIENT_TYPE':  np.float64,
  'ADMISSION_TYPE':  np.object,
  'TREATMENT_CATEGORY':  np.object,
  'REFERRAL_TYPE':  np.object,
  'DEPT_OU':  np.object,
  'ADMITTING_SMC_NUMBER':  np.object,
  'ATTENDING_SMC_NUMBER':  np.object,
  'REFERRAL_SMC_NUMBER':  np.object,
  'ADM_CLASS_DISC':  np.object,
  'PATIENT_NUMBER':  np.float64,
  'GENDER':  np.object,
  'DOB':  np.datetime64,
  'MARITAL_STATUS':  np.object,
  'RELIGION':  np.object,
  'NATIONALITY':  np.object,
  'RESID_CTY':  np.object,
  'RESID_POSTALCODE':  np.object,
  'OCCUPATION':  np.object,
  'RESID_GEOAREA':  np.object,
  'NONRESID_FLAG':  np.object,
  'IDENT_TYPE':  np.object,
  'CONT_POSTAL':  np.object,
  'CONT_RELATION':  np.object,
  'TOTAL_HOSP':  np.float64
}
  
columns_categories = {
  'ACTUAL_CASE_INSTITUTION': [x.upper() for x in constants_uc3.INSTITUTION_TYPES],
  'ACTUAL_TREATMENT_CATEGORY': [x.upper() for x in constants_uc3.TREATMENT_TYPES],
  'ACTUAL_SURGICAL_CODE_1': [x.upper() for x in constants_uc3.TOSP_CODES],
  'ACTUAL_SURGICAL_CODE_2': [x.upper() for x in constants_uc3.TOSP_CODES],
  'ACTUAL_SURGICAL_CODE_3': [x.upper() for x in constants_uc3.TOSP_CODES],
  'ACTUAL_DIAGNOSIS_CODE': [x.upper() for x in constants_uc3.ICD_CODES],
  'ETBS_MOVE_TYPE': [x.upper() for x in constants_uc3.ADMISSION_TYPES],
  'ETBS_TOSP_1': [x.upper() for x in constants_uc3.TOSP_CODES],
  'ETBS_TOSP_2': [x.upper() for x in constants_uc3.TOSP_CODES],
  'ETBS_TOSP_3': [x.upper() for x in constants_uc3.TOSP_CODES],
  'ETBS_ICD10_1': [x.upper() for x in constants_uc3.ICD_CODES],
  'ETBS_ICD10_2': [x.upper() for x in constants_uc3.ICD_CODES],
  'ETBS_ICD10_3': [x.upper() for x in constants_uc3.ICD_CODES],
  'ADMISSION_TYPE': [x.upper() for x in constants_uc3.ADMISSION_TYPES],
  'TREATMENT_CATEGORY': [x.upper() for x in constants_uc3.TREATMENT_TYPES],
  'REFERRAL_TYPE': [x.upper() for x in constants_uc3.REFERRAL_TYPES],
  'DEPT_OU': [x.upper() for x in constants_uc3.DEPT_OUS],
  'ADMITTING_SMC_NUMBER': [x.upper() for x in constants_uc3.DOCTOR_CODES],
  'ATTENDING_SMC_NUMBER': [x.upper() for x in constants_uc3.DOCTOR_CODES],
  'REFERRAL_SMC_NUMBER': [x.upper() for x in constants_uc3.DOCTOR_CODES],
  'ADM_CLASS_DISC': [x.upper() for x in constants_uc3.ADM_CLASSES],
  'GENDER': [x.upper() for x in constants_uc3.GENDER_TYPES],
  'MARITAL_STATUS': [x.upper() for x in constants_uc3.MARITAL_STATUSES],
  'RELIGION': [x.upper() for x in constants_uc3.RELIGION_TYPES],
  'NATIONALITY': [x.upper() for x in constants_uc3.NATIONALITIES],
  'OCCUPATION': [x.upper() for x in constants_uc3.OCCUPATIONS],
  'RESID_GEOAREA': [x.upper() for x in constants_uc3.RESID_LOCATIONS],
  'RESID_CTY': [x.upper() for x in constants_uc3.COUNTRY_CODES.keys()],       # This should be keys()
  'NONRESID_FLAG': ['N', 'Y'],
  'IDENT_TYPE': [x.upper() for x in constants_uc3.ID_TYPES],
  'CONT_RELATION': [x.upper() for x in constants_uc3.RELATIONS]
}
# Append Empty value to the category
for k in columns_categories.keys():
  columns_categories[k] = [""] + columns_categories[k]


# Session States
# For categorical field, it is the index
session_objects = {
  'ACTUAL_CASE_NUMBER': 1019003485,
  'ACTUAL_TREATMENT_CATEGORY': 0,
  'ACTUAL_PATIENT_NUMBER': 6114934,           # Drop anyway
  'ACTUAL_PRIMARY_DIAGNOSIS_SID': 27276,      # Drop anyway
  'ACTUAL_CASE_INSTITUTION': 0,
  'ACTUAL_ADMISSION_DTE': None,               # Default to today, Drop anyway
  'ACTUAL_LOS': 1,
  'ACTUAL_ICU_HDU_LOS': 0,
  'ACTUAL_SURGICAL_CODE_1': 0,
  'ACTUAL_SURGICAL_CODE_2': 0,
  'ACTUAL_SURGICAL_CODE_3': 0,
  'ACTUAL_DIAGNOSIS_CODE': 0,
  'ADM_DATE': None,                           # Auto-value, Drop anyway
  'INST': None,                               # Auto-value, Drop anyway
  'CASE_NO': None,                            # Auto-value, Drop anyway
  'ETBS_LOS': None,                           # Auto-value
  'ETBS_ICU_HDU_LOS': None,                   # Auto-value
  'ETBS_MOVE_TYPE': 0,
  'ETBS_TOSP_1': 0,
  'ETBS_TOSP_2': 0,
  'ETBS_TOSP_3': 0,
  'ETBS_ICD10_1': 0,
  'ETBS_ICD10_2': 0,
  'ETBS_ICD10_3': 0,
  'CASE_TYPE': None,                          # Drop anyway
  'PATIENT_TYPE': None,                       # Drop anyway
  'ADMISSION_TYPE': 0,
  'TREATMENT_CATEGORY': 0,
  'REFERRAL_TYPE': 0,
  'DEPT_OU': 0,
  'ADMITTING_SMC_NUMBER': 0,
  'ATTENDING_SMC_NUMBER': 0,
  'REFERRAL_SMC_NUMBER': 0,
  'ADM_CLASS_DISC': 0,
  'PATIENT_NUMBER':None,                      # Auto-value, Drop anyway
  'GENDER': 0,
  'DOB': None,                                # Default to today, Drop anyway
  'MARITAL_STATUS': 0,
  'RELIGION': 0,
  'NATIONALITY': 0,
  'RESID_CTY': 0,
  'RESID_POSTALCODE': None,                   # Drop anyway
  'OCCUPATION': 0,
  'RESID_GEOAREA': 0,
  'NONRESID_FLAG': 0,
  'IDENT_TYPE': 0,
  'CONT_POSTAL': None,                        # Drop anyway
  'CONT_RELATION': 0,
  'TOTAL_HOSP': None,                         # Drop anyway
  'dpl': None,
  'model': None,
}

session_state = SessionState.get1(session_objects)
# print(str(datetime.now()) + "-A")
RERUN = False


# Load models
scaler_pkl_file_path = f'./models/all_hosp_data_uc3_scaler.pkl'
ohe_pkl_file_path = f'./models/all_hosp_data_uc3_ohe.pkl'
feature_importance_file_path = f'./models/all_hosp_forest_feat_impt_uc3.npy'
model_file_path = f'./models/uc3_NN2_model.h5'
if session_state.session_objects_dict['dpl'] is None:
  st.write("Setting up the application ...")
  session_state.session_objects_dict['dpl'] = Datapipeline()
  session_state.session_objects_dict['model'] = load_model(model_file_path)



# RENDER FUNCTIONS
### Dropdown - selectbox
def render_selectbox(column_name,index_value):
  global RERUN

  pre = session_state.session_objects_dict[column_name]
  session_state.session_objects_dict[column_name] = st.selectbox(custom_column_names[column_name],list(range(len(columns_categories[column_name]))),index=index_value,format_func=lambda x: columns_categories[column_name][x])
  post = session_state.session_objects_dict[column_name]
  RERUN = RERUN or (pre != post)

def render_selectbox_state(column_name):
  render_selectbox(column_name,session_state.session_objects_dict[column_name])

### Number - number_input
def render_number(column_name,value,min_value,step):
  global RERUN

  pre = session_state.session_objects_dict[column_name]
  session_state.session_objects_dict[column_name] = st.number_input(custom_column_names[column_name], value=value, min_value=min_value, step=step)
  post = session_state.session_objects_dict[column_name]
  RERUN = RERUN or (pre != post) 

def render_number_state(column_name,min_value,step):
  render_number(column_name,session_state.session_objects_dict[column_name],min_value,step)

### text_input
def render_text_input(column_name,value):
  global RERUN

  pre = session_state.session_objects_dict[column_name]
  session_state.session_objects_dict[column_name] = st.text_input(custom_column_names[column_name], value=value)
  post = session_state.session_objects_dict[column_name]
  RERUN = RERUN or (pre != post) 

def render_text_input_state(column_name):
  render_text_input(column_name,session_state.session_objects_dict[column_name])

### Date - date_input
def render_date_input(column_name,value=None):
  global RERUN

  pre = session_state.session_objects_dict[column_name]
  session_state.session_objects_dict[column_name] = st.date_input(custom_column_names[column_name], value=value)
  post = session_state.session_objects_dict[column_name]
  RERUN = RERUN or (pre != post) 

def render_date_input_state(column_name):
  render_date_input(column_name,session_state.session_objects_dict[column_name])

def update_ADM_DATE():
  session_state.session_objects_dict['ADM_DATE'] = session_state.session_objects_dict['ACTUAL_ADMISSION_DTE']

def update_INST():
  session_state.session_objects_dict['INST'] = session_state.session_objects_dict['ACTUAL_CASE_INSTITUTION']

def update_CASE_NO():
  session_state.session_objects_dict['CASE_NO'] = session_state.session_objects_dict['ACTUAL_CASE_NUMBER']

def update_PATIENT_NUMBER():
  session_state.session_objects_dict['PATIENT_NUMBER'] = session_state.session_objects_dict['ACTUAL_PATIENT_NUMBER']

def update_ETBS_LOS():
  session_state.session_objects_dict['ETBS_LOS'] = session_state.session_objects_dict['ACTUAL_LOS']

def update_ETBS_ICU_HDU_LOS():
  session_state.session_objects_dict['ETBS_ICU_HDU_LOS'] = session_state.session_objects_dict['ACTUAL_ICU_HDU_LOS']

def update_CASE_TYPE():
  session_state.session_objects_dict['CASE_TYPE'] = ""

def update_PATIENT_TYPE():
  session_state.session_objects_dict['PATIENT_TYPE'] = ""

def update_RESID_POSTALCODE():
  session_state.session_objects_dict['RESID_POSTALCODE'] = ""

def update_CONT_POSTAL():
  session_state.session_objects_dict['CONT_POSTAL'] = ""

def update_TOTAL_HOSP():
  session_state.session_objects_dict['TOTAL_HOSP'] = ""



# TITLE
st.title('Hospital Bill Estimation upon Admission')

# DEFAULT Button
DEFAULT = st.button("Demo - Default values")

# START RENDERING
if DEFAULT:
  render_text_input('ACTUAL_CASE_NUMBER',1019003485)
  render_selectbox('ACTUAL_TREATMENT_CATEGORY',1)
  render_text_input('ACTUAL_PATIENT_NUMBER',6114934)
  render_text_input('ACTUAL_PRIMARY_DIAGNOSIS_SID',27276)
  render_selectbox('ACTUAL_CASE_INSTITUTION',1)
  render_date_input('ACTUAL_ADMISSION_DTE', date(2019, 1, 11))
  render_number('ACTUAL_LOS',value=3,min_value=1,step=1)
  render_number('ACTUAL_ICU_HDU_LOS',value=0,min_value=0,step=1)
  render_selectbox('ACTUAL_SURGICAL_CODE_1',0)
  render_selectbox('ACTUAL_SURGICAL_CODE_2',0)
  render_selectbox('ACTUAL_SURGICAL_CODE_3',0)
  render_selectbox('ACTUAL_DIAGNOSIS_CODE',3176)
  update_ADM_DATE()
  update_INST()
  update_CASE_NO()
  update_ETBS_LOS()
  update_ETBS_ICU_HDU_LOS()
  render_selectbox('ETBS_MOVE_TYPE',5)
  render_selectbox('ETBS_TOSP_1',0)
  render_selectbox('ETBS_TOSP_2',0)
  render_selectbox('ETBS_TOSP_3',0)
  render_selectbox('ETBS_ICD10_1',3177)
  render_selectbox('ETBS_ICD10_2',0)
  render_selectbox('ETBS_ICD10_3',0)
  update_CASE_TYPE()
  update_PATIENT_TYPE()
  render_selectbox('ADMISSION_TYPE',5)
  render_selectbox('TREATMENT_CATEGORY',46)
  render_selectbox('REFERRAL_TYPE',23)
  render_selectbox('DEPT_OU',57)
  render_selectbox('ADMITTING_SMC_NUMBER',0)
  render_selectbox('ATTENDING_SMC_NUMBER',0)
  render_selectbox('REFERRAL_SMC_NUMBER',0)
  render_selectbox('ADM_CLASS_DISC',0)
  update_PATIENT_NUMBER()
  render_selectbox('GENDER',2)
  render_date_input('DOB', date(1961, 4, 29))
  render_selectbox('MARITAL_STATUS',2)
  render_selectbox('RELIGION',0)
  render_selectbox('NATIONALITY',128)
  render_selectbox('RESID_CTY',199)
  update_RESID_POSTALCODE()
  render_selectbox('OCCUPATION',0)
  render_selectbox('RESID_GEOAREA',187)
  render_selectbox('NONRESID_FLAG',1)
  render_selectbox('IDENT_TYPE',4)
  update_CONT_POSTAL()
  render_selectbox('CONT_RELATION',2)
  update_TOTAL_HOSP()
  
else:
  render_text_input_state('ACTUAL_CASE_NUMBER')
  render_selectbox_state('ACTUAL_TREATMENT_CATEGORY')
  render_text_input_state('ACTUAL_PATIENT_NUMBER')
  render_text_input_state('ACTUAL_PRIMARY_DIAGNOSIS_SID')
  render_selectbox_state('ACTUAL_CASE_INSTITUTION')
  render_date_input_state('ACTUAL_ADMISSION_DTE')
  render_number_state('ACTUAL_LOS',min_value=1,step=1)
  render_number_state('ACTUAL_ICU_HDU_LOS',min_value=0,step=1)
  render_selectbox_state('ACTUAL_SURGICAL_CODE_1')
  render_selectbox_state('ACTUAL_SURGICAL_CODE_2')
  render_selectbox_state('ACTUAL_SURGICAL_CODE_3')
  render_selectbox_state('ACTUAL_DIAGNOSIS_CODE')
  update_ADM_DATE()
  update_INST()
  update_CASE_NO()
  update_ETBS_LOS()
  update_ETBS_ICU_HDU_LOS() 
  render_selectbox_state('ETBS_MOVE_TYPE')
  render_selectbox_state('ETBS_TOSP_1')
  render_selectbox_state('ETBS_TOSP_2')
  render_selectbox_state('ETBS_TOSP_3')
  render_selectbox_state('ETBS_ICD10_1')
  render_selectbox_state('ETBS_ICD10_2')
  render_selectbox_state('ETBS_ICD10_3')
  update_CASE_TYPE()
  update_PATIENT_TYPE()
  render_selectbox_state('ADMISSION_TYPE')
  render_selectbox_state('TREATMENT_CATEGORY')
  render_selectbox_state('REFERRAL_TYPE')
  render_selectbox_state('DEPT_OU')
  render_selectbox_state('ADMITTING_SMC_NUMBER')
  render_selectbox_state('ATTENDING_SMC_NUMBER')
  render_selectbox_state('REFERRAL_SMC_NUMBER')
  render_selectbox_state('ADM_CLASS_DISC')
  update_PATIENT_NUMBER()
  render_selectbox_state('GENDER')
  render_date_input_state('DOB')
  render_selectbox_state('MARITAL_STATUS')
  render_selectbox_state('RELIGION')
  render_selectbox_state('NATIONALITY')
  render_selectbox_state('RESID_CTY')
  update_RESID_POSTALCODE()
  render_selectbox_state('OCCUPATION')
  render_selectbox_state('RESID_GEOAREA')
  render_selectbox_state('NONRESID_FLAG')
  render_selectbox_state('IDENT_TYPE')
  update_CONT_POSTAL()
  render_selectbox_state('CONT_RELATION')
  update_TOTAL_HOSP()

if RERUN:
  rerun()



########################################

SUBMIT = st.button("SUBMIT")
if not SUBMIT:
  st.stop()


# Preparing the input dataframe
test_dict = session_state.session_objects_dict.copy()
del test_dict['dpl']
del test_dict['model']

# Processing index -> label
for k in columns_categories.keys():
  test_dict[k] = columns_categories[k][session_state.session_objects_dict[k]]

# Processing Nan
for k in test_dict.keys():
  # if (test_dict[k] == "" and col_dtypes[k] == np.float64):
  if (test_dict[k] == ""):
    test_dict[k] = np.nan

def update_test_dict():
  test_dict['ADM_DATE'] = test_dict['ACTUAL_ADMISSION_DTE']
  test_dict['INST'] = test_dict['ACTUAL_CASE_INSTITUTION']
  test_dict['CASE_NO'] = test_dict['ACTUAL_CASE_NUMBER']
  test_dict['PATIENT_NUMBER'] = test_dict['ACTUAL_PATIENT_NUMBER']
  test_dict['ETBS_LOS'] = test_dict['ACTUAL_LOS']
  test_dict['ETBS_ICU_HDU_LOS'] = test_dict['ACTUAL_ICU_HDU_LOS']
  # test_dict['CASE_TYPE'] = test_dict['']
  # test_dict['PATIENT_TYPE'] = test_dict['']
  # test_dict['RESID_POSTALCODE'] = test_dict['']
  # test_dict['CONT_POSTAL'] = test_dict['']

update_test_dict()

df_test = pd.DataFrame(test_dict, index=[0])


# idx = [8089]
# df = pd.read_csv('./models/BillEstimate.csv')
# df_test = df.iloc[idx, :]

st.write(df_test.T)

df_test_proc = session_state.session_objects_dict['dpl'].transform_raw_test_data(df_test, split_hosp=False)
df_test_new = session_state.session_objects_dict['dpl'].transform_test_data(df_test_proc.drop('TOTAL_HOSP', axis=1),
                                      scaler_pkl_file_path,
                                      ohe_pkl_file_path,
                                      feature_importance_file_path)

y_pred = session_state.session_objects_dict['model'].predict(df_test_new)
st.write('Prediction: SGD $', round(y_pred[0,0],2)) #result







