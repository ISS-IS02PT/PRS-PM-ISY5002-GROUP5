import streamlit as st
import SessionState
import pickle
import numpy as np
import pandas as pd
import time
from datetime import datetime


def rerun():
  # print(str(datetime.now()) + " Re-running")
  raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))



# INSTITUTION
# CASE_NUMBER
# ??? TOTAL_PAID_AMT
# PAYER_CODE_1
# PAYER_NAME_1
# PAYER_1_PAID_AMT
# PAYER_CODE_2
# PAYER_NAME_2
# PAYER_2_PAID_AMT
# PAYER_CODE_3
# PAYER_NAME_3
# PAYER_3_PAID_AMT
# PAYER_CODE_4
# PAYER_NAME_4
# PAYER_4_PAID_AMT
# PAYER_CODE_5
# PAYER_NAME_5
# PAYER_5_PAID_AMT
# CASE_TYPE
# BED_TYPE
# REFERRAL_TYPE
# TREATMENT_CATEGORY
# ADMISSION_DTE
# ADMISSION_TYPE
# DISCHARGE_DTE
# DISCHARGE_TYPE
# xxx DISCHARGE_TYPE_DESC
# LOS_DAYS
# DOCTOR_CODE
# DOCTOR_NAME
# SPECIALTY_CODE
# SPECIALTY_DESC
# SPECIALTY_GRP
# TOSP_COUNT
# TOSP_STRING
# TOSP_CODE1
# TOSP_CODE2
# TOSP_CODE3
# TOSP_CODE4
# TOSP_DESC1
# TOSP_DESC2
# TOSP_DESC3
# TOSP_DESC4
# NATIONALITY
# RESID_CTY
# RESID_POSTALCODE
# DOB
# NONRESID_FLAG
# PATIENT_SID
# PATIENT_NUMBER
# GENDER
# DECEASED_FLAG
# MARITAL_STATUS
# RELIGION
# LANGUAGE
# VIP_FLAG
# RACE
# DRG_CODE
# DRG_DESC
# PAYER_CODE1_V
# PAYER_NAME1_V
# PAYER_CODE2_V
# PAYER_NAME2_V
# PAYER_CODE3_V
# PAYER_NAME3_V
# PAYER_CODE4_V
# PAYER_NAME4_V
# PACKAGE_CODE
# PACKAGE_PRICE
# PACKAGE_EXCL
# PACKAGE_ADJ
# PACKAGE_DESC
# PACKAGE_CODE1
# PACKAGE_CODE2
# PACKAGE_DESC1
# PACKAGE_DESC2
# ICD_CODE1
# ICD_CODE2
# ICD_CODE3
# ICDCODE_STRING
# PROF_FEE
# TOTAL_FEES
# WRITE_OFF



# CONSTANTS / DEFAULTS
INSTITUTION_NAME = (
  'GHL',
  'MEH',
  'PEH',
  'PNH'
)
INSTITUTION_INDEX = list(range(len(INSTITUTION_NAME)))

CASE_NUMBER = 1234567

PAYER_CODE = (
  'NA',
  '',
  '0002000000',
  '0002000166'
)
PAYER_CODE_NAME = (
  'NA',
  'Self-Paid',
  '0002000000: MEDISAVE',
  '0002000166: BUPA INSURANCE SERVICES LTD'
)
PAYER_CODE_INDEX = list(range(len(PAYER_CODE_NAME)))

CASE_TYPE = (
  'INPATIENT',
  'OUTPATIENT',
)
CASE_TYPE_INDEX = list(range(len(CASE_TYPE)))

BED_TYPE = (
  'SGL',
  'DC',
  '2BD'
)
BED_TYPE_INDEX = list(range(len(BED_TYPE)))

REFERRAL_TYPE = (
  'MC',
  'ZS'
)
REFERRAL_TYPE_INDEX = list(range(len(REFERRAL_TYPE)))

TREATMENT_CATEGORY = (
  'SGL',
  'DW2B',
  'ENDO',
  'LASIK'
)
TREATMENT_CATEGORY_INDEX = list(range(len(TREATMENT_CATEGORY)))

ADMISSION_TYPE = (
  'DS',
  'MA',
  'IN'
)
ADMISSION_TYPE_INDEX = list(range(len(ADMISSION_TYPE)))

DISCHARGE_TYPE_DESC = (
  '01: DISCHARGED',
  '06: DEATH'
)
DISCHARGE_TYPE_INDEX = list(range(len(DISCHARGE_TYPE_DESC)))

DOCTOR_CODE = (
  '08492A',
  '07990A'
)
DOCTOR_CODE_NAME_SPECIALTY = (
  '08492A: DR THIA TECK JOO KELVIN (GASTROENTEROLOGY)',
  '07990A: DR TAN JYE YNG JANE (GENERAL SURGERY)'
)
DOCTOR_CODE_INDEX = list(range(len(DOCTOR_CODE_NAME_SPECIALTY)))

TOSP_CODE = (
  '',
  'SA817B',
  'SF702C'
)
TOSP_DESC = (
  '',
  'SA817B: Breast, Post Mastectomy, latissmus dorsi',
  "SF702C: Colon, Colonoscopy (diagnostic), fibreop"
)
TOSP_CODE_INDEX = list(range(len(TOSP_DESC)))

NATIONALITY = (
  'Indian',
  'Malaysian',
  'Singaporean'
)
NATIONALITY_INDEX = list(range(len(NATIONALITY)))

RESID_CTY = (
  'SINGAPORE',
  'Malaysia',
  'Indonesia'
)
RESID_CTY_INDEX = list(range(len(RESID_CTY)))

RESID_POSTALCODE = ''

NONRESID_FLAG = (
  'Y',
  'N'
)
NONRESID_FLAG_INDEX = list(range(len(NONRESID_FLAG)))

GENDER = (
  'MALE',
  'FEMALE'
)
GENDER_INDEX = list(range(len(GENDER)))

MARITAL_STATUS = (
  '',
  'MARRI',
  'SINGLE',
  'WIDOW'
)
MARITAL_STATUS_INDEX = list(range(len(MARITAL_STATUS)))

RELIGION = (
  '',
  'ROMAN CATHOLIC',
  'CHRISTIAN',
  'OTHER'
)
RELIGION_INDEX = list(range(len(RELIGION)))

LANGUAGE = (
  'E'
)
LANGUAGE_INDEX = list(range(len(LANGUAGE)))

VIP_FLAG = (
  'N',
  'Y'
)
VIP_FLAG_INDEX = list(range(len(VIP_FLAG)))

RACE = (
  'CHINESE',
  'CAUCASIAN',
  'OTHERS'
)
RACE_INDEX = list(range(len(RACE)))




# INPUT VARIABLES INIT
session_state = SessionState.get(
  st_INSTITUTION=0,
  st_CASE_NUMBER=CASE_NUMBER,
  st_PAYER_1=0,
  st_PAYER_1_PAID_AMT=0,
  st_PAYER_2=0,
  st_PAYER_2_PAID_AMT=0,
  st_PAYER_3=0,
  st_PAYER_3_PAID_AMT=0,
  st_PAYER_4=0,
  st_PAYER_4_PAID_AMT=0,
  st_PAYER_5=0,
  st_PAYER_5_PAID_AMT=0,
  st_CASE_TYPE=0,
  st_BED_TYPE=0,
  st_REFERRAL_TYPE=0,
  st_TREATMENT_CATEGORY=0,
  #st_ADMISSION_DTE,
  st_ADMISSION_TYPE=0,
  #st_DISCHARGE_DTE,
  st_DISCHARGE_TYPE=0,
  st_DOCTOR=0,
  st_TOSP_1=0,
  st_TOSP_2=0,
  st_TOSP_3=0,
  st_TOSP_4=0,
  st_NATIONALITY=0,
  st_RESID_CTY=0,
  st_RESID_POSTALCODE=RESID_POSTALCODE,
  st_NONRESID_FLAG=0,
  st_GENDER=0,
  st_MARITAL_STATUS=0,
  st_RELIGION=0,
  st_LANGUAGE=0,
  st_VIP_FLAG=0,
  st_RACE=0
)

# print(str(datetime.now()) + "-A")
RERUN = False


# RENDER FUNCTIONS
def render_INSTITUTION(index_value):
  global RERUN

  pre = session_state.st_INSTITUTION
  session_state.st_INSTITUTION = st.selectbox("INSTITUTION",INSTITUTION_INDEX,index=index_value,format_func=lambda x: INSTITUTION_NAME[x])
  post = session_state.st_INSTITUTION
  RERUN = RERUN or (pre != post)
  # print(str(datetime.now()) + "-B: " + str(RERUN))

def render_CASE_NUMBER(value):
  global RERUN

  pre = session_state.st_CASE_NUMBER
  session_state.st_CASE_NUMBER = st.text_input("CASE_NUMBER",value=value)
  post = session_state.st_CASE_NUMBER
  RERUN = RERUN or (pre != post)

def render_PAYER_1(index_value):
  global RERUN

  pre = session_state.st_PAYER_1
  session_state.st_PAYER_1 = st.selectbox("PAYER_1",PAYER_CODE_INDEX,index=index_value,format_func=lambda x: PAYER_CODE_NAME[x])
  post = session_state.st_PAYER_1
  RERUN = RERUN or (pre != post)

def render_PAYER_1_PAID_AMT(value):
  global RERUN

  if session_state.st_PAYER_1 == 0:
    value = 0

  pre = session_state.st_PAYER_1_PAID_AMT
  session_state.st_PAYER_1_PAID_AMT = st.number_input("PAYER_1_PAID_AMT",value=value, step=50)
  post = session_state.st_PAYER_1_PAID_AMT
  RERUN = RERUN or (pre != post)

def render_PAYER_2(index_value):
  global RERUN

  pre = session_state.st_PAYER_2
  session_state.st_PAYER_2 = st.selectbox("PAYER_2",PAYER_CODE_INDEX,index=index_value,format_func=lambda x: PAYER_CODE_NAME[x])
  post = session_state.st_PAYER_2
  RERUN = RERUN or (pre != post)

def render_PAYER_2_PAID_AMT(value):
  global RERUN

  if session_state.st_PAYER_2 == 0:
    value = 0

  pre = session_state.st_PAYER_2_PAID_AMT
  session_state.st_PAYER_2_PAID_AMT = st.number_input("PAYER_2_PAID_AMT",value=value, step=50)
  post = session_state.st_PAYER_2_PAID_AMT
  RERUN = RERUN or (pre != post)

def render_PAYER_3(index_value):
  global RERUN

  pre = session_state.st_PAYER_3
  session_state.st_PAYER_3 = st.selectbox("PAYER_3",PAYER_CODE_INDEX,index=index_value,format_func=lambda x: PAYER_CODE_NAME[x])
  post = session_state.st_PAYER_3
  RERUN = RERUN or (pre != post)

def render_PAYER_3_PAID_AMT(value):
  global RERUN

  if session_state.st_PAYER_3 == 0:
    value = 0

  pre = session_state.st_PAYER_3_PAID_AMT
  session_state.st_PAYER_3_PAID_AMT = st.number_input("PAYER_3_PAID_AMT",value=value, step=50)
  post = session_state.st_PAYER_3_PAID_AMT
  RERUN = RERUN or (pre != post)

def render_PAYER_4(index_value):
  global RERUN

  pre = session_state.st_PAYER_4
  session_state.st_PAYER_4 = st.selectbox("PAYER_4",PAYER_CODE_INDEX,index=index_value,format_func=lambda x: PAYER_CODE_NAME[x])
  post = session_state.st_PAYER_4
  RERUN = RERUN or (pre != post)

def render_PAYER_4_PAID_AMT(value):
  global RERUN

  if session_state.st_PAYER_4 == 0:
    value = 0

  pre = session_state.st_PAYER_4_PAID_AMT
  session_state.st_PAYER_4_PAID_AMT = st.number_input("PAYER_4_PAID_AMT",value=value, step=50)
  post = session_state.st_PAYER_4_PAID_AMT
  RERUN = RERUN or (pre != post)

def render_PAYER_5(index_value):
  global RERUN

  pre = session_state.st_PAYER_5
  session_state.st_PAYER_5 = st.selectbox("PAYER_5",PAYER_CODE_INDEX,index=index_value,format_func=lambda x: PAYER_CODE_NAME[x])
  post = session_state.st_PAYER_5
  RERUN = RERUN or (pre != post)

def render_PAYER_5_PAID_AMT(value):
  global RERUN

  if session_state.st_PAYER_5 == 0:
    value = 0

  pre = session_state.st_PAYER_5_PAID_AMT
  session_state.st_PAYER_5_PAID_AMT = st.number_input("PAYER_5_PAID_AMT",value=value, step=50)
  post = session_state.st_PAYER_5_PAID_AMT
  RERUN = RERUN or (pre != post)

def render_CASE_TYPE(index_value):
  global RERUN

  pre = session_state.st_CASE_TYPE
  session_state.st_CASE_TYPE = st.selectbox("CASE_TYPE",CASE_TYPE_INDEX,index=index_value,format_func=lambda x: CASE_TYPE[x])
  post = session_state.st_CASE_TYPE
  RERUN = RERUN or (pre != post)

def render_BED_TYPE(index_value):
  global RERUN

  pre = session_state.st_BED_TYPE
  session_state.st_BED_TYPE = st.selectbox("BED_TYPE",BED_TYPE_INDEX,index=index_value,format_func=lambda x: BED_TYPE[x])
  post = session_state.st_BED_TYPE
  RERUN = RERUN or (pre != post)

def render_REFERRAL_TYPE(index_value):
  global RERUN

  pre = session_state.st_REFERRAL_TYPE
  session_state.st_REFERRAL_TYPE = st.selectbox("REFERRAL_TYPE",REFERRAL_TYPE_INDEX,index=index_value,format_func=lambda x: REFERRAL_TYPE[x])
  post = session_state.st_REFERRAL_TYPE
  RERUN = RERUN or (pre != post)

def render_TREATMENT_CATEGORY(index_value):
  global RERUN

  pre = session_state.st_TREATMENT_CATEGORY
  session_state.st_TREATMENT_CATEGORY = st.selectbox("TREATMENT_CATEGORY",TREATMENT_CATEGORY_INDEX,index=index_value,format_func=lambda x: TREATMENT_CATEGORY[x])
  post = session_state.st_TREATMENT_CATEGORY
  RERUN = RERUN or (pre != post)

def render_ADMISSION_TYPE(index_value):
  global RERUN

  pre = session_state.st_ADMISSION_TYPE
  session_state.st_ADMISSION_TYPE = st.selectbox("ADMISSION_TYPE",ADMISSION_TYPE_INDEX,index=index_value,format_func=lambda x: ADMISSION_TYPE[x])
  post = session_state.st_ADMISSION_TYPE
  RERUN = RERUN or (pre != post)

def render_DISCHARGE_TYPE(index_value):
  global RERUN

  pre = session_state.st_DISCHARGE_TYPE
  session_state.st_DISCHARGE_TYPE = st.selectbox("DISCHARGE_TYPE",DISCHARGE_TYPE_INDEX,index=index_value,format_func=lambda x: DISCHARGE_TYPE_DESC[x])
  post = session_state.st_DISCHARGE_TYPE
  RERUN = RERUN or (pre != post)

def render_DOCTOR(index_value):
  global RERUN

  pre = session_state.st_DOCTOR
  session_state.st_DOCTOR = st.selectbox("DOCTOR",DOCTOR_CODE_INDEX,index=index_value,format_func=lambda x: DOCTOR_CODE_NAME_SPECIALTY[x])
  post = session_state.st_DOCTOR
  RERUN = RERUN or (pre != post)

def render_TOSP_1(index_value):
  global RERUN

  pre = session_state.st_TOSP_1
  session_state.st_TOSP_1 = st.selectbox("TOSP_1",TOSP_CODE_INDEX,index=index_value,format_func=lambda x: TOSP_DESC[x])
  post = session_state.st_TOSP_1
  RERUN = RERUN or (pre != post)

def render_TOSP_2(index_value):
  global RERUN

  pre = session_state.st_TOSP_2
  session_state.st_TOSP_2 = st.selectbox("TOSP_2",TOSP_CODE_INDEX,index=index_value,format_func=lambda x: TOSP_DESC[x])
  post = session_state.st_TOSP_2
  RERUN = RERUN or (pre != post)

def render_TOSP_3(index_value):
  global RERUN

  pre = session_state.st_TOSP_3
  session_state.st_TOSP_3 = st.selectbox("TOSP_3",TOSP_CODE_INDEX,index=index_value,format_func=lambda x: TOSP_DESC[x])
  post = session_state.st_TOSP_3
  RERUN = RERUN or (pre != post)

def render_TOSP_4(index_value):
  global RERUN

  pre = session_state.st_TOSP_4
  session_state.st_TOSP_4 = st.selectbox("TOSP_4",TOSP_CODE_INDEX,index=index_value,format_func=lambda x: TOSP_DESC[x])
  post = session_state.st_TOSP_4
  RERUN = RERUN or (pre != post)

def render_NATIONALITY(index_value):
  global RERUN

  pre = session_state.st_NATIONALITY
  session_state.st_NATIONALITY = st.selectbox("NATIONALITY",NATIONALITY_INDEX,index=index_value,format_func=lambda x: NATIONALITY[x])
  post = session_state.st_NATIONALITY
  RERUN = RERUN or (pre != post)

def render_RESID_CTY(index_value):
  global RERUN

  pre = session_state.st_RESID_CTY
  session_state.st_RESID_CTY = st.selectbox("RESID_CTY",RESID_CTY_INDEX,index=index_value,format_func=lambda x: RESID_CTY[x])
  post = session_state.st_RESID_CTY
  RERUN = RERUN or (pre != post)

def render_RESID_POSTALCODE(value):
  global RERUN

  pre = session_state.st_RESID_POSTALCODE
  session_state.st_RESID_POSTALCODE = st.text_input("RESID_POSTALCODE",value=value)
  post = session_state.st_RESID_POSTALCODE
  RERUN = RERUN or (pre != post)

def render_NONRESID_FLAG(index_value):
  global RERUN

  pre = session_state.st_NONRESID_FLAG
  session_state.st_NONRESID_FLAG = st.selectbox("NONRESID_FLAG",NONRESID_FLAG_INDEX,index=index_value,format_func=lambda x: NONRESID_FLAG[x])
  post = session_state.st_NONRESID_FLAG
  RERUN = RERUN or (pre != post)

def render_GENDER(index_value):
  global RERUN

  pre = session_state.st_GENDER
  session_state.st_GENDER = st.selectbox("GENDER",GENDER_INDEX,index=index_value,format_func=lambda x: GENDER[x])
  post = session_state.st_GENDER
  RERUN = RERUN or (pre != post)

def render_MARITAL_STATUS(index_value):
  global RERUN

  pre = session_state.st_MARITAL_STATUS
  session_state.st_MARITAL_STATUS = st.selectbox("MARITAL_STATUS",MARITAL_STATUS_INDEX,index=index_value,format_func=lambda x: MARITAL_STATUS[x])
  post = session_state.st_MARITAL_STATUS
  RERUN = RERUN or (pre != post)

def render_RELIGION(index_value):
  global RERUN

  pre = session_state.st_RELIGION
  session_state.st_RELIGION = st.selectbox("RELIGION",RELIGION_INDEX,index=index_value,format_func=lambda x: RELIGION[x])
  post = session_state.st_RELIGION
  RERUN = RERUN or (pre != post)

def render_GENDER(index_value):
  global RERUN

  pre = session_state.st_GENDER
  session_state.st_GENDER = st.selectbox("GENDER",GENDER_INDEX,index=index_value,format_func=lambda x: GENDER[x])
  post = session_state.st_GENDER
  RERUN = RERUN or (pre != post)

def render_LANGUAGE(index_value):
  global RERUN

  pre = session_state.st_LANGUAGE
  session_state.st_LANGUAGE = st.selectbox("LANGUAGE",LANGUAGE_INDEX,index=index_value,format_func=lambda x:LANGUAGE[x])
  post = session_state.st_LANGUAGE
  RERUN = RERUN or (pre != post)  

def render_VIP_FLAG(index_value):
  global RERUN

  pre = session_state.st_VIP_FLAG
  session_state.st_VIP_FLAG = st.selectbox("VIP_FLAG",VIP_FLAG_INDEX,index=index_value,format_func=lambda x:VIP_FLAG[x])
  post = session_state.st_VIP_FLAG
  RERUN = RERUN or (pre != post)  

def render_RACE(index_value):
  global RERUN

  pre = session_state.st_RACE
  session_state.st_RACE = st.selectbox("RACE",RACE_INDEX,index=index_value,format_func=lambda x:RACE[x])
  post = session_state.st_RACE
  RERUN = RERUN or (pre != post)  




# TITLE
st.title('Use Case #1: Prediction of Write-Off Cases')

# DEFAULT Button
DEFAULT = st.button("DEFAULT")

# START RENDERING
if DEFAULT:
  render_INSTITUTION(0)
  render_CASE_NUMBER(CASE_NUMBER)
  render_PAYER_1(1)                                         # Self-Paid
  render_PAYER_1_PAID_AMT(8833)
  render_PAYER_2(2)                                         # NA
  render_PAYER_2_PAID_AMT(15000)
  render_PAYER_3(0)                                         # NA
  render_PAYER_3_PAID_AMT(0)
  render_PAYER_4(0)                                         # NA
  render_PAYER_4_PAID_AMT(0)
  render_PAYER_5(0)                                         # NA
  render_PAYER_5_PAID_AMT(0)
  render_CASE_TYPE(0)                                       # INPATIENT
  render_BED_TYPE(0)                                        # SGL
  render_REFERRAL_TYPE(0)                                   # MC
  render_TREATMENT_CATEGORY(0)                              # SGL
  # ADMISSION_DTE
  render_ADMISSION_TYPE(0)                                  # PI
  # DISCHARGE_DTE
  render_DISCHARGE_TYPE(0)                                  # 01
  render_DOCTOR(0)                                          # 08492A: DR THIA TECK JOO KELVIN
  render_TOSP_1(1)                                          # SA817B: Breast, Post Mastectomy, latissmus dorsi
  render_TOSP_2(0)                                          # ''
  render_TOSP_3(0)                                          # ''
  render_TOSP_4(0)                                          # ''
  render_NATIONALITY(0)                                     # 'INDIAN'
  render_RESID_CTY(0)                                       # 'SINGAPORE'
  render_RESID_POSTALCODE(RESID_POSTALCODE)
  # DOB
  render_NONRESID_FLAG(1)                                   # N
  render_GENDER(0)                                          # MALE
  render_MARITAL_STATUS(0)                                  # ''
  render_RELIGION(0)                                        # ''
  render_LANGUAGE(0)                                        # E
  render_VIP_FLAG(0)                                        # N
  render_RACE(0)                                            # CHINESE
else:
  render_INSTITUTION(session_state.st_INSTITUTION)
  render_CASE_NUMBER(session_state.st_CASE_NUMBER)
  render_PAYER_1(session_state.st_PAYER_1)
  render_PAYER_1_PAID_AMT(session_state.st_PAYER_1_PAID_AMT)
  render_PAYER_2(session_state.st_PAYER_2)
  render_PAYER_2_PAID_AMT(session_state.st_PAYER_2_PAID_AMT)
  render_PAYER_3(session_state.st_PAYER_3)
  render_PAYER_3_PAID_AMT(session_state.st_PAYER_3_PAID_AMT)
  render_PAYER_4(session_state.st_PAYER_4)
  render_PAYER_4_PAID_AMT(session_state.st_PAYER_4_PAID_AMT)
  render_PAYER_5(session_state.st_PAYER_5)
  render_PAYER_5_PAID_AMT(session_state.st_PAYER_5_PAID_AMT)
  render_CASE_TYPE(session_state.st_CASE_TYPE)
  render_BED_TYPE(session_state.st_BED_TYPE)
  render_REFERRAL_TYPE(session_state.st_REFERRAL_TYPE)
  render_TREATMENT_CATEGORY(session_state.st_TREATMENT_CATEGORY)
  # ADMISSION_DTE
  render_ADMISSION_TYPE(session_state.st_ADMISSION_TYPE)
  # DISCHARGE_DTE
  render_DISCHARGE_TYPE(session_state.st_DISCHARGE_TYPE)
  render_DOCTOR(session_state.st_DOCTOR)
  render_TOSP_1(session_state.st_TOSP_1)
  render_TOSP_2(session_state.st_TOSP_2)
  render_TOSP_3(session_state.st_TOSP_3)
  render_TOSP_4(session_state.st_TOSP_4)
  render_NATIONALITY(session_state.st_NATIONALITY)
  render_RESID_CTY(session_state.st_RESID_CTY)
  render_RESID_POSTALCODE(session_state.st_RESID_POSTALCODE)
  # DOB
  render_NONRESID_FLAG(session_state.st_NONRESID_FLAG)
  render_GENDER(session_state.st_GENDER)
  render_MARITAL_STATUS(session_state.st_MARITAL_STATUS)
  render_RELIGION(session_state.st_RELIGION)
  render_LANGUAGE(session_state.st_LANGUAGE)
  render_VIP_FLAG(session_state.st_VIP_FLAG)
  render_RACE(session_state.st_RACE)


# DRG_CODE
# DRG_DESC
# PAYER_CODE1_V
# PAYER_NAME1_V
# PAYER_CODE2_V
# PAYER_NAME2_V
# PAYER_CODE3_V
# PAYER_NAME3_V
# PAYER_CODE4_V
# PAYER_NAME4_V
# PACKAGE_CODE
# PACKAGE_PRICE
# PACKAGE_EXCL
# PACKAGE_ADJ
# PACKAGE_DESC
# PACKAGE_CODE1
# PACKAGE_CODE2
# PACKAGE_DESC1
# PACKAGE_DESC2
# ICD_CODE1
# ICD_CODE2
# ICD_CODE3
# ICDCODE_STRING
# PROF_FEE
# TOTAL_FEES
# WRITE_OFF



# print(str(datetime.now()) + "-C: " + str(RERUN))
if RERUN:
  rerun()

SUBMIT = st.button("SUBMIT")
if not SUBMIT:
  st.stop()

st.write("INSTITUTION: " + str(session_state.st_INSTITUTION))
st.write("CASE_NUMBER: " + str(session_state.st_CASE_NUMBER))
st.write("PAYER_CODE_1: " + str(PAYER_CODE[session_state.st_PAYER_1]))
st.write("PAYER_1_PAID_AMT: " + str(session_state.st_PAYER_1_PAID_AMT))
st.write("PAYER_CODE_2: " + str(PAYER_CODE[session_state.st_PAYER_2]))
st.write("PAYER_2_PAID_AMT: " + str(session_state.st_PAYER_2_PAID_AMT))
st.write("PAYER_CODE_3: " + str(PAYER_CODE[session_state.st_PAYER_3]))
st.write("PAYER_3_PAID_AMT: " + str(session_state.st_PAYER_3_PAID_AMT))
st.write("PAYER_CODE_4: " + str(PAYER_CODE[session_state.st_PAYER_4]))
st.write("PAYER_4_PAID_AMT: " + str(session_state.st_PAYER_4_PAID_AMT))
st.write("PAYER_CODE_5: " + str(PAYER_CODE[session_state.st_PAYER_5]))
st.write("PAYER_5_PAID_AMT: " + str(session_state.st_PAYER_5_PAID_AMT))
st.write("CASE_TYPE: " + str(session_state.st_CASE_TYPE))
st.write("BED_TYPE: " + str(session_state.st_BED_TYPE))
st.write("REFERRAL_TYPE: " + str(session_state.st_REFERRAL_TYPE))
st.write("TREATMENT_CATEGORY: " + str(session_state.st_TREATMENT_CATEGORY))
# ADMISSION_DTE
st.write("ADMISSION_TYPE: " + str(session_state.st_ADMISSION_TYPE))
# DISCHARGE_DTE
st.write("DISCHARGE_TYPE: " + str(session_state.st_DISCHARGE_TYPE))
# LOS_DAYS
st.write("DOCTOR_CODE: " + str(DOCTOR_CODE[session_state.st_DOCTOR]))
st.write("TOSP_CODE1: " + str(TOSP_CODE[session_state.st_TOSP_1]))
st.write("TOSP_CODE2: " + str(TOSP_CODE[session_state.st_TOSP_2]))
st.write("TOSP_CODE3: " + str(TOSP_CODE[session_state.st_TOSP_3]))
st.write("TOSP_CODE4: " + str(TOSP_CODE[session_state.st_TOSP_4]))
st.write("NATIONALITY: " + str(session_state.st_NATIONALITY))
st.write("RESID_CTY: " + str(session_state.st_RESID_CTY))
st.write("RESID_POSTALCODE: " + str(session_state.st_RESID_POSTALCODE))
st.write("NONRESID_FLAG: " + str(session_state.st_NONRESID_FLAG))
st.write("GENDER: " + str(session_state.st_GENDER))
st.write("MARITAL_STATUS: " + str(session_state.st_MARITAL_STATUS))
st.write("RELIGION: " + str(session_state.st_RELIGION))
st.write("LANGUAGE: " + str(session_state.st_LANGUAGE))
st.write("VIP_FLAG: " + str(session_state.st_VIP_FLAG))
st.write("RACE: " + str(session_state.st_RACE))
# DOB

# ##### Progress Bar - 100%
# st.header("=== Progress Bar ===")

# import time

# # Progress time = 5 seconds
# progress_time = 5

# # Add a text_placeholder
# text_placeholder = st.empty()
# bar = st.progress(0)

# for i in range(progress_time):
#   # Update the progress bar with each iteration.
#   time.sleep(1)
#   current_percentage = (i+1) * 1 / progress_time
#   # temp_text = 
#   text_placeholder.text(f'Percentage = {current_percentage*100:0.0f}%')
#   bar.progress(current_percentage)
# '...and now we\'re done!'





