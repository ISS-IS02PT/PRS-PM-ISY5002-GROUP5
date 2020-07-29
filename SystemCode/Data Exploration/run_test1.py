import pandas as pd
import numpy as np

filePath = '.\data\ParkwaySampleDataForProject_05.xlsx'
df = pd.read_excel(filePath)

# drop columns
col_drop = ['PAYER_NAME_1', 'PAYER_NAME_2', 'PAYER_NAME_3', 'PAYER_NAME_4', 'PAYER_NAME_5', 'DISCHARGE_TYPE_DESC', 'DOCTOR_NAME', 'SPECIALTY_DESC',
            'TOSP_STRING', 'TOSP_DESC1', 'TOSP_DESC2', 'TOSP_DESC3', 'TOSP_DESC4', 'DRG_DESC', 'PAYER_CODE1_V', 'PAYER_NAME1_V', 'PAYER_CODE2_V',
            'PAYER_NAME2_V', 'PAYER_CODE3_V', 'PAYER_NAME3_V', 'PAYER_CODE4_V', 'PAYER_NAME4_V', 'PACKAGE_DESC', 'PACKAGE_DESC1', 'PACKAGE_DESC2',
            'ICDCODE_STRING']
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

# Get random sample
sample_frac = 0.1
np.random.seed(0)
sample_size = int(round(abs(sample_frac * df1.shape[0])))
print(sample_size)
sample_indices = np.random.choice(df1.shape[0], size=sample_size, replace=False)
print(len(sample_indices))
df1_sample = df1.iloc[sample_indices]
print(df1_sample.shape)