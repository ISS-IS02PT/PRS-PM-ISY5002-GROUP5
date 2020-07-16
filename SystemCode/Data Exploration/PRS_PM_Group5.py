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
import matplotlib.pyplot as plt
# %matplotlib inline
pd.set_option('display.max_rows', 500)

# #### Read data from file

filePath = 'ParkwaySampleDataForProject_05.xlsx'
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
# -

# ### Feature Engineering



# ### Preprocessing



# ### Training



# ### Validation


