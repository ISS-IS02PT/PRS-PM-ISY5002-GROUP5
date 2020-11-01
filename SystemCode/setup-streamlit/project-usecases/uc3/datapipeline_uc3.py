import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from statistics import variance, mean, stdev
import constants_uc3
import os
#from dask import dataframe as dd
#import dask
import pickle


class Datapipeline():
    '''
    Data pipeline class
    '''

    def __init__(self):
        '''
        Initialise
        '''
        self.col_dtypes = {
            'ACTUAL_CASE_NUMBER': np.int64,
            'ACTUAL_TREATMENT_CATEGORY': np.object,
            'ACTUAL_PATIENT_NUMBER': np.int64,
            'ACTUAL_PRIMARY_DIAGNOSIS_SID': np.int64,
            'ACTUAL_CASE_INSTITUTION':  np.object,
            #'ACTUAL_ADMISSION_DTE':  np.datetime64,
            'ACTUAL_LOS':  np.int64,
            'ACTUAL_ICU_HDU_LOS' :  np.float64,
            'ACTUAL_SURGICAL_CODE_1':  np.object,
            'ACTUAL_SURGICAL_CODE_2':  np.object,
            'ACTUAL_SURGICAL_CODE_3':  np.object,
            'ACTUAL_DIAGNOSIS_CODE':  np.object,
            #'ADM_DATE':  np.datetime64,
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
            #'DOB':  np.datetime64,
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
            'TOTAL_HOSP':  np.float64}
        self.col_dte = ['ACTUAL_ADMISSION_DTE', 'ADM_DATE', 'DOB']
        self.cols_to_drop = [
            'INST',  # represented by 'ACTUAL_CASE_INSTITUTION'
            # represented by 'ACTUAL_CASE_NUMBER'
            'ACTUAL_PATIENT_NUMBER', 'CASE_NO', 'PATIENT_NUMBER', 'ACTUAL_PRIMARY_DIAGNOSIS_SID',
            'CASE_TYPE', 'PATIENT_TYPE',  # single value features
            'ADM_DATE', 'ACTUAL_ADMISSION_DTE', 'DOB',  # aggregate to admission age
            'RESID_POSTALCODE', 'CONT_POSTAL',  # drop postal codes for now
            'Resid_Cty_Code']#temporary
        #self.bool_cols = ['NONRESID_FLAG']
        self.index_col = 'ACTUAL_CASE_NUMBER'
        self.target_col = 'TOTAL_HOSP'
        self.hosp_col = 'ACTUAL_CASE_INSTITUTION'
        self.data_folder_path = './data/uc3'
        self.dict_columns_categories = {
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
            'RESID_CTY': [x.upper() for x in constants_uc3.COUNTRY_CODES.values()],
            'NONRESID_FLAG': ['N', 'Y'],
            'IDENT_TYPE': [x.upper() for x in constants_uc3.ID_TYPES],
            'CONT_RELATION': [x.upper() for x in constants_uc3.RELATIONS]}
        self.drop_agg_cols = []
        self.dict_col_ohe = {}

    def transform_raw_data(self, raw_data_path):
        """
        Transform raw data into pre-processed raw data and save into '<hospital>_data_uc3.pkl'

        :param data_path: Path of data file
        :return: dictionary of file paths of each hospital data
        """

        # Extract directory
        self.data_folder_path = os.path.dirname(raw_data_path)

        # Read from csv
        df = pd.read_csv(raw_data_path, dtype=self.col_dtypes, parse_dates=self.col_dte)

        # set 'ACTUAL_CASE_NUMBER' as index
        df = self._set_index(df)

        # preprocess dataframe
        df = self._preprocess_raw_data(df)

        # scale and encode dataframe
        #df = self.__encode_categorical(df)

        df.to_pickle(f'{self.data_folder_path}/all_hosp_data_uc3.pkl')

        # split data by hospital
        dict_hosp_df = self._split_by_hosp(df)

        # save hospital data into pickle
        self.hosp_file_paths = {}
        for hosp, df_hosp in dict_hosp_df.items():
            self.hosp_file_paths[hosp] = f'{self.data_folder_path}/{hosp}_data_uc3.pkl'
            df_hosp.to_pickle(self.hosp_file_paths[hosp])

        return self.hosp_file_paths

    def transform_raw_test_data(self, df_raw, split_hosp=False):
        """
        Transform raw data into pre-processed raw data and save into '<hospital>_data_uc3.pkl'

        :param df: dataframe of test data
        :return: processed dataframe
        """

        df = df_raw.copy()

        df = df.astype(self.col_dtypes)
 
        for col in self.col_dte:
            df[col] = pd.to_datetime(df[col])
        
        # set 'ACTUAL_CASE_NUMBER' as index
        df = self._set_index(df)

        # preprocess dataframe
        df = self._preprocess_raw_data(df)

        # scale and encode dataframe
        #df = self.__encode_categorical(df)

        #df.to_pickle(f'{self.data_folder_path}/all_hosp_data_uc3.pkl')

        # split data by hospital
        if split_hosp:
            dict_hosp_df = self._split_by_hosp(df)

        # save hospital data into pickle
        #self.hosp_file_paths = {}
        #for hosp, df_hosp in dict_hosp_df.items():
            #self.hosp_file_paths[hosp] = f'{self.data_folder_path}/{hosp}_data_uc3.pkl'
            #df_hosp.to_pickle(self.hosp_file_paths[hosp])

            return dict_hosp_df
        else:
            return df

    def transform_train_test_data(self, data_file_path):
        """
        Get training data from pickle file path and split to features columns and target columns

        :param df: training dataframe
        :return: X_train, y_train
        """

        # Extract new file path
        data_dir = os.path.splitext(data_file_path)[0]
        X_train_file_path = data_dir + '_X_train_uc3.pkl'
        y_train_file_path = data_dir + '_y_train_uc3.pkl'
        X_test_file_path = data_dir + '_X_test_uc3.pkl'
        y_test_file_path = data_dir + '_y_test_uc3.pkl'

        # Read from pickle or csv
        if data_file_path.endswith('_uc3.pkl'):
            df = pd.read_pickle(data_file_path)
        else:
            df = pd.read_csv(data_file_path, dtype=self.col_dtypes)

            # set 'id' as index
            df = self._set_index(df)

            #df = self._convert_bool_col(df)

            df = self._process_null(df)

        df_train, df_test = self.train_test_split(df, random_state=0)

        # split dataframe into features, X, and target, y
        X_train, y_train = self._split_to_X_y(df_train)
        X_test, y_test = self._split_to_X_y(df_test)

        # fit scale dataframe
        self._fit_std_scaler(X_train)
        pickle.dump(self.std_scaler, open(data_dir + '_scaler.pkl', 'wb'))

        # transform scale dataframe
        self.std_scaler = pickle.load(open(data_dir + '_scaler.pkl', 'rb'))
        X_train = self._transform_std_scaler(X_train)
        X_test = self._transform_std_scaler(X_test)

        # fit label encode dataframe target column
        #self._fit_label_encoder(y_train)

        # transform label encode dataframe target column
        #y_train = self._transform_label_encoder(y_train)
        #y_test = self._transform_label_encoder(y_test)

        # fit encode dataframe
        self._fit_one_hot_encode(X_train)
        pickle.dump(self.one_hot_enc, open(data_dir + '_ohe.pkl', 'wb'))

        # transform encode dataframe
        self.one_hot_enc = pickle.load(open(data_dir + '_ohe.pkl', 'rb'))
        check_dir = os.path.dirname(data_file_path)
        for file in os.listdir(check_dir):
            file_path = os.path.join(check_dir, file)
            basename = os.path.basename(data_dir)
            if os.path.isfile(file_path) and file.endswith('_uc3.pkl') and (file.find(f'{basename}_X_train') >= 0):
                os.remove(file_path)
        #store = pd.HDFStore(X_train_file_path)
        slice_size = 10000
        slice_start = 0
        idx = 0
        X_train_file_paths = []
        while slice_start < len(X_train):
            X_train_file_path = data_dir + f'_X_train_{idx}_uc3.pkl'
            slice_end = slice_start + slice_size
            print(f'Processing train X rows: {slice_start} ~ {slice_end}')
            X_slice = self._transform_one_hot_encode(
                X_train.iloc[slice_start: slice_end, :].copy())
            if slice_start == 0:
                X_slice.to_pickle(X_train_file_path)
                #store.append('X_train', X_slice)
            else:
                X_slice.to_pickle(X_train_file_path)
                #store.append('X_train', X_slice)
            X_train_file_paths.append(X_train_file_path)
            slice_start = slice_end
            idx += 1

        if os.path.exists(y_train_file_path):
            os.remove(y_train_file_path)
        y_train.to_pickle(y_train_file_path)

        for file in os.listdir(check_dir):
            file_path = os.path.join(check_dir, file)
            basename = os.path.basename(data_dir)
            if os.path.isfile(file_path) and file.endswith('_uc3.pkl') and (file.find(f'{basename}_X_test') >= 0):
                os.remove(file_path)
        slice_start = 0
        idx = 0
        X_test_file_paths = []
        while slice_start < len(X_test):
            X_test_file_path = data_dir + f'_X_test_{idx}_uc3.pkl'
            slice_end = slice_start + slice_size
            print(f'Processing test X rows: {slice_start} ~ {slice_end}')
            X_slice = self._transform_one_hot_encode(
                X_test.iloc[slice_start: slice_end, :].copy())
            if slice_start == 0:
                X_slice.to_pickle(X_test_file_path)
                #store.append('X_test', X_slice)
            else:
                X_slice.to_pickle(X_test_file_path)
                #store.append('X_test', X_slice)
            X_test_file_paths.append(X_test_file_path)
            slice_start = slice_end
            idx += 1

        if os.path.exists(y_test_file_path):
            os.remove(y_test_file_path)
        y_test.to_pickle(y_test_file_path)

        return X_train_file_paths, y_train_file_path, X_test_file_paths, y_test_file_path

    def transform_test_data(self, df_X_test, scaler_pkl_file_path, ohe_pkl_file_path, feature_importance_file_path):
        """
        Get test data from file path and split to features columns and target column

        :param df_X_test: test dataframe features only
        :param scaler_pkl_file_path: scaler pickle file
        :param ohe_pkl_file_path: one hot encoder pickle file
        :param columns: feature columns to use after one hot encode
        :return: X_test
        """

        # transform scale dataframe
        self.std_scaler = pickle.load(open(scaler_pkl_file_path, 'rb'))
        X_test = self._transform_std_scaler(df_X_test)

        # transform encode dataframe
        self.one_hot_enc = pickle.load(open(ohe_pkl_file_path, 'rb'))
        X_slice = self._transform_one_hot_encode(X_test)

        feature_importances = np.load(feature_importance_file_path)
        return X_slice.loc[:, feature_importances > 0]


    def _set_index(self, df):
        '''
        Set index columns and drop duplicate indexes

        :param df: dataframe to set index
        :return: processed dataframe
        '''
        if self.index_col in df.columns:
            df.drop_duplicates(subset=self.index_col, inplace=True)
            df.set_index(self.index_col, verify_integrity=False,
                         drop=True, inplace=True)  # drop duplicates
        return df

    def _split_by_hosp(self, df):
        '''
        Split data by hospital

        :param df: dataframe to split
        :return: dictionary of hospital to dataframe corresponding to hospital
        '''
        dict_hosp_df = {}

        list_hosp = df[self.hosp_col].unique()

        for hosp in list_hosp:
            dict_hosp_df[hosp.upper()] = df[df[self.hosp_col] == hosp].drop(columns=self.hosp_col)

        return dict_hosp_df

    def _agg_admit_age(self, df):
        '''
        Aggregate admission age from ADMISSION_DTE and DOB

        :param df: dataframe to aggregate admission age feature from
        :return: dataframe that contains the admission age
        '''
        df['Admission_Age'] = df['ACTUAL_ADMISSION_DTE'].dt.year-df['DOB'].dt.year
        self.drop_agg_cols.append(['ACTUAL_ADMISSION_DTE', 'DOB'])
        return df

    def _split_to_X_y(self, df, col_target=None, col_features=None):
        '''
        Split dataframe into features and target

        :param df: dataframe to split features and target from
        :param col_target: name of column to use as target
        :param col_features: list of columns to use in features, default is None which use all
                             columns except target column
        :return: tuple of features numpy array and target numpy array
        '''

        if col_target is None:
            col_target = self.target_col
        
            if col_target not in df.columns:
                col_target = self.target_col

        if col_features is None:
            X = df.drop(col_target, axis=1)
        else:
            X = df[col_features]

        y = df[col_target]

        return X, y

    def _preprocess_raw_data(self, df):
        """
        Preprocess raw dataframe

        :param df: raw dataframe to preprocess
        :return: preprocessed dataframe
        """

        # convert country names to country codes
        df['RESID_CTY'] = df['RESID_CTY'].fillna('').astype('object')
        dict_country_code = {
            k.lower(): v for k, v in constants_uc3.COUNTRY_CODES.items()}
        df['Resid_Cty_Code'] = [dict_country_code[row['RESID_CTY'].lower()]
                                if (row['RESID_CTY'].lower() != '') and
                                (row['RESID_CTY'].lower() != 'unknown')
                                else ''
                                for index, row in df.iterrows()]

        df['ETBS_ICU_HDU_LOS'] = df['ETBS_ICU_HDU_LOS'].astype(str).str.replace('MISSING', '0').fillna('0').astype(np.float64)
        df['ETBS_LOS'] = df['ETBS_LOS'].astype(str).str.replace('MISSING', '0').fillna('0').astype(np.float64)
        # Aggregate features
        df = self._agg_admit_age(df)

        # drop columns
        df.drop(columns=self.cols_to_drop, inplace=True)

        # process null of NaN values
        df = self._process_null(df)

        cols_obj = df.select_dtypes(include=np.object).columns
        for col in cols_obj:
            df[col] = df[col].str.upper()

        #df = self.bin_column(df, col=self.target_col, bin_thresh=self.bin_threshold)

        return df

    def _process_null(self, df):
        '''
        Process null or NaN values

        :param df: raw dataframe to preprocess
        :return: processed dataframe
        '''
        cols_obj = df.select_dtypes(include=np.object).columns
        cols_num = df.select_dtypes(include=np.number).columns

        df[cols_obj] = df[cols_obj].fillna(
            '').replace(' ', '').astype('object')
        df[cols_num] = df[cols_num].fillna(0)

        return df

    def _fit_one_hot_encode(self, df):
        """
        One hot encode categorical features fit

        :param df: dataframe to encode categorical features
        :param dict_col_cat: dictionary of col and corresponding list of categories
        """

        if self.dict_columns_categories is not None:
            columns = []
            categories = []

            for column in self.dict_columns_categories:
                columns.append(column)
                categories.append(self.dict_columns_categories[column][:])

            ohe = OneHotEncoder(categories=categories, drop=None, sparse=False,
                                handle_unknown='ignore')
            ohe.fit(df[columns])
            self.one_hot_enc = (columns, ohe)

    def _transform_one_hot_encode(self, df):
        """
        One hot encode categorical features transform

        :param df: dataframe to encode categorical features
        :return: encoded dataframe
        """

        columns, ohe = self.one_hot_enc
        enc_ohe = ohe.transform(df[columns])
        enc_cols = [f'{columns[i]}_{ohe.categories_[i][j]}'
                    for i in range(len(columns))
                    for j in range(len(ohe.categories_[i]))
                    if (ohe.drop_idx_ is None) or
                       (ohe.drop_idx_[i] is None) or
                       (j != ohe.drop_idx_[i])]

        df_enc = pd.DataFrame(enc_ohe, index=df.index, columns=enc_cols)

        # drop one hot encoded columns
        df.drop(columns, axis=1, inplace=True)

        # merge new columns from onehot encode into dataframe
        #df = pd.concat([df, df_enc], axis=1)
        df = df.join(df_enc)

        return df

    def _fit_std_scaler(self, df):
        '''
        Standard scaler fit

        :param df: Dataframe to fit scaler
        '''
        self.std_scaler = StandardScaler()
        self.std_scaler.fit(df[df.select_dtypes(include=[np.number]).columns])

    def _transform_std_scaler(self, df):
        '''
        Standard scaler transform

        :param df: Datafraame to scale
        :return: scaled dataframe
        '''
        df[df.select_dtypes(include=[np.number]).columns] = self.std_scaler.transform(
            df[df.select_dtypes(include=[np.number]).columns])
        return df

    def _fit_label_encoder(self, df):
        '''
        label encoder fit target column

        :param df: Dataframe to fit encoder
        '''
        
        self.ord_enc = OrdinalEncoder()
        self.ord_enc.fit(df)

    def _transform_label_encoder(self, df):
        '''
        label encoder transform target column

        :param df: Dataframe to transform 
        :return: transformed dataframe
        '''
        col = self.target_col
        if col not in df.columns:
            col = self.target_col
        if col in df.columns:
            df = self.ord_enc.transform(df)
            return df.drop(columns=col)
        else:
            return df

    def train_test_split(self, df, test_frac=0.25, random_state=None):
        """
        Split dataset into training set and test set

        :param df: dataframe to split
        :param test_frac: fraction of dataset for test set
        :param random_state: random state for random seed
        :return: train dataframe, test dataframe
        """

        # set random seed
        np.random.seed(random_state)

        # check test size fraction is within range
        if test_frac is None:
            test_frac = 0.25
        elif test_frac > 1:
            test_frac = 1
        elif test_frac < 0:
            test_frac = 0

        # get indices of dataframe
        indices = df.index.tolist()

        # get train and test indices
        train_size = int(round((1 - test_frac) * len(indices)))
        if train_size > 0:
            train_indices = np.random.choice(
                indices, size=train_size, replace=False)
        else:
            train_indices = []

        # return train dataframe and test dataframe
        return df[df.index.isin(train_indices)], df[~df.index.isin(train_indices)]
