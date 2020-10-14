import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from statistics import variance, mean, stdev
import constants
import os
from dask import dataframe as dd
import dask


class Datapipeline():
    '''
    Data pipeline class
    '''

    def __init__(self):
        '''
        Initialise
        '''
        self.col_dtypes = {
            'INSTITUTION': np.object,
            'CASE_NUMBER': np.int64,
            'TOTAL_PAID_AMT': np.float64,
            'PAYER_CODE_1': np.object,
            'PAYER_NAME_1': np.object,
            'PAYER_1_PAID_AMT': np.float64,
            'PAYER_CODE_2': np.object,
            'PAYER_NAME_2': np.object,
            'PAYER_2_PAID_AMT': np.float64,
            'PAYER_CODE_3': np.object,
            'PAYER_NAME_3': np.object,
            'PAYER_3_PAID_AMT': np.float64,
            'PAYER_CODE_4': np.object,
            'PAYER_NAME_4': np.object,
            'PAYER_4_PAID_AMT': np.float64,
            'PAYER_CODE_5': np.object,
            'PAYER_NAME_5': np.object,
            'PAYER_5_PAID_AMT': np.float64,
            'CASE_TYPE': np.object,
            'BED_TYPE': np.object,
            'REFERRAL_TYPE': np.object,
            'TREATMENT_CATEGORY': np.object,
            'ADMISSION_DTE': np.datetime64,
            'ADMISSION_TYPE': np.object,
            'DISCHARGE_DTE': np.datetime64,
            'DISCHARGE_TYPE': np.int64,
            'DISCHARGE_TYPE_DESC': np.object,
            'LOS_DAYS': np.int64,
            'DOCTOR_CODE': np.object,
            'DOCTOR_NAME': np.object,
            'SPECIALTY_CODE': np.object,
            'SPECIALTY_DESC': np.object,
            'SPECIALTY_GRP': np.object,
            'TOSP_COUNT': np.int64,
            'TOSP_STRING': np.object,
            'TOSP_CODE1': np.object,
            'TOSP_CODE2': np.object,
            'TOSP_CODE3': np.object,
            'TOSP_CODE4': np.object,
            'TOSP_DESC1': np.object,
            'TOSP_DESC2': np.object,
            'TOSP_DESC3': np.object,
            'TOSP_DESC4': np.object,
            'NATIONALITY': np.object,
            'RESID_CTY': np.object,
            'RESID_POSTALCODE': np.object,
            'DOB': np.datetime64,
            'NONRESID_FLAG': np.object,
            'PATIENT_SID': np.int64,
            'PATIENT_NUMBER': np.int64,
            'GENDER': np.object,
            'DECEASED_FLAG': np.object,
            'MARITAL_STATUS': np.object,
            'RELIGION': np.object,
            'LANGUAGE': np.object,
            'VIP_FLAG': np.object,
            'RACE': np.object,
            'DRG_CODE': np.object,
            'DRG_DESC': np.object,
            'PAYER_CODE1_V': np.object,
            'PAYER_NAME1_V': np.object,
            'PAYER_CODE2_V': np.object,
            'PAYER_NAME2_V': np.object,
            'PAYER_CODE3_V': np.object,
            'PAYER_NAME3_V': np.object,
            'PAYER_CODE4_V': np.object,
            'PAYER_NAME4_V': np.object,
            'PACKAGE_CODE': np.object,
            'PACKAGE_PRICE': np.float64,
            'PACKAGE_EXCL': np.float64,
            'PACKAGE_ADJ': np.float64,
            'PACKAGE_DESC': np.object,
            'PACKAGE_CODE1': np.object,
            'PACKAGE_CODE2': np.object,
            'PACKAGE_DESC1': np.object,
            'PACKAGE_DESC2': np.object,
            'ICD_CODE1': np.object,
            'ICD_CODE2': np.object,
            'ICD_CODE3': np.object,
            'ICDCODE_STRING': np.object,
            'PROF_FEE': np.float64,
            'TOTAL_FEES': np.float64,
            'WRITE_OFF': np.float64}
        self.cols_to_drop = [
            'PAYER_NAME_1', 'PAYER_NAME_2', 'PAYER_NAME_3', 'PAYER_NAME_4', 'PAYER_NAME_5',
            'CASE_TYPE',  # only 1 value
            'DISCHARGE_DTE', 'DISCHARGE_TYPE_DESC', 'ADMISSION_DTE',
            'DOCTOR_NAME', 'SPECIALTY_DESC',
            'TOSP_STRING',
            'TOSP_DESC1', 'TOSP_DESC2', 'TOSP_DESC3', 'TOSP_DESC4',
            'LANGUAGE',  # only 1 value
            'RESID_CTY', 'RESID_POSTALCODE',  # convert to latitude longitude
            'PATIENT_SID', 'PATIENT_NUMBER',
            'DRG_DESC', 'DRG_CODE',  # removed because it is concluded after discharge
            'PAYER_NAME1_V', 'PAYER_NAME2_V', 'PAYER_NAME3_V', 'PAYER_NAME4_V',
            'PAYER_CODE1_V', 'PAYER_CODE2_V', 'PAYER_CODE3_V', 'PAYER_CODE4_V',
            'PACKAGE_CODE', 'PACKAGE_DESC', 'PACKAGE_DESC1', 'PACKAGE_DESC2',
            'ICDCODE_STRING', 'DOB',
            'Resid_Cty_Code']#temporary
        self.cols_num_to_object = [
            'SPECIALTY_CODE', 'SPECIALTY_GRP', 'PACKAGE_CODE', 'PACKAGE_CODE1', 'PACKAGE_CODE2',
            'PAYER_CODE_1', 'PAYER_CODE_2', 'PAYER_CODE_3', 'PAYER_CODE_4', 'PAYER_CODE_5']
        self.bool_cols = ['NONRESID_FLAG', 'DECEASED_FLAG', 'VIP_FLAG']
        self.index_col = 'CASE_NUMBER'
        self.target_col = 'WRITE_OFF'
        self.target_col_binned = f'bin_{self.target_col}'
        self.hosp_col = 'INSTITUTION'
        self.data_folder_path = './data'
        self.dict_columns_categories = {
            'PAYER_CODE_1': constants.PAYER_CODES,
            'PAYER_CODE_2': constants.PAYER_CODES,
            'PAYER_CODE_3': constants.PAYER_CODES,
            'PAYER_CODE_4': constants.PAYER_CODES,
            'PAYER_CODE_5': constants.PAYER_CODES,
            'BED_TYPE': constants.BED_TYPES,
            'REFERRAL_TYPE': constants.REFERRAL_TYPES,
            'TREATMENT_CATEGORY': constants.TREATMENT_TYPES,
            'ADMISSION_TYPE': constants.ADMISSION_TYPES,
            'DOCTOR_CODE': constants.DOCTOR_CODES,
            'SPECIALTY_CODE': constants.SPECIALTY_CODES,
            'SPECIALTY_GRP': constants.SPECIALTY_GROUPS,
            'TOSP_CODE1': constants.TOSP_CODES,
            'TOSP_CODE2': constants.TOSP_CODES,
            'TOSP_CODE3': constants.TOSP_CODES,
            'TOSP_CODE4': constants.TOSP_CODES,
            'NATIONALITY': constants.NATIONALITIES,
            'GENDER': constants.GENDER_TYPES,
            'MARITAL_STATUS': constants.MARITAL_STATUSES,
            'RELIGION': constants.RELIGION_TYPES,
            'RACE': constants.RACE_TYPES,
            # 'DRG_CODE' : constants.DRG_CODES,
            'PACKAGE_CODE1': constants.PACKAGE_CODES,
            'PACKAGE_CODE2': constants.PACKAGE_CODES,
            'ICD_CODE1': constants.ICD_CODES,
            'ICD_CODE2': constants.ICD_CODES,
            'ICD_CODE3': constants.ICD_CODES}
        self.bin_threshold = [0, 100, 1000, 10000]#[0]
        self.drop_agg_cols = []
        self.dict_col_ohe = {}

    def transform_raw_data(self, raw_data_path):
        """
        Transform raw data into pre-processed raw data and save into '<hospital>_data.pkl'

        :param data_path: Path of data file
        :return: dictionary of file paths of each hospital data
        """

        # Extract directory
        self.data_folder_path = os.path.dirname(raw_data_path)

        # Read from excel
        df = pd.read_excel(raw_data_path, dtype=self.col_dtypes)

        # set 'CASE_NUMBER' as index
        df = self._set_index(df)

        # preprocess dataframe
        df = self._preprocess_raw_data(df)

        # scale and encode dataframe
        #df = self.__encode_categorical(df)

        df.to_pickle(f'{self.data_folder_path}/all_hosp_data.pkl')

        # split data by hospital
        dict_hosp_df = self._split_by_hosp(df)

        # save hospital data into pickle
        self.hosp_file_paths = {}
        for hosp, df_hosp in dict_hosp_df.items():
            self.hosp_file_paths[hosp] = f'{self.data_folder_path}/{hosp}_data.pkl'
            df_hosp.to_pickle(self.hosp_file_paths[hosp])

        return self.hosp_file_paths

    def transform_train_test_data(self, data_file_path):
        """
        Get training data from pickle file path and split to features columns and target columns

        :param df: training dataframe
        :return: X_train, y_train
        """

        # Extract new file path
        data_dir = os.path.splitext(data_file_path)[0]
        X_train_file_path = data_dir + '_X_train.pkl'
        y_train_file_path = data_dir + '_y_train.pkl'
        X_test_file_path = data_dir + '_X_test.pkl'
        y_test_file_path = data_dir + '_y_test.pkl'

        # Read from pickle or csv
        if data_file_path.endswith('.pkl'):
            df = pd.read_pickle(data_file_path)
        else:
            df = pd.read_csv(data_file_path, dtype=self.col_dtypes)

            # set 'id' as index
            df = self._set_index(df)

            # convert features from numeric to object as they should be categorical features
            df = self.convert_numeric_to_object(df)

            df = self._convert_bool_col(df)

            df = self._process_null(df)

        df_train, df_test = self.train_test_split(df, random_state=0)

        # split dataframe into features, X, and target, y
        X_train, y_train = self._split_to_X_y(df_train)
        X_test, y_test = self._split_to_X_y(df_train)

        # fit scale dataframe
        self._fit_std_scaler(X_train)

        # transform scale dataframe
        X_train = self._transform_std_scaler(X_train)
        X_test = self._transform_std_scaler(X_test)

        # fit label encode dataframe target column
        #self._fit_label_encoder(y_train)

        # transform label encode dataframe target column
        #y_train = self._transform_label_encoder(y_train)
        #y_test = self._transform_label_encoder(y_test)

        # fit encode dataframe
        self._fit_one_hot_encode(X_train)

        # transform encode dataframe
        check_dir = os.path.dirname(data_file_path)
        for file in os.listdir(check_dir):
            file_path = os.path.join(check_dir, file)
            basename = os.path.basename(data_dir)
            if os.path.isfile(file_path) and file.endswith('.pkl') and (file.find(f'{basename}_X_train') >= 0):
                os.remove(file_path)
        #store = pd.HDFStore(X_train_file_path)
        slice_size = 10000
        slice_start = 0
        idx = 0
        X_train_file_paths = []
        while slice_start < len(X_train):
            X_train_file_path = data_dir + f'_X_train_{idx}.pkl'
            slice_end = slice_start + slice_size
            print(f'Processing train X rows: {slice_start} ~ {slice_end}')
            X_slice = self._transform_one_hot_encode(
                X_train.iloc[slice_start: slice_end, :])
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
            if os.path.isfile(file_path) and file.endswith('.pkl') and (file.find(f'{basename}_X_test') >= 0):
                os.remove(file_path)
        slice_start = 0
        idx = 0
        X_test_file_paths = []
        while slice_start < len(X_test):
            X_test_file_path = data_dir + f'_X_test_{idx}.pkl'
            slice_end = slice_start + slice_size
            print(f'Processing test X rows: {slice_start} ~ {slice_end}')
            X_slice = self._transform_one_hot_encode(
                X_test.iloc[slice_start: slice_end, :])
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

    def transform_test_data(self, df):
        """
        Get test data from file path and split to features columns and target column

        :param df: test dataframe
        :return: X_test, y_test
        """

        # set 'id' as index
        df = self._set_index(df)

        # convert features from numeric to object as they should be categorical features
        df = self.convert_numeric_to_object(df)

        df = self._process_null(df)

        # split dataframe into features, X, and target, y
        X_test, y_test = self._split_to_X_y(df)

        # fit encode dataframe
        self._fit_one_hot_encode(X_test)

        # transform encode dataframe
        X_test = self._transform_one_hot_encode(X_test)

        # fit scale dataframe
        self._fit_std_scaler(X_test)

        # transform scale dataframe
        X_test = self._transform_std_scaler(X_test)

        return X_test, y_test

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
            dict_hosp_df[hosp] = df[df[self.hosp_col]
                                    == hosp].drop(columns=self.hosp_col)

        return dict_hosp_df

    def _agg_admit_age(self, df):
        '''
        Aggregate admission age from ADMISSION_DTE and DOB

        :param df: dataframe to aggregate admission age feature from
        :return: dataframe that contains the admission age
        '''
        df['Admission_Age'] = df['ADMISSION_DTE'].dt.year-df['DOB'].dt.year
        self.drop_agg_cols.append(['Admission_Age', 'DOB'])
        return df

    def convert_numeric_to_object(self, df, features=None):
        '''
        Convert features in dataframe into object type

        :param df: dataframe to process
        :return: processed dataframe
        '''
        if features is None:
            features = self.cols_num_to_object[:]

        dict_features = {
            feature: 'object' for feature in features if feature in df.columns}
        df = df.astype(dict_features)

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
            col_target = self.target_col_binned
        
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

        # convert features from numeric to object as they should be categorical features
        df = self.convert_numeric_to_object(df)

        df = self._convert_bool_col(df)

        # convert country names to country codes
        df['RESID_CTY'] = df['RESID_CTY'].fillna('').astype('object')
        dict_country_code = {
            k.lower(): v for k, v in constants.COUNTRY_CODES.items()}
        df['Resid_Cty_Code'] = [dict_country_code[row['RESID_CTY'].lower()]
                                if (row['RESID_CTY'].lower() != '') and
                                (row['RESID_CTY'].lower() != 'unknown')
                                else ''
                                for index, row in df.iterrows()]

        # Aggregate features
        df = self._agg_admit_age(df)

        # drop columns
        df.drop(columns=self.cols_to_drop, inplace=True)

        # process null of NaN values
        df = self._process_null(df)

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

    def _convert_bool_col(self, df):
        for col in self.bool_cols:
            df[col] = df[col].replace({'N': '0', 'Y': '1'}).astype(int)
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
        col = self.target_col_binned
        if col not in df.columns:
            col = self.target_col
        if col in df.columns:
            df = self.ord_enc.transform(df)
            return df.drop(columns=col)
        else:
            return df

    def bin_column(self, df, col=None, bin_thresh=[0]):
        '''
        Split column values by thresholds

        :param df: dataframe to process
        :param col: column in dataframe to bin
        :param bin_thresh: list of thresholds to bin into, default [0]
        '''

        if col is None:
            col = self.target_col

        if col in df.columns:
            if bin_thresh is None:
                bin_intervals = [0]
            else:
                bin_intervals = bin_thresh[:]

            bin_intervals.sort()
            bin_intervals.insert(0, np.NINF)
            bin_intervals.append(np.Inf)

            df1 = df.copy()
            df1[self.target_col_binned] = pd.cut(df1[col], bin_intervals).astype('object')
            return df1.drop(columns=col)
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

    def sample_data(self, df, sample_frac=0.1, n_searches=1000):
        '''
        Extract a sample that best represent the data

        :param sample_frac: fraction of the dataset to sample, default = 0.1
        :param n_searches: number of random sampling to search for the best sample, default = 1000
        :return: sample dataframe
        '''
        df1 = df.copy()
        col_obj = df1.select_dtypes(include=['object', 'category']).columns
        for col in col_obj:
            df1[col] = pd.factorize(df1[col])[0] + 1

        rand_tstats = {}
        rand_f_test = {}
        sample_size = int(round(abs(sample_frac * df1.shape[0])))
        random.seed(0)
        rand_list = random.sample(range(2**32 - 1), n_searches)
        for rand in rand_list:
            np.random.seed(rand)
            sample_indices = np.random.choice(
                df1.shape[0], size=sample_size, replace=False)
            df_sample = df1.iloc[sample_indices]
            tstats = []
            f_tests = []
            for col in df1.columns:
                var_df = df1[col].std()**2
                var_df_sample = df_sample[col].std()**2

                if (var_df != 0) or (var_df_sample != 0):
                    tstats.append(abs((df1[col].mean() - df_sample[col].mean()) / (
                        (var_df / len(df1[col])) + (var_df_sample / len(df_sample[col])))**0.5))
                else:
                    tstats.append(0.0)

                if var_df_sample != 0:
                    f_test = var_df / var_df_sample
                else:
                    f_test = 1.0
                f_tests.append(abs(1-f_test))

            max_tstats = max(tstats)
            max_f_test = max(f_tests)
            rand_tstats[rand] = max_tstats
            rand_f_test[rand] = max_f_test

        mean_tstats = mean(rand_tstats.values())
        std_tstats = stdev(rand_tstats.values())
        mean_f_test = mean(rand_f_test.values())
        std_f_test = stdev(rand_f_test.values())
        std_rand_tstats = {
            key:
            abs(value - mean_tstats) / std_tstats for (key, value) in rand_tstats.items()}
        std_rand_f_test = {
            key:
            abs(value - mean_f_test) / std_f_test for (key, value) in rand_f_test.items()}
        std_rand_tstats_f_test_ms = {
            key:
            (((abs(value - mean_tstats) / std_tstats) ** 2
              + (abs(std_rand_f_test[key] - mean_f_test) / std_f_test) ** 2) / 2) ** 0.5
            for (key, value) in std_rand_tstats.items()}
        min_key = min(std_rand_tstats_f_test_ms,
                      key=std_rand_tstats_f_test_ms.get)

        np.random.seed(min_key)
        sample_indices = np.random.choice(
            df1.shape[0], size=sample_size, replace=False)
        return df.iloc[sample_indices]
