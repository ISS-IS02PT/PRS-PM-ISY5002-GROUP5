import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

class Datapipeline():
    
    COLS_TO_DROP = ['PAYER_NAME_1', 'PAYER_NAME_2', 'PAYER_NAME_3', 'PAYER_NAME_4', 'PAYER_NAME_5',
                    'DISCHARGE_DTE', 'DISCHARGE_TYPE_DESC',
                    'DOCTOR_NAME', 'SPECIALTY_DESC',
                    'TOSP_STRING', 'TOSP_CODE1', 'TOSP_CODE2', 'TOSP_CODE4',
                    'TOSP_DESC1', 'TOSP_DESC2', 'TOSP_DESC3', 'TOSP_DESC4',
                    'RESID_CTY', 'RESID_POSTALCODE', # convert to latitude longitude
                    'PATIENT_SID', 'DRG_DESC',
                    'PAYER_NAME1_V', 'PAYER_NAME2_V', 'PAYER_NAME3_V', 'PAYER_NAME4_V',
                    'PACKAGE_CODE', 'PACKAGE_DESC', 'PACKAGE_DESC1', 'PACKAGE_DESC2',
                    'ICDCODE_STRING']
    COLS_TO_USE_NN = {}
    INDEX_COL = 'CASE_NUMBER'
    TARGET_COL = 'WRITE_OFF'
    HOSP_COL = 'INSTITUTION'
    DATA_FOLDER = './data/'
    
    def __init__(self):
        self.drop_agg_cols = []
    
    def transform_raw_data(self, raw_data_path):
        """
        Transform raw data into pre-processed raw data and save into '<hospital>_data.csv'

        :param data_path: Path of data file
        :return: dictionary of file paths of each hospital data
        """
        # Read from csv
        df = pd.read_excel(raw_data_path)
        
        # set 'CASE_NUMBER' as index
        df = self._set_index(df)
        
        # preprocess dataframe
        #df = self._preprocess_data(df)
        
        # scale and encode dataframe
        #df = self.__encode_categorical(df)
        
        df.to_csv(f'{self.DATA_FOLDER}all_data.csv')
        
        # split data by hospital
        dict_hosp_df = self._split_by_hosp(df)
        
        #save hospital data into csv
        self.hosp_file_paths = {}
        for hosp, df_hosp in dict_hosp_df.items():
            self.hosp_file_paths[hosp] = f'{self.DATA_FOLDER}{hosp}_data.csv'
            df_hosp.to_csv(self.hosp_file_paths[hosp])
        
        return (, self.hosp_file_paths)
       
        
    def transform_hosp_data(self, hosp_data_path):
        """
        Get hospital data from file path and split to features columns and target columns

        :param train_data_path: path to training data file
        :return: X_train, y_train
        """

        # read train data file
        df = pd.read_csv(train_data_path)
        
        # set 'id' as index
        df = self.__set_index(df)
        
        # preprocess dataframe
        df = self.__preprocess_data(df)
        
        # fit encode dataframe
        self.__fit_ordinal_encode(df)
        self.__fit_one_hot_encode(df)
        
        # transform encode dataframe
        df = self.__transform_ordinal_encode(df)
        df = self.__transform_one_hot_encode(df)
         
        # split dataframe into 
        X_train, y_train = self.__split_to_X_y(df)
        
        return X_train, y_train
    

    def transform_test_data(self, test_data_path):
        """
        Get test data from file path and split to features columns and target column

        :param test_data_path: path to test data file
        :return: X_test, y_test
        """

        # read train data file
        df = pd.read_csv(test_data_path)
        
        # set 'id' as index
        df = self.__set_index(df)
        
        # preprocess dataframe
        df = self.__preprocess_data(df)
        
        # transform encode dataframe
        df = self.__transform_ordinal_encode(df)
        df = self.__transform_one_hot_encode(df)
        
        # split dataframe into 
        X_test, y_test = self.__split_to_X_y(df)
        
        return X_test, y_test
    
    def _set_index(self, df):
        '''
        Set index columns
        :param df: dataframe to set index
        :return: processed dataframe
        '''
        df.drop_duplicates(subset=[self.INDEX_COL], inplace=True)
        df.set_index(self.INDEX_COL, verify_integrity=False, drop=True, inplace=True) # drop duplicates
        return df
    
    def _split_by_hosp(self, df):
        '''
        Split data by hospital
        
        :param df: dataframe to split
        :return: dictionary of hospital to dataframe corresponding to hospital
        '''
        dict_hosp_df = {}
        
        list_hosp = df[self.HOSP_COL].unique()
        
        for hosp in list_hosp:
            dict_hosp_df[hosp] = df[df[hosp]]
            
        return dict_hosp_df
    
    def _agg_admit_age(df):
        '''
        Aggregate 
        '''
        df['Admission_Age'] = df['ADMISSION_DTE'].dt.year-df['DOB'].dt.year
        self.drop_agg_cols.append(['Admission_Age', 'DOB'])
        return df
    
    def __split_to_X_y(self, df):
        col_target = 'resale_price'
        X = df.drop(col_target, axis=1)
        y = df[col_target]
        return X.values, y.values
    
    def _preprocess_data(self, df):
        """
        Preprocess dataframe
        :param df: dataframe to preprocess
        :return: preprocessed dataframe
        """
        # drop columns
        for col in self.cols_to_drop:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)     
        
        # month -> trans_year, trans_month
        df['trans_date'] = pd.to_datetime(df.month, format='%Y-%m')

        # floor_area_sqm -> float
        df.floor_area_sqm.astype('float')

        # lease_commence_date -> int
        df.lease_commence_date.astype('int')

        # remaining_lease -> lease_years (float)
        df['remaining_lease_years'] = df.remaining_lease.str \
                                        .extract(r'(\d*) years (\d*) months|(\d*) years|(\d*)') \
                                        .astype(np.number) \
                                        .apply(lambda row: self.__convert_lease_date(row), axis=1)

        # resale_price -> float
        df.resale_price.astype('float')

        # 99 - remaining_lease -> age_at_transaction
        df['age_at_trans'] = 99 - df.remaining_lease_years
        
        # transaction date -> trans_year and trans_month
        df['trans_year'] = df.trans_date.dt.year
        df['trans_month'] = df.trans_date.dt.month
        
        # remove punctuation from 'Premium Apartment.'
        df['flatm_name'] = df.flatm_name.str.replace('Premium Apartment.', 'Premium Apartment',
                                                     regex=False)

        # group 'Type S2', 'Type S1' to 'Type S'
        list_type_s = ['Type S2', 'Type S1']
        for name in list_type_s:
            df['flatm_name'] = df.flatm_name.str.replace(name, 'Type S', regex=False)

        # group 'Model A-Maisonette', 'Adjoined flat', 'Terrace', 'Multi Generation',
        # 'Improved-Maisonette', 'Premium Apartment Loft', 'Premium Maisonette', '2-room'
        # to 'Others'
        list_others = ['Model A-Maisonette', 'Adjoined flat', 'Terrace', 'Multi Generation',
                       'Improved-Maisonette', 'Premium Apartment Loft', 'Premium Maisonette', '2-room']
        
        for name in list_others:
            df['flatm_name'] = df.flatm_name.str.replace(name, 'Others', regex=False)

        # drop 'month', 'lease_commence_date', 'remaining_lease', 'trans_date', 'remaining_lease_years'
        df.drop(['month', 'lease_commence_date', 'remaining_lease', 'trans_date', 'remaining_lease_years'], axis=1, inplace=True)

        return df

    def __convert_lease_date(self, row):
        """
        Convert 'remaining_lease' into remaining lease years(float)
        Variations of data in column is '67 years 07 months', '67 years', '67'
        
        :param row: date string row to convert
        :return: datetime format of the 'remaining_lease' date
        """
        
        # 1st 2 columns are "67 years 07 months"
        if len(row) > 0 and ~np.isnan(row[0]):
            if len(row) > 1 and ~np.isnan(row[1]):
                return int(row[0])+int(row[1])/12.0
            else:
                return int(row[0])
        
        # 3rd column is "67 years"
        elif len(row) > 2 and ~np.isnan(row[2]):
            return int(row[2])
        
        # 4th column is "67"
        elif len(row) > 3 and ~np.isnan(row[3]):
            return int(row[3])
        else:
            return np.nan
    
    def __fit_ordinal_encode(self, df):
        """
        Ordinal encode categorical features fit
        
        :param df: dataframe to encode categorical features
        """

        # label encode 'flat_type', 'storey_range'
        colLabel = ['flat_type', 'storey_range']
        categories = [['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION'],
                      ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21',
                       '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42',
                       '43 TO 45', '46 TO 48', '49 TO 51']]
        oe = OrdinalEncoder(categories=categories)
        oe.fit(df[colLabel])
        self.ord_enc = (colLabel, oe)

    def __transform_ordinal_encode(self, df):
        """
        Ordinal encode categorical features transform
        
        :param df: dataframe to encode categorical features
        :return: encoded dataframe
        """
        colLabel, oe = self.ord_enc
        df[colLabel] = oe.transform(df[colLabel])
        return df
    
    def __fit_one_hot_encode(self, df, cols, categories):
    def __fit_one_hot_encode(self, df, cols, categories):
        """
        One hot encode categorical features fit
        
        :param df: dataframe to encode categorical features
        """

        # one-hot encode categorical features
        colCat = ['town_name', 'flatm_name']#df.select_dtypes(include=np.object).columns
        categories = [['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG',
                       'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG',
                       'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 
                       'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
                       'TOA PAYOH', 'WOODLANDS', 'YISHUN'],
                      ['Apartment', 'DBSS', 'Improved', 'Maisonette', 'Model A', 'Model A2',
                       'New Generation', 'Others', 'Premium Apartment', 'Simplified', 'Standard', 'Type S']]
        ohe = OneHotEncoder(categories=categories, drop='first', sparse=False)
        ohe.fit(df[colCat])
        self.one_hot_enc = (colCat, ohe)
    
    def __transform_one_hot_encode(self, df):
        """
        One hot encode categorical features transform
        
        :param df: dataframe to encode categorical features
        :return: encoded dataframe
        """
        colCat, ohe = self.one_hot_enc
        enc_ohe = ohe.transform(df[colCat])
        enc_cols = [f'{colCat[i]}_{ohe.categories_[i][j]}'
                    for i in range(len(colCat))
                    for j in range(len(ohe.categories_[i]))
                    if j != ohe.drop_idx_[i]]
         
        # merge new columns from onehot encode into dataframe
        df = pd.concat([df, pd.DataFrame(enc_ohe, index=df.index, columns=enc_cols)], axis=1)     
        # drop one hot encoded columns
        df.drop(colCat, axis=1, inplace=True)

        return df
    
    def train_test_split(self, df, test_frac=None, random_state=None):
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
            train_indices = np.random.choice(indices, size=train_size, replace=False)
        else:
            train_indices = []
            
        # return train dataframe and test dataframe
        return df[df.index.isin(train_indices)], df[~df.index.isin(train_indices)]