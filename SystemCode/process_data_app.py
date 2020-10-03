import argparse
from datapipeline import Datapipeline
import pandas as pd
import os

def process_data(data_path):
    dpl = Datapipeline()
    print('transform_raw_data')
    dict_hosp_datapaths = dpl.transform_raw_data(data_path)
    for hosp in dict_hosp_datapaths:
        print(f"read_csv(dict_hosp_datapaths['{hosp}'])")
        df = pd.read_csv(dict_hosp_datapaths[hosp], dtype=dpl.col_dtypes)

        df_train, df_test = dpl.train_test_split(df, random_state=0)
        X_train, y_train = dpl.transform_train_data(df_train)

def get_argparser():
    parser = argparse.ArgumentParser()

    # data path
    parser.add_argument('--data_path',
                        default='./SystemCode/Data Exploration/data/'
                        + 'ParkwaySampleDataForProject_09_withTOSP3.xlsx',
                        help="raw data path")

    return parser


def get_params(args):

    if args is not None:
        dict_args = vars(args)

        if 'data_path' in dict_args:
            if dict_args['data_path'] is not None:
                data_path = dict_args['data_path']

            del dict_args['data_path']

    return data_path

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    data_path = get_params(args)
    process_data(data_path)
