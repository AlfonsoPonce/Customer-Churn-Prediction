'''
This Script test functions in churn_library.py

Author: Alfonso Ponce
Date: 14/10/2023
'''


import logging

import pandas as pd
import pytest

from churn_library import import_data, perform_eda, \
    encoder_helper, perform_feature_engineering, train_models
import constants as C
# import churn_library_solution as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture
def data_path():
    '''
    return data path
    '''
    return str(C.DATA_PATH)


@pytest.fixture
def df():
    '''
    return dataframe with target variable transformed
    '''
    data = pd.read_csv(str(C.DATA_PATH))
    data[C.RESPONSE_VARIABLE] = data["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return data


@pytest.fixture
def response():
    '''
    return name of response variable
    '''
    return C.RESPONSE_VARIABLE


@pytest.fixture
def category_list():
    '''
    return list of categorical variables
    '''
    return C.CAT_COLUMNS_LIST


@pytest.fixture
def model_data():
    '''
    return data to be used during training of model
    '''
    dataframe = pd.read_csv(C.DATA_PATH)
    dataframe[C.RESPONSE_VARIABLE] = dataframe["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return perform_feature_engineering(dataframe, C.RESPONSE_VARIABLE)


def test_import(data_path):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data(data_path)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(df):
    '''
    test perform eda function
    '''

    try:
        logging.info("Trying to perform EDA")
        perform_eda(df)
        for image_name in ["Customer_Age_Histogram",
                           "Marital_Status_Histogram",
                           "Target_Variable_Histogram",
                           "Total_Trans_Ct_KDE",
                           "Heatmap"]:

            with open("images/eda/%s.png" % image_name, "r"):
                logging.info("EDA Performance: SUCCESS")

    except Exception as err:
        logging.error("Error while  in eda performance")
        raise err


def test_encoder_helper(df, category_list, response):
    '''
    test encoder helper
    '''

    try:
        logging.info("Encoding Dataframe")
        encoder_helper(df, category_list, response)
        logging.info("Encoding Dataframe: SUCCESS")
    except Exception as err:
        logging.error("Error while encoding dataframe")
        raise err


def test_perform_feature_engineering(df, response):
    '''
    test perform_feature_engineering
    '''

    try:
        logging.info("Trying Feature Engineering")
        perform_feature_engineering(df, response)
        logging.info("Feature Engineering: SUCCESS")
    except Exception as err:
        logging.error("Error while Feature engineering")
        raise err


def test_train_models(model_data):
    '''
    test train_models
    '''
    try:
        logging.info("Trying to train models and get results")
        x_train, x_test, y_train, y_test = model_data
        train_models(x_train, x_test, y_train, y_test)
        logging.info("Training models: SUCCESS")
    except Exception as err:
        logging.error("Error while trying to train models")
        raise err


if __name__ == "__main__":
    pass
