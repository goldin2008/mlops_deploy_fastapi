"""
Author: Ibrahim Sherif
Date: October, 2021
This script holds the conftest data used with pytest module
"""
import os
import pytest
import pandas as pd
import great_expectations as ge
from sklearn.model_selection import train_test_split

import config
from pipeline.data import get_clean_data


@pytest.fixture(scope='session')
def data():
    """
    Data loaded from csv file used for tests

    Returns:
        df (ge.DataFrame): Data loaded from csv file
    """
    if not os.path.exists(config.DATA_DIR):
        pytest.fail(f"Data not found at path: {config.DATA_DIR}")

    X_df, y_df = get_clean_data(config.DATA_DIR)
    X_df['salary'] = y_df
    X_df['salary'] = X_df['salary'].map({1: '>50k', 0: '<=50k'})

    df = ge.from_pandas(X_df)

    return df


@pytest.fixture(scope='session')
def sample_data():
    """
    Sampled data from csv file used for tests

    Returns:
        X_train: Features train data
        X_test: Features test data
        y_train: Labels train data
        y_test: Labels test data
    """
    if not os.path.exists(config.DATA_DIR):
        pytest.fail(f"Data not found at path: {config.DATA_DIR}")

    data_df = pd.read_csv(config.DATA_DIR, nrows=10)

    # chaning column names to use _ instead of -
    columns = data_df.columns
    columns = [col.replace('-', '_') for col in columns]
    data_df.columns = columns

    # make all characters to be lowercase in string columns
    data_df = data_df.applymap(
        lambda s: s.lower() if isinstance(s, str) else s)

    data_df['salary'] = data_df['salary'].map({'>50k': 1, '<=50k': 0})

    y_df = data_df.pop('salary')
    X_df = data_df

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.3, random_state=config.RANDOM_STATE, stratify=y_df)

    return X_train, X_test, y_train, y_test
