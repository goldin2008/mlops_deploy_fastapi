"""
Author: Ibrahim Sherif
Date: October, 2021
This script holds the test functions for dataset
"""
import great_expectations as ge


def test_columns_exist(data: ge.dataset.PandasDataset):
    """
    Test if all expected columns exists

    Args:
        data (ge.dataset.PandasDataset): Data to be tested
    """
    columns = [
        'age',
        'workclass',
        'fnlgt',
        'education',
        'education_num',
        'marital_status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital_gain',
        'capital_loss',
        'hours_per_week',
        'native_country',
        'salary'
    ]

    for column in columns:
        assert data.expect_column_to_exist(
            column)['success'], f"{column} does not exist"


def test_column_dtypes(data: ge.dataset.PandasDataset):
    """
    Tests if columns are of correct datatypes

    Args:
        data (ge.dataset.PandasDataset): Data to be tested
    """
    columns = {
        'age': 'int64',
        'workclass': 'object',
        'fnlgt': 'int64',
        'education': 'object',
        'education_num': 'int64',
        'marital_status': 'object',
        'occupation': 'object',
        'relationship': 'object',
        'race': 'object',
        'sex': 'object',
        'capital_gain': 'int64',
        'capital_loss': 'int64',
        'hours_per_week': 'int64',
        'native_country': 'object',
        'salary': 'object'
    }

    for column, dtype in columns.items():
        assert data.expect_column_values_to_be_of_type(
            column, dtype)['success'], f"{column} should be of type {dtype}"


def test_education_num_column(data: ge.dataset.PandasDataset):
    """
    Tests if education column values is within correct range

    Args:
        data (ge.dataset.PandasDataset): Data to be tested
    """
    assert data.expect_column_values_to_be_between(
        'education_num', 1, 17
    )['success'], "education_num column includes unknown category"


def test_marital_status(data: ge.dataset.PandasDataset):
    """
    Tests if martial status values are with correct categories

    Args:
        data (ge.dataset.PandasDataset):  Data to be tested
    """
    categs = [
        'never-married',
        'married-civ-spouse',
        'divorced',
        'married-spouse-absent',
        'separated',
        'married-af-spouse',
        'widowed'
    ]

    assert data.expect_column_distinct_values_to_equal_set(
        'marital_status', categs
    )['success'], "marital_status column includes unknown category"


def test_label_salary(data: ge.dataset.PandasDataset):
    """
    Tests if salary label contains only two correct classes

    Args:
        data (ge.dataset.PandasDataset):  Data to be tested
    """
    assert data.expect_column_distinct_values_to_equal_set(
        'salary', ['<=50k', '>50k']
    )['success'], "salary column includes more than two classes"


def test_hours_per_week_range(data: ge.dataset.PandasDataset):
    """
    Tests if hours_per_week column values are within correct range

    Args:
        data (ge.dataset.PandasDataset):  Data to be tested
    """
    data.expect_column_values_to_be_between
    assert data.expect_column_values_to_be_between(
        'hours_per_week', 1, 99
    )['success'], "hours_per_week column is not within range of 1 and 99"
