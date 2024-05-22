import numpy as np
import pandas as pd


def _preprocess(df):
    """
    Preprocess the population data by removing missing values and
    converting the data types of the columns.
    """

    df['year'] = df['year'].astype(str)
    df['gdp_per_capita'] = df['gdp_per_capita'].astype(float)
    df = df.set_index('year')

    df['urban_population_rate'] = df['urban_population_rate'].interpolate()

    return df


def read_population_data(file_path):
    """
    Read population data from CSV file and return a pandas DataFrame.
    """
    df = pd.read_csv(file_path)

    df = _preprocess(df)

    return df


def ratio_and_diff(df):
    return (df.divide(df.shift(1))[1:]).diff(1)[1:]


if __name__ == '__main__':
    df = read_population_data('data/data.csv')
    print(df.head())
    print(len(df))
    print(type(df['population']['2001']))
    print(type(df['gdp_per_capita']['2001']))
