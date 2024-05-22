import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
# 创建一个标准化器
scaler = StandardScaler()

def do_standardize(df: pd.DataFrame) -> pd.DataFrame:
    # 使用标准化器拟合并转换DataFrame
    scaled_df = scaler.fit_transform(df)
    return pd.DataFrame(scaled_df, columns=df.columns, index=df.index)


def inverse_standardize(df: pd.DataFrame) -> pd.DataFrame:
    # 使用标准化器逆转换DataFrame
    # cols = ["population", "urban_population_rate", "gdp_per_capita", "unemployment_rate"]
    # scaled_df = df.copy(deep=True)
    # for col in cols:
    #     if col not in scaled_df.columns:
    #         scaled_df[col] = pd.NA
    # scaled_df = scaler.inverse_transform(scaled_df)
    # return pd.DataFrame(scaled_df, columns=cols, index=df.index)
    scaled_df = scaler.inverse_transform(df)
    return pd.DataFrame(scaled_df, columns=df.columns, index=df.index)



def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
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


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the population data by adding a column of the previous
    population and returning the transformed DataFrame.
    """
    df = df.copy(deep=True)
    df['prev_population'] = df['population'].shift(1)
    return df[1:]


def do_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(deep=True)
    
    ratio_x0 = df.loc[[df.index[0]], :]
    ratio_df = df.divide(df.shift(1))[1:]

    return ratio_df, ratio_x0


def do_diff(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(deep=True)
    
    diff_x0 = df.loc[[df.index[0]], :]
    diff_df = df.diff(1)[1:]

    return diff_df, diff_x0


def ratio_and_diff(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(deep=True)

    ratio_df, ratio_x0 = do_ratio(df)
    ratio_diff_df, diff_x0 = do_diff(ratio_df)
    
    return ratio_diff_df, ratio_x0, diff_x0


def reverse_diff(diff_df: pd.DataFrame, diff_x0: pd.DataFrame) -> pd.DataFrame:
    df = diff_df.copy(deep=True)
    df = pd.concat([diff_x0, df]).cumsum()
    return df


def reverse_ratio(ratio_df: pd.DataFrame, ratio_x0: pd.DataFrame) -> pd.DataFrame:
    df = ratio_df.copy(deep=True)
    df = pd.concat([ratio_x0, df]).cumprod()
    return df


def reverse_diff_ratio(diff_df: pd.DataFrame, diff_x0: pd.DataFrame, ratio_x0: pd.DataFrame) -> pd.DataFrame:
    df = reverse_diff(diff_df, diff_x0)
    df = reverse_ratio(df, ratio_x0)
    return df


if __name__ == '__main__':
    df = read_population_data('data/data.csv')
    print(f"Original DataFrame:\n{df.head()}\nlength: {len(df)}")

    standardized_df = do_standardize(df)
    print(f"Standardized DataFrame:\n{standardized_df.head()}\nlength: {len(standardized_df)}")

    standardized_test_df = do_standardize(df["population"])
    print(f"Functional Test for Standardize:\n{standardized_test_df.head()}\nlength: {len(standardized_test_df)}")

    transformed_df = transform(standardized_df)
    print(f"Added Previous Population Column:\n{transformed_df.head()}\nlength: {len(transformed_df)}")

    ratio_diff_df, ratio_x0, diff_x0 = ratio_and_diff(standardized_df)
    print(f"Ratio and Diff DataFrame:\n{ratio_diff_df.head()}\nlength: {len(ratio_diff_df)}")

    restored_df = reverse_diff_ratio(ratio_diff_df, diff_x0, ratio_x0)
    print(f"Restored DataFrame:\n{restored_df.head()}\nlength: {len(restored_df)}")


    print(inverse_standardize(restored_df.head()))