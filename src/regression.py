import statsmodels.api as sm
import numpy as np
import pandas as pd
import os
import csv

def regression(data, y, *xs):
    X = data[[*xs]]
    X = sm.add_constant(X)
    Y = data[[y]]

    model = sm.OLS(Y, X).fit()
    
    return model

def calculate_sse(model, y):
    sse = np. sum ((model.fittedvalues - y ) ** 2)
    return sse

def calculate_ssr(model, y):
    ssr = np. sum ((model.fittedvalues - y.mean() ) ** 2)
    return ssr

def write_results(model, y, path):
    sse = calculate_sse(model, y)
    ssr = calculate_ssr(model, y)
    sst = sse + ssr
    messe = sse / model.df_resid
    messr = ssr / model.df_model
    f_value = (messr / messe)

    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)

        df_result = pd.DataFrame(model.summary().tables[1]).reindex(columns=[0, 1, 5, 6, 3, 4])
        for i in range(len(df_result)):
            writer.writerow(df_result.iloc[i])

        writer.writerow([])
        for p_value in model.pvalues:
            writer.writerow([p_value])

        writer.writerow([])
        writer.writerow([model.df_model, ssr, messr, f_value, model.f_pvalue, model.rsquared])
        writer.writerow([model.df_resid, sse, messe])
        writer.writerow([model.df_model + model.df_resid, sst])

if __name__ == '__main__':
    from data_preprocess import read_population_data, do_standardize

    df = read_population_data('data/data.csv')
    std_df = do_standardize(df)

    model = regression(std_df, *['population', 'urban_population_rate', 'gdp_per_capita', 'unemployment_rate'])
    print(model.summary())

    sse = calculate_sse(model, std_df["population"])
    ssr = calculate_ssr(model, std_df["population"])
    print(f'sse: {sse}')
    print(f'ssr: {ssr}')
    print(f'f value: {model.fvalue}')
    print(f'f value calculated: {ssr / (sse / 21)}')

    write_results(model, std_df["population"], 'test.csv')