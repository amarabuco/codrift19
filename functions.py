import json
import numpy as pd
import pandas as pd

def get_data(country):
    source = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    
    cases = pd.read_csv(source)
    countries = cases.loc[cases['Province/State'].isna()]
    df = pd.DataFrame(countries.T[4:-1])
    df.columns = countries['Country/Region']
    df['total'] = df.T.sum()
    df = df.astype('int32')
    df = df[country].diff().dropna().reset_index()
    df.columns = ["date","new_cases"]
    #df.rename_axis("date", axis="index")
    
    return json.dumps(df.to_dict(orient="records"))