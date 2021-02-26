import json
from datetime import datetime as dt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats as sm_stats
import statsmodels.tsa.api as tsa
from river.drift import ADWIN



def get_data(country, state, type):
    
    if type == 'cases':
        source = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    elif type == 'deaths':
        source = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    elif type == 'deaths_br' or 'cases_br' or 'vaccines_br' :
        source = 'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv'
    
    cases = pd.read_csv(source)
    if type == 'cases' or type == 'deaths':
        countries = cases.loc[cases['Province/State'].isna()]
        df = pd.DataFrame(countries.T[4:-1])
        df.columns = countries['Country/Region']
        df['total'] = df.T.sum()
        df = df.astype('int32')
        data = df[country].diff().dropna().reset_index()
        data.columns = ['date','count']
        #data['date'] = data['date']
    else:
        state = cases.loc[cases['state'] == state]
        state['date'] = state['date'].apply(lambda x: x[5:7]+'/'+x[8:]+'/'+x[2:4])
        #state['date'] = state['date'].dt.to_datetime()
        if type == 'cases_br':
            data = state[['date','newCases']]
            
        elif type == 'deaths_br':
            data = state[['date','deathsMS']]
        elif type == 'vaccines_br':
            data = state[['date','vaccinated']].dropna()
            data['vaccinated'] = data['vaccinated'].diff().fillna(0)
        data.columns = ['date','count']
    return data.to_dict(orient="records")

def get_countries():
    return pd.read_csv('./data/countries.csv')['Country/Region'].to_dict()

def get_states():
    return pd.read_csv('./data/states.csv')['state'].to_dict()
        
def get_ma(data,range):
    data = pd.DataFrame(data)
    data['type'] = 'day'
    new_data = data.rolling(window=int(range)).mean().fillna(0).astype('int64')
    new_data['date'] = data['date']
    new_data['type'] = 'average'
    data = data.append(new_data)
    return data.to_dict(orient="records")

def get_drift(data,drift):
    data = pd.DataFrame(data)
    data['type'] = 'day'
    new_data = pd.DataFrame(drift)
    new_data['type'] = 'drift'
    data = data.append(new_data)
    return data.to_dict(orient="records")

def decompose(data):
    #print(tsa.seasonal_decompose(ts(data)).seasonal)
    trend = tsa.seasonal_decompose(ts(data)).trend.fillna(0).reset_index(drop=True).reset_index().to_dict(orient="records")
    seasonal = tsa.seasonal_decompose(ts(data)).seasonal.fillna(0).reset_index(drop=True).reset_index().to_dict(orient="records") 
    resid = tsa.seasonal_decompose(ts(data)).resid.fillna(0).reset_index(drop=True).reset_index().to_dict(orient="records")
    return trend, seasonal, resid
    

def get_acf(data):
    return pd.Series(tsa.acf(ts(data))).reset_index().to_dict(orient='records')

def ts(data):
    data = pd.DataFrame(data)
    data['date'] = data['date'].apply(lambda x : dt.strptime(x,'%m/%d/%y'))
    data = data.set_index('date').asfreq('D')
    return data

def get_stats(data):
    stats = sm_stats.descriptivestats.describe(ts(data))
    ix = stats.index
    ix = ix.append(pd.Index(['total']))
    stats = stats.append(pd.Series(ts(data)['count'].sum(), index=['count']), ignore_index=True)
    stats = stats.set_index(ix)
    #stats.add(pd.Series(ts(data)['count'].sum(), index=['total']))
    #stats[-0] = ts(data)['count'].sum()
    print(stats)
    return stats.to_dict()

def lb_test(acf):
    n = len(acf)
    print(tsa.stattools.q_stat(pd.DataFrame(acf)[0], n))
    return tsa.stattools.q_stat(pd.DataFrame(acf)[0], n)
    return True

def adwin(data):
    adwin = ADWIN()
    i=0
    val=0
    print(data)
    drifts = []
    for row in data:
        in_drift, in_warning = adwin.update(row['count'])
        if in_drift:
            print(f"Change detected at index {row['date']}, input value: {row['count']}")
            drifts.append({'date':row['date'],'count':row['count']})
    return drifts
    