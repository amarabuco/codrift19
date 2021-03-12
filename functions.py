import json
import math
import calendar
from datetime import timedelta as delta
from datetime import datetime as dt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats as sm_stats
import statsmodels.tsa.api as tsa
from collections import OrderedDict

from river import evaluate
from river import linear_model
from river import metrics
from river import compose
from river import optim
from river import preprocessing
from river import datasets
from river import stream
from river import time_series
from river import drift

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

def get_pacf(data):
    return pd.Series(tsa.pacf_ols(ts(data))).reset_index().to_dict(orient='records')

def ts(data):
    data = pd.DataFrame(data)
    data['date'] = data['date'].apply(lambda x : dt.strptime(x,'%m/%d/%y'))
    data = data.set_index('date').asfreq('D')
    return data

def tsr(data,freq):
    data = pd.DataFrame(data)
    data['date'] = data['date'].apply(lambda x : dt.strptime(x,'%m/%d/%y'))
    base = data.to_dict(orient='records')
    data = [({'date': d['date']}, d['count']) for d in base]
    return data

def sum(data):
    data = pd.DataFrame(data)
    return "{:,}".format( data['count'].sum() )

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

def get_model():
    extract_features = compose.TransformerUnion(
    get_ordinal_date,
    get_day_distances
    )
    
    model = (
     extract_features |
     time_series.SNARIMAX(
        p=0,
        d=0,
        q=0,
        m=7,
        sp=3,
        sq=0,
        regressor=(
            preprocessing.StandardScaler() |
            linear_model.LinearRegression(
                intercept_init=0,
                intercept_lr=0.3,
                optimizer=optim.SGD(0.01)
                )
            )
        )
    )
    return model

def get_metric(period):
    return metrics.Rolling(metrics.MAE(), period)

def train(data):
    model = get_model()
    metric = get_metric(7)
    predictions = []

    for x, y in data:
        y_pred = model.forecast(horizon=1, xs=[x])
        model = model.learn_one(x,y)
        metric = metric.update(y, y_pred[0])
        predictions.append(int(y_pred[0]))
    
    return model, predictions, metric

def forecast(data, time):
    X = tsr(data,'D')
    df = pd.DataFrame(data)
    model, pred, metr  = train(X)
    
    df['pred'] = pd.Series(pred)
    df['err'] = (df['count'] - df['pred'])/df['count']
    df.fillna(0, inplace=True)
    tdf = df[['date','count']].copy()
    tdf['type'] = 'hist'
    pdf = df[['date','pred']].copy()
    pdf.columns=['date','count']
    pdf['type'] = 'fit'
    cdf = tdf.append(pdf)
    
    horizon = int(time)
    future = [
        {'date': X[-1][0]['date']+delta(days=d)}
        for d in range(1, horizon + 1)
    ]
    future_data = model.forecast(horizon=horizon, xs=future)
    f = [{'date':x['date'], 'forecast':y_pred} for x, y_pred in zip(future,future_data)]
    fdf = pd.DataFrame(f)
    fdf['date'] = fdf['date'].apply(lambda x : x.strftime('%m/%d/%y'))
    fdf.columns=['date','count']
    fdf['type'] = 'future'
    cdf = cdf.append(fdf)
    fdf['diff'] = fdf['count'].pct_change()
    fdf = fdf.fillna(0)
    fdf['sum'] = fdf['count'].cumsum(axis=0)
        
    return df.to_dict(orient='records'), cdf.to_dict(orient='records'), fdf.to_dict(orient='records'), metr

def get_day_distances(x):
    #print(x)
    distances = {
        calendar.day_name[day]: math.exp(-(x['date'].weekday() - day) ** 2)
        for day in range(0, 7)
    }
    #print(distances)
    return distances

def get_ordinal_date(x):
    #print(x)
    ordinal_date = {'ordinal_date': x['date'].toordinal()}
    #print(ordinal_date)
    return ordinal_date


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
    