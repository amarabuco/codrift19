from flask import Flask, render_template, request
from statsmodels.tsa.stattools import q_stat
import functions as fn
app = Flask(__name__)

@app.route('/')
def home():
    countries = fn.get_countries()
    states = fn.get_states()
    return render_template('home.html', countries=countries, states=states)

@app.route('/analysis', methods=["GET"])
def analysis():
    form = request.args
    data = fn.get_data(form['country'],form['state'],form['type'])
    trend, seasonal, resid  = fn.decompose(data)
    acf = fn.get_acf(data)
    #q_stat =  fn.lb_test(acf)
    stats = fn.get_stats(data)
    decomp = {}
    decomp['trend'] = trend
    decomp['seasonal'] = seasonal
    decomp['resid'] = resid
    data = fn.get_ma(data,7)
    
    out = {
        'data' : data,
        'decomp' : decomp, 
        'acf' : acf,
        'stats': stats
    }
    
    return render_template('analysis.html', out=out, form=form)

@app.route('/forecast', methods=["GET"])
def forecast():
    return render_template('forecast.html')

@app.route('/drift', methods=["GET"])
def drift():
    form = request.args
    data = fn.get_data(form['country'],form['state'],form['type'])
    adwin = fn.adwin(data)
    data = fn.get_drift(data,adwin)
    out = {
        'data' : data
    }
    
    return render_template('drift.html', out=out)