from flask import Flask, render_template, request
from functions import get_data
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analysis', methods=["GET"])
def analysis():
    form = request.args
    data = get_data(form['country'])
    
    return render_template('analysis.html', data=data, form=form)

@app.route('/forecast', methods=["GET"])
def forecast():
    return render_template('forecast.html')

@app.route('/drift', methods=["GET"])
def drift():
    return render_template('drift.html')