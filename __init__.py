import flask
import dill
import numpy as np
import pandas as pd
import joblib
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler
from lifetimes import GammaGammaFitter
from lifetimes import BetaGeoFitter
from lifetimes.plotting import plot_history_alive,plot_probability_alive_matrix
from lifetimes.plotting import plot_frequency_recency_matrix,plot_expected_repeat_purchases,plot_cumulative_transactions
app = flask.Flask(__name__)

with open('./BGFprpur.pkl', 'rb') as f:
    print(f)
    PREDICTOR = dill.load(f)
with open('./ggf.pkl', 'rb') as g:
    print(g)
    PREDICTOR_CLV = dill.load(g)
with open('./kmeans.pkl', 'rb') as g:
    print(g)
    PREDICTOR_kmeans = dill.load(g)

##################################
@app.route("/plots")
def plot():

    
    return flask.render_template('chart.html')


##################################
@app.route('/greet/<name>')
def greet(name):
    '''Say hello to your first parameter'''
    return "Hello, %s!" %name

@app.route('/predict', methods=["GET"])
def predict():
    t = flask.request.args['t']
    frequency = flask.request.args['frequency']
    recency = flask.request.args['recency']
    T = flask.request.args['T']
    MV = flask.request.args['MV']
    t=np.float(t)
    frequency=np.float(frequency)
    recency=np.float(recency)
    T=np.float(T)
    MV=np.float(MV)

    item = pd.DataFrame([[t, frequency, recency, T, MV]], columns=['t', 'frequency', 'recency', 'T', 'MV'])
    score = PREDICTOR.predict(t,frequency,recency,T)
    clv=PREDICTOR_CLV.customer_lifetime_value(PREDICTOR,item['frequency'],item['recency'],item['T'],item['MV'],12,0.01,'D')


    #item = np.array([pclass, sex, age, fare, sibsp])


    results = {'prediction': score,'CLV':clv}
    return flask.jsonify(results)

##################################
#@app.route('/page')
#def show_page():
#    return flask.render_template('dataentrypage.html')

##################################
@app.route('/page', methods=['POST', 'GET'])
def page():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':

       inputs = flask.request.form

       t = inputs['t']
       frequency = inputs['frequency']
       recency = inputs['recency']
       T = inputs['T']
       MV = inputs['MV']

       t=np.float(t)
       frequency=np.float(frequency)
       recency=np.float(recency)
       T=np.float(T)
       MV=np.float(MV)
       sm=StandardScaler()


       item = pd.DataFrame([[t, frequency, recency, T, MV]], columns=['t', 'frequency', 'recency', 'T', 'MV'])
       score = PREDICTOR.predict(t,frequency,recency,T)
       clv=PREDICTOR_CLV.customer_lifetime_value(PREDICTOR,item['frequency'],item['recency'],item['T'],item['MV'],12,0.01,'D')
       #results = {'survival chances': score[0,1], 'death chances': score[0,0]}
       reg=pd.DataFrame()
       reg['frequency'] = np.log1p(item['frequency'])
       reg['recency'] = np.log1p(item['recency'])
       reg['MV'] = np.log1p(item['MV'])
       reg=sm.fit_transform(reg)
       cluster=PREDICTOR_kmeans.predict(reg)
       print("hello",clv[0],"ds",cluster[0])
       n_clv=clv[0]


    else:
        score = 0
        n_clv=0
        cluster=[0]

    return flask.render_template('dataentrypage.html', score=score,clv=n_clv,cluster=cluster[0])

##################################
if __name__ == '__main__':
    app.run(debug=True)
