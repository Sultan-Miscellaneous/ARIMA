import pandas as pd
import plotly as ply
import numpy as np
from plotly.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
import cufflinks as cf
from pyramid.arima import auto_arima
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller

def load_data(file):
    data = pd.read_csv(file,index_col=0)
    data.head()
    data.index = pd.to_datetime(data.index)
    # data.columns = ['Production']
    data.columns = ['open','high','low','close','volume','nan']
    data = data.drop('high', 1)
    data = data.drop('low', 1)
    data = data.drop('close', 1)
    data = data.drop('volume', 1)
    data = data.drop('nan', 1)
    data = data.loc['2007-01-01':]
    return data


def test_stationarity(data):
    print 'running Dickey-Fuller Test:'  
    dftest = adfuller(data, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    print dfoutput

def apply_differencing(data, lag = 1):
    df = data - data.shift(lag)
    df = df.iloc[lag:]
    return df

def main():

    print 'loading data'
    data = load_data('aapl.us.txt')

    test_stationarity(data['open'])

    first_difference = apply_differencing(data)

    test_stationarity(first_difference['open'])

    first_difference.plot()
    plt.show()
    autocorrelation_plot(first_difference)
    plt.show()

    random_array = pd.DataFrame(np.random.rand(1,len(first_difference.index))).transpose()
    autocorrelation_plot(random_array)
    plt.show()

    print 'auto correlation shows near random signal'

    # result = seasonal_decompose(df['first_difference'], model='multiplicative', freq = 1)
    # result.plot()
    # train = df['first_difference'].loc['1984-09-10':'2016-12-01']
    # test = data.loc['2017-01-01':]

    # stepwise_model = auto_arima(data, start_p=1, start_q=1,
    #                            max_p=20, max_q=20, m=1,
    #                            start_P=0, seasonal=False,
    #                            d=1, D=1, trace=True,
    #                            error_action='ignore',  
    #                            suppress_warnings=True, 
    #                            stepwise=True)
    # print(stepwise_model.aic())

    # train = data.loc['1985-01-01':'2016-12-01']
    # test = data.loc['2017-01-01':]

    # stepwise_model.fit(train)
    # future_forecast = stepwise_model.predict(n_periods=17)

    # print(future_forecast)

    # future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=['Prediction'])
    # pd.concat([test,future_forecast],axis=1).iplot()

    

if __name__ == '__main__':
    main()
