import statsmodels.api as sm
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.stattools import acovf, acf, pacf, pacf_yw, pacf_ols
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR, ARResults
from statsmodels.tsa.arima_model import ARMA, ARIMA, ARMAResults, ARIMAResults

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string(),'\n\n\n')          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Reject the null hypothesis",'\n')
        print("Data has no unit root and is stationary")
    else:
        print("Fail to reject the null hypothesis",'\n')
        print("Data has a unit root and is non-stationary")
        

def create_ARIMA_summary(df_states,state_postal_code):
    
    '''
    *purpose: returns order for input into SARIMA MODEL or custom function 'build_SARIMA_model_verify_predictions
    
    *inputs:
    df_states: a dataframe of the state Covid data
    state_postal_code: state postal code to get state related death data
    
    '''
    # create dataframe based on state_postal_code
    df_state = df_states[df_states['state']==state_postal_code]    
    
    # sort index, lowest index to oldest date, drop na's in death column
    df_state = df_state.sort_index()
    df_state = df_state.dropna(subset=['death'])
    df_state_new = pd.DataFrame(df_state)

    # create stepwise fit model, see summary
    stepwise_fit = auto_arima(df_state_new['death'],start_p=0,start_q=0,max_p=6,max_q=5, seasonal=True)
    print(stepwise_fit.summary()) ## find correct ARIMA order
    
    arima_order = stepwise_fit.order
    return arima_order

def build_SARIMA_verify_forecast(df_states,state_postal_code,arima_order):

    '''
    *purpose: build test vs. prediction graph, then plot forecast
    
    *inputs:
    df_states: a dataframe of the state Covid data
    state_postal_code: state postal code to get state related death data
    order: three argument thruple determined in create_ARIMA_summary function
    
    '''
    # create dataframe based on state_postal_code
    df_state = df_states[df_states['state']==state_postal_code]    
    
    # sort index, lowest index to oldest date, drop na's in death column
    df_state = df_state.sort_index()
    df_state = df_state.dropna(subset=['death'])
    df_state_new = pd.DataFrame(df_state) 
    
    length = len(df_state_new)-45

    train_data = df_state_new.iloc[:length]
    test_data = df_state_new.iloc[length:]

    model = sm.tsa.statespace.SARIMAX(train_data['death'],trend='ct', order=arima_order)
    res = model.fit(disp=False)
    res.summary()

    start = len(train_data)
    end = len(train_data) + len(test_data) - 1

    predictions_state = res.predict(start,end,typ='endogenous').rename(f'SARIMAX{arima_order} Predictions')

    # ensure predictions are in DataFrame format, label index as date to match df_alaska
    predictions_state = pd.DataFrame(predictions_state)
    predictions_state.index.name = 'date'

    train_data.index.freq = 'D'
    test_data.index.freq = 'D' # -1D is reverse index, ie most recent date is at top of dataframe
    # perform sort_index on dataframe to correct. set frequencies to match for plotting
    # on same visualization

    # graph test vs. prediction data
    pd.DataFrame(test_data['death']).plot(figsize=(16,8),legend=True,title='Test Data vs SARIMA')
    plt.plot(pd.DataFrame(predictions_state))
    plt.show()

    # train model for forecast
    model = sm.tsa.statespace.SARIMAX(df_state['death'],trend='ct', order=arima_order)
    res = model.fit(disp=False)
    res.summary()

    # create forecast
    fcast = res.predict(start=len(df_state_new),end=len(df_state_new)+45, typ='endogenous').rename('SARIMAX FORECAST')

    # graph forecast deaths, breakout of train and test split is present in graph
    ax1 = train_data['death'].plot(figsize=(16,8),legend=True,ylabel='Deaths',title=f'Forecast Deaths, {state_postal_code}');
    test_data['death'].plot()
    fcast.plot(legend=True,figsize=(18,8)); 

    # graph forecast deaths along with predicted deaths compared with actual over test period, test period matches forecast
    ax2 = test_data['death'].plot(figsize=(16,8),legend=True,title=f'Forecast Deaths, {state_postal_code}');
    train_data['death'].plot(figsize=(16,8),legend=True,ylabel='Deaths');
    plt.plot(predictions_state) # 'FORECAST' FROM END OF TRAINING DATA
    fcast.plot(legend=True,figsize=(18,8)); # SARIMA FORECAST
    
def create_ARIMA_summary_whole(df_states):
    
    '''
    *purpose: returns order for input into SARIMA MODEL or custom function 'build_SARIMA_model_verify_predictions
    
    *inputs:
    df_states: a dataframe of the state Covid data
    
    '''
    
    # sort index, lowest index to oldest date, drop na's in death column
    df_states = df_states.sort_index()
    df_states = df_states.dropna(subset=['death'])
    df_state_new = pd.DataFrame(df_states)

    # create stepwise fit model, see summary
    stepwise_fit = auto_arima(df_state_new['death'],start_p=0,start_q=0,max_p=6,max_q=5, seasonal=True)
    print(stepwise_fit.summary()) ## find correct ARIMA order
    
    arima_order = stepwise_fit.order
    return arima_order

def build_SARIMA_verify_forecast_whole(df_states,arima_order):

    '''
    *purpose: build test vs. prediction graph, then plot forecast
    
    *inputs:
    df_states: a dataframe of the state Covid data
    order: three argument thruple determined in create_ARIMA_summary function
    
    '''
    
    # sort index, lowest index to oldest date, drop na's in death column
    df_state = df_states.sort_index()
    df_state = df_state.dropna(subset=['death'])
    df_state_new = pd.DataFrame(df_state) 
    
    length = len(df_state_new)-45

    train_data = df_state_new.iloc[:length]
    test_data = df_state_new.iloc[length:]

    model = sm.tsa.statespace.SARIMAX(train_data['death'],trend='ct', order=arima_order)
    res = model.fit(disp=False)
    res.summary()

    start = len(train_data)
    end = len(train_data) + len(test_data) - 1

    predictions_state = res.predict(start,end,typ='endogenous').rename(f'SARIMAX{arima_order} Predictions')

    # ensure predictions are in DataFrame format, label index as date to match df_alaska
    predictions_state = pd.DataFrame(predictions_state)
    predictions_state.index.name = 'date'

    train_data.index.freq = 'D'
    test_data.index.freq = 'D' # -1D is reverse index, ie most recent date is at top of dataframe
    # perform sort_index on dataframe to correct. set frequencies to match for plotting
    # on same visualization

    # graph test vs. prediction data
    pd.DataFrame(test_data['death']).plot(figsize=(16,8),legend=True,title='Test Data vs SARIMA')
    plt.plot(pd.DataFrame(predictions_state))
    plt.show()

    # train model for forecast
    model = sm.tsa.statespace.SARIMAX(df_state['death'],trend='ct', order=arima_order)
    res = model.fit(disp=False)
    res.summary()

    # create forecast
    fcast = res.predict(start=len(df_state_new),end=len(df_state_new)+45, typ='endogenous').rename('SARIMAX FORECAST')

    # graph forecast deaths, breakout of train and test split is present in graph
    ax1 = train_data['death'].plot(figsize=(16,8),legend=True,ylabel='Deaths',title='Forecast Deaths, US');
    test_data['death'].plot()
    fcast.plot(legend=True,figsize=(18,8)); 

    # graph forecast deaths along with predicted deaths compared with actual over test period, test period matches forecast
    ax2 = test_data['death'].plot(figsize=(16,8),legend=True,title='Forecast Deaths, US');
    train_data['death'].plot(figsize=(16,8),legend=True,ylabel='Deaths');
    plt.plot(predictions_state) # 'FORECAST' FROM END OF TRAINING DATA
    fcast.plot(legend=True,figsize=(18,8)); # SARIMA FORECAST
