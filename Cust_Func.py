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
from statsmodels.tsa.seasonal import seasonal_decompose as sd
from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
from matplotlib.lines import Line2D

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
        

def create_ARIMA_summary_forecast_state(df_states,state_postal_code,days):
    
    '''
    *purpose: creates a SARIMA model based on datetime dataframe with column 'death'
              and a state postal code under column 'state'
    
    *inputs:
    df_states: a dataframe of the state Covid data
    state_postal_code: state postal code to get state related death data
    days: number of days out you wish to forecast
    '''
    # create dataframe based on state_postal_code
    df_state = df_states[df_states['state']==state_postal_code]    

    # sort index, lowest index to oldest date, drop na's in death column
    df_state = df_state.sort_index()
    df_state = df_state.dropna(subset=['death'])
    df_state_new = pd.DataFrame(df_state)

    # create stepwise fit model, see summary
    stepwise_fit = auto_arima(df_state_new['death'],start_p=0,start_q=0,max_p=10,
                              max_q=10, seasonal=True, maxiter=1000, method='bfgs',
                              n_jobs=-1,stepwise=True) 

    # auto_arima automatically differences and returns that differencing for the model in the arima_order = stepwise_fit.order below
    ## find correct ARIMA order

    arima_order = stepwise_fit.order

    length = len(df_state_new)-days

    train_data = df_state_new.iloc[:length]
    test_data = df_state_new.iloc[length:]

    model = sm.tsa.statespace.SARIMAX(train_data['death'],trend='ct', order=arima_order)
    res = model.fit(disp=False)

    start = len(train_data)
    end = len(train_data) + len(test_data) - 1

    predictions_state = res.predict(start,end,typ='endogenous').rename(f'SARIMA{arima_order} Predictions')

    # ensure predictions are in DataFrame format, label index as date to match df_alaska
    # predictions_state = pd.DataFrame(predictions_state)
    predictions_state.index.name = 'date'

    train_data.index.freq = 'D'
    test_data.index.freq = 'D' # -1D is reverse index, ie most recent date is at top of dataframe
    # perform sort_index on dataframe to correct. set frequencies to match for plotting
    # on same visualization

    # graph test vs. prediction data - {PLOT}
    legend_elements = [Line2D([0], [0], color='b', lw=4, label='Actual Deaths'),
                       Line2D([0], [0], color='r', lw=4, label=f'SARIMA{arima_order} Predictions')]

    fig, ax = plt.subplots(figsize=(20,10));
    ax.plot(test_data['death'])
    ax.plot(predictions_state);
    ax.grid(b=True,alpha=.5)
    plt.title(f'Test Data vs SARIMA, {state_postal_code}')
    ax.legend(handles=legend_elements)
    plt.xlabel('Date')
    plt.ylabel('Deaths')
    plt.show();

    # train model for forecast
    model = sm.tsa.statespace.SARIMAX(df_state['death'],trend='ct', order=arima_order)
    res = model.fit(disp=False)

    # create forecast
    fcast = res.predict(start=len(df_state_new),end=len(df_state_new)+days, typ='endogenous').rename(f'SARIMA{arima_order} {days} Days Forecast')

    # graph forecast deaths, breakout of train and test split is present in graph - {PLOT}
    legend_elements = [Line2D([0], [0], color='b', lw=5, label='Actual Deaths'),
                       Line2D([0], [0], color='r', lw=5, label='Actual Deaths'),
                       Line2D([0], [0], color='y', lw=5, label=f'SARIMA{arima_order} Predictions'),
                       Line2D([0], [0], color='g', lw=5, label=f'SARIMA{arima_order} {days} Day Forecast')]

    fig, ax = plt.subplots(figsize=(20,10));
    ax.plot(train_data['death'])
    ax.plot(test_data['death'])
    ax.plot(predictions_state)
    ax.plot(fcast)
    ax.grid(b=True,alpha=.5)
    plt.title(f'SARIMA {days} Day Forecast, {state_postal_code}')
    ax.legend(handles=legend_elements)
    plt.xlabel('Date')
    plt.ylabel('Deaths')
    plt.show();
    
def create_ARIMA_summary_forecast_USA(df_states,days):
    
    '''
    *purpose: creates a SARIMA model based on datetime dataframe with column 'death'
    
    *inputs:
    df_states: a dataframe of the state Covid data
    days: number of days out you wish to forecast
    '''
    
    # sort index, lowest index to oldest date, drop na's in death column
    df_states = df_states.sort_index()
    df_states = df_states.dropna(subset=['death'])
    df_state_new = pd.DataFrame(df_states)

    # create stepwise fit model, see summary
    stepwise_fit = auto_arima(df_state_new['death'],start_p=0,start_q=0,max_p=10,
                          max_q=10, seasonal=True, maxiter=1000, method='bfgs',
                          n_jobs=-1,stepwise=True)
    # auto_arima automatically differences and returns that differencing for the model in the arima_order = stepwise_fit.order below
    ## find correct ARIMA order
    
    arima_order = stepwise_fit.order
    
    length = len(df_state_new)-days

    train_data = df_state_new.iloc[:length]
    test_data = df_state_new.iloc[length:]

    model = sm.tsa.statespace.SARIMAX(train_data['death'],trend='ct', order=arima_order)
    res = model.fit(disp=False)

    start = len(train_data)
    end = len(train_data) + len(test_data) - 1

    predictions = res.predict(start,end,typ='endogenous').rename(f'SARIMAX{arima_order} Predictions')

    # ensure predictions are in DataFrame format, label index as date to match df_alaska
    predictions = pd.DataFrame(predictions)
    predictions.index.name = 'date'

    train_data.index.freq = 'D'
    test_data.index.freq = 'D' # -1D is reverse index, ie most recent date is at top of dataframe
    # perform sort_index on dataframe to correct. set frequencies to match for plotting
    # on same visualization

    # graph test vs. prediction data - {PLOT}
    legend_elements = [Line2D([0], [0], color='b', lw=4, label='Actual Deaths'),
                       Line2D([0], [0], color='r', lw=4, label=f'SARIMA{arima_order} Predictions')]

    fig, ax = plt.subplots(figsize=(20,10));
    ax.plot(test_data['death'])
    ax.plot(predictions);
    ax.grid(b=True,alpha=.5)
    plt.title('Test Data vs SARIMA, US')
    ax.legend(handles=legend_elements)
    plt.xlabel('Date')
    plt.ylabel('Deaths')
    plt.show();

    # train model for forecast
    model = sm.tsa.statespace.SARIMAX(df_state_new['death'],trend='ct', order=arima_order)
    res = model.fit(disp=False)

    # create forecast
    fcast = res.predict(start=len(df_state_new),end=len(df_state_new)+days, typ='endogenous').rename(f'SARIMAX{arima_order} Predictions')

    # graph forecast deaths, breakout of train and test split is present in graph - {PLOT}
    legend_elements = [Line2D([0], [0], color='b', lw=5, label='Actual Deaths'),
                   Line2D([0], [0], color='r', lw=5, label='Actual Deaths'),
                   Line2D([0], [0], color='y', lw=5, label=f'SARIMA{arima_order} Predictions'),
                   Line2D([0], [0], color='g', lw=5, label=f'SARIMA{arima_order} {days} Day Forecast')]

    fig, ax = plt.subplots(figsize=(20,10));
    ax.plot(train_data['death'])
    ax.plot(test_data['death'])
    ax.plot(predictions)
    ax.plot(fcast)
    ax.grid(b=True,alpha=.5)
    plt.title(f'SARIMA {days} Day Forecast, US')
    ax.legend(handles=legend_elements)
    plt.xlabel('Date')
    plt.ylabel('Deaths')
    plt.show();
    
# def create_VARMAX_summary_forecast_state(df_states,state_postal_code,days):
    
#     '''
#     *purpose: creates a VARMA model based on datetime dataframe with column 'death'
#               and a state postal code under column 'state'
    
#     *inputs:
#     df_states: a dataframe of the state Covid data
#     state_postal_code: state postal code to get state related death data
#     days: number of days out you wish to forecast
#     '''
# create dataframe based on state_postal_code
# df_state = df_states[df_states['state']==state_postal_code]    

# # sort index, lowest index to oldest date, drop na's in death column
# df_state = df_state.sort_index()
# df_state = df_state.dropna(subset=['death','hospitalizedCurrently'])
# df_state = df_state[['death','hospitalizedCurrently']]
# df_state_new = pd.DataFrame(df_state)

# # create stepwise fit model, see summary
# death_fit = auto_arima(df_state_new['death'],maxiter=1000)
# hospitalized_fit = auto_arima(df_state_new['hospitalizedCurrently'],maxiter=1000)
# # auto_arima automatically differences and returns that differencing for the model in the arima_order = stepwise_fit.order below
# ## find correct ARIMA order

# arima_order = str(stepwise_fit.order[:1]),str(stepwise_fit.order[2:]) # grab the numbers of the order tuple and put into list format
# arima_order_mod = int(stepwise_fit.order[1])
# arima_order_mod2 = int(stepwise_fit.order[1]) # to use in while loop to invert the transformation

# # initialize tuple 
# test_tuple = stepwise_fit.order 
# a,b,c = test_tuple

# a = np.int(a)
# b = np.int(b)
# c = np.int(c)

# arima_order = (a,c)

# while arima_order_mod>0:
#     df_state_new = df_state_new.diff()
#     arima_order_mod -= 1

# df_state_new = df_state_new.dropna(subset=['death','hospitalizedCurrently'])

# length = len(df_state_new)-days

# train_data = df_state_new.iloc[:length]
# test_data = df_state_new.iloc[length:]

# model = VARMAX(train_data, order=arima_order, trend='ct')
# res = model.fit(maxiter=1000, disp=False)

# df_forecast = res.forecast(days)

# while arima_order_mod2>1: # roll back the difference to a first order difference
#     df_forecast['death1d'] = 
#     (df['death'].iloc[-days-(arima_order_mod2-)]-df['death']
#      .iloc[-days-(arima_order_mod)]) 
#     + df_forecast['death']
    
#     df_forecast['deathForecast'] = df['death'].iloc[-nobs-1] + df_forecast['death']

#     df_forecast['currentlyHospitalized1d'] = 
#     (df['currentlyHospitalized'].iloc[-nobs-1]-df['currentlyHospitalized']
#      .iloc[-nobs-2]) + df_forecast['currentlyHospitalized'].cumsum()
    
#     df_forecast['currentlyHospitalizedForecast'] = 
#     df['currentlyHospitalized'].iloc[-nobs-1] + df_forecast['currentlyHospitalized'].cumsum()
    
#     arima_order_mod -= 1 # exit while loop once arimd_order_mod2 is reduced to 0

# start = len(train_data)
# end = len(train_data) + len(test_data) - 1

# predictions_state = res.predict(start,end,typ='endogenous').rename(f'VARMAX{arima_order_mod} Predictions')

# # ensure predictions are in DataFrame format, label index as date to match df_alaska
# predictions_state = pd.DataFrame(predictions_state)
# predictions_state.index.name = 'date'

# train_data.index.freq = 'D'
# test_data.index.freq = 'D' # -1D is reverse index, ie most recent date is at top of dataframe
# # perform sort_index on dataframe to correct. set frequencies to match for plotting
# # on same visualization

# # graph test vs. prediction data - {PLOT}
# fig, ax = plt.subplots();

# pd.DataFrame(test_data['death']).plot(figsize=(16,8),legend=True,title='Test Data vs SARIMA',grid=True);
# plt.plot(predictions_state);
# ax.grid();
# plt.show();

# # train model for forecast
# model = VARMAX(df_state_new['death'],trend='ct', order=arima_order_mod)
# res = model.fit(disp=False)

# # create forecast
# fcast = res.predict(start=len(df_state_new),end=len(df_state_new)+days, typ='endogenous').rename(f'VARMAX{arima_order_mod} Predictions')

# # graph forecast deaths, breakout of train and test split is present in graph - {PLOT}
# fig, ax = plt.subplots();

# train_data['death'].plot(figsize=(16,8),legend=True,ylabel='Deaths',title=f'Forecast Deaths, {state_postal_code}');
# test_data['death'].plot(legend=True);
# fcast.plot(legend=True,figsize=(18,8));
# ax.grid();
# plt.show();

# # graph forecast deaths along with predicted deaths compared with actual over test period, test period matches forecast - {PLOT}
# fig, ax = plt.subplots();

# test_data['death'].plot(figsize=(16,8),legend=True,title=f'Forecast Deaths, {state_postal_code}');
# train_data['death'].plot(figsize=(16,8),legend=True,ylabel='Deaths');
# plt.plot(predictions_state); # 'FORECAST' FROM END OF TRAINING DATA
# fcast.plot(legend=True,figsize=(18,8)); # SARIMA FORECAST
# ax.grid();
# plt.show();
    
# def create_VARMAX_summary_forecast_USA(df_states,days):
    
#     '''
#     *purpose: creates a VARMA model based on datetime dataframe with column 'death'
    
#     *inputs:
#     df_states: a dataframe of the state Covid data
#     days: number of days out you wish to forecast
#     '''
    
#     # sort index, lowest index to oldest date, drop na's in death column
#     df_states = df_states.sort_index()
#     df_states = df_states.dropna(subset=['death'])
#     df_state_new = pd.DataFrame(df_states)

#     # create stepwise fit model, see summary
#     stepwise_fit = auto_arima(df_state_new['death'],maxiter=1000)
#     # auto_arima automatically differences and returns that differencing for the model in the arima_order = stepwise_fit.order below
#     ## find correct ARIMA order
    
#     arima_order = stepwise_fit.order
    
#     length = len(df_state_new)-days

#     train_data = df_state_new.iloc[:length]
#     test_data = df_state_new.iloc[length:]

#     model = VARMAX(train_data['death'],trend='ct', order=arima_order)
#     res = model.fit(disp=False)

#     start = len(train_data)
#     end = len(train_data) + len(test_data) - 1

#     predictions = res.predict(start,end,typ='endogenous').rename(f'SARIMAX{arima_order} Predictions')

#     # ensure predictions are in DataFrame format, label index as date to match df_alaska
#     predictions = pd.DataFrame(predictions)
#     predictions.index.name = 'date'

#     train_data.index.freq = 'D'
#     test_data.index.freq = 'D' # -1D is reverse index, ie most recent date is at top of dataframe
#     # perform sort_index on dataframe to correct. set frequencies to match for plotting
#     # on same visualization

#     # graph test vs. prediction data - {PLOT}
#     fig, ax = plt.subplots();
    
#     pd.DataFrame(test_data['death']).plot(figsize=(16,8),legend=True,title='Test Data vs SARIMA',grid=True);
#     plt.plot(predictions);
#     ax.grid();
#     plt.show();

#     # train model for forecast
#     model = VARMAX(df_state_new['death'],trend='ct', order=arima_order)
#     res = model.fit(disp=False)

#     # create forecast
#     fcast = res.predict(start=len(df_state_new),end=len(df_state_new)+days, typ='endogenous').rename(f'SARIMAX{arima_order} Predictions')

#     # graph forecast deaths, breakout of train and test split is present in graph - {PLOT}
#     fig, ax = plt.subplots();
    
#     train_data['death'].plot(figsize=(16,8),legend=True,ylabel='Deaths',title='Forecast Deaths, US');
#     test_data['death'].plot()
#     fcast.plot(legend=True,figsize=(18,8));
#     ax.grid();
#     plt.show();

#     # graph forecast deaths along with predicted deaths compared with actual over test period, test period matches forecast
#     fig, ax = plt.subplots();
    
#     test_data['death'].plot(figsize=(16,8),legend=True,title='Forecast Deaths, US');
#     train_data['death'].plot(figsize=(16,8),legend=True,ylabel='Deaths');
#     plt.plot(predictions) # 'FORECAST' FROM END OF TRAINING DATA
#     fcast.plot(legend=True,figsize=(18,8)); # SARIMA FORECAST
#     ax.grid();
#     plt.show();
    