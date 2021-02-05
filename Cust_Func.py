from tensorflow.random import set_seed
import numpy as np

set_seed(42)
np.random.seed(42)

import statsmodels.api as sm
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.stattools import acovf, acf, pacf, pacf_yw, pacf_ols
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import mse,rmse
from pmdarima import auto_arima
import pandas as pd

import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR, ARResults
from statsmodels.tsa.arima_model import ARMA, ARIMA, ARMAResults, ARIMAResults
from statsmodels.tsa.seasonal import seasonal_decompose as sd
from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
from statsmodels.tsa.statespace.sarimax import SARIMAX
from matplotlib.lines import Line2D

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

def adf_test(series, title=''):
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
        

## please give some feedback on this one regarding the .drop method to drop specific columns that I am using to make the df a little more manageable - is that ridiculous or appropriate code?

def sort_and_clean_df(dataframe, target_column, percent_data_threshold): # sort_df()
    """ 
    *purpose: Pass in dataframe and threshold percent as a decimal, returns a dataframe based on that threshold
    
    *inputs:
    dataframe: dataframe
    target_column: target column in string format
    percent_data_missing_threshold: desired threshold expressed in decimal form
        eg. .05 = 5%  this would result in columns missing 5% or more of their data as NaN to be removed from the dataframe
            .10 = 10% this would result in columns missing 10% or more of their data as NaN to be removed from the dataframe
    
    """
    dataframe = dataframe.drop(columns=['pending', 'totalTestResultsSource', 
        'totalTestResults', 'hospitalizedCurrently',
        'hospitalizedCumulative', 'inIcuCurrently', 'inIcuCumulative',
        'onVentilatorCumulative', 'recovered',
        'dataQualityGrade', 'lastUpdateEt', 'dateModified', 'checkTimeEt',
        'hospitalized', 'dateChecked', 'totalTestsViral',
        'positiveTestsViral', 'negativeTestsViral', 'positiveCasesViral',
        'deathProbable', 'totalTestEncountersViral',
        'totalTestsPeopleViral', 'totalTestsAntibody', 'positiveTestsAntibody',
        'negativeTestsAntibody', 'totalTestsPeopleAntibody',
        'positiveTestsPeopleAntibody', 'negativeTestsPeopleAntibody',
        'totalTestsPeopleAntigen', 'positiveTestsPeopleAntigen',
        'totalTestsAntigen', 'positiveTestsAntigen', 'fips', 'positiveIncrease',
        'negativeIncrease', 'total', 'totalTestResultsIncrease', 'posNeg',
        'deathIncrease', 'hospitalizedIncrease', 'hash', 'commercialScore',
        'negativeRegularScore', 'negativeScore', 'positiveScore', 'score',
        'grade'])

    dataframe['onVentilatorCurrently'] = dataframe['onVentilatorCurrently'].fillna(0)
    dataframe['death'] = dataframe['death'].fillna(0)
    dataframe['negative'] = dataframe['negative'].fillna(0)

    # calculate threshold as a percent of dataframe
    threshold_num = len(dataframe)*percent_data_threshold
    dataframe = dataframe.dropna(axis=1,thresh=len(dataframe)-threshold_num)
    dataframe = dataframe.fillna(0)

    return dataframe


## concerned here that there is not enough code in here but it does serve its purpose to work on a state specific df

def state_dataframe(dataframe, state_postal_code):

    ''' 
    Notes: function assumes all state and US data are seasonal on a weekly basis, but can be specified. if data has no seasonality, use return_arima_order()
    
    *inputs:
    dataframe: a dataframe of the state Covid data
        type = dataframe
    state_postal_code: state postal code to specify state
        type = str
   
    *outputs: returns state specific dataframe to work with
    '''
    # create dataframe based on state_postal_code
    dataframe = dataframe[dataframe['state']==state_postal_code]
    dataframe = pd.DataFrame(dataframe)
    
    # sort index, lowest index to oldest date
    dataframe = dataframe.sort_index()
    dataframe.index.freq = 'D'
    
    print(f"You now have a properly indexed dataframe that contains the state you are targeting.")
    return dataframe

## not worried here but you could glance at it and let me know if you see anything odd, code works and returns a stepwise_fit object

def return_arima_order(dataframe, target_column, m_periods=52, seasonal=True):
    ''' 
    Notes: function assumes all state and US data are seasonal on a weekly basis, but can be set to None if the state data does not
    appear to be seasonal. Additionally, auto_ARIMA is calculating based on a trailing 6 month period.
    
    *inputs:
    dataframe: a dataframe of Covid data
        type = dataframe
    target_column: target column in string format
        type = str
    seasonal: if the data appears to be seasonal, set seasonal to True
        type = bool
    m_periods: seasonality frequency (12 for monthly seasonality, 52 for weekly seasonality, etc
        type = int
   
    *outputs: returns arima order and seasonal arima order
    '''
    trailing_6_months = -180

    # create model fit, see summary
    if seasonal is not None:
        stepwise_fit = auto_arima(dataframe.iloc[trailing_6_months:][target_column],start_p=0,start_q=0,max_p=10,
                              max_q=10,seasonal=True,m=m_periods, method='lbfgs',
                              n_jobs=-1,stepwise=True) 
    elif seasonal is None:
        stepwise_fit = auto_arima(dataframe.iloc[trailing_6_months:][target_column],start_p=0,start_q=0,max_p=10,
                              max_q=10,seasonal=False, method='lbfgs',
                              n_jobs=-1,stepwise=True) 

    print("ARIMA order is: ", stepwise_fit.order)
    if seasonal is not None:
        print("Seasonal ARIMA order is: ", stepwise_fit.seasonal_order)
    else: 
        pass
    print("Use ARIMA object stepwise_fit to store ARIMA and seasonal ARIMA orders in variables.")

    return stepwise_fit

##  this builds the test vs. prediction model, but it required external code in my notebook to get a graphed result. 
    
def evaluate_predictions(dataframe, target_column, days, arima_order, m_periods=52, exogenous_column=None, seasonal_arima_order=None):
    '''
    #purpose: creates a SARIMA or SARIMAX model based on datetime dataframe with any target column
    must specify arima_order at least, but seasonal_arima_order is optional
    
    #inputs:
    dataframe: a dataframe of the state Covid data
        type = dataframe
    target_column: column to forecast trend
        type = str
    days: number of days into the future you wish to forecast
        type = int
    arima_order = arima order from stepwise_fit.order
        type = tuple
    m_periods: periods in season
    *exogenous_column: name of exogenous column data
    *seasonal_arima_order: if trend is seasonal, include seasonal arima order from stepwise_fit.seasonal_order
        type = tuple
        
    #outputs: train data, test data, prediction data (to evaluate against test data), and forecast data 
    '''
    length = len(dataframe)-days

    train_data = dataframe.iloc[:length]
    test_data = dataframe.iloc[length:]

    # train data model build
    if seasonal_arima_order is not None:
        if exogenous_column is not None:
            model = SARIMAX(train_data[target_column],exogenous=train_data[exogenous_column],trend='ct', order=arima_order, seasonal_order=seasonal_arima_order,m=m_periods)
        elif exogenous_column is None:
            model = SARIMAX(train_data[target_column],trend='ct', order=arima_order, seasonal_order=seasonal_arima_order,m=m_periods)
        return model
    elif seasonal_arima_order is None:
        if exogenous_column is not None:
            model = SARIMAX(train_data[target_column],exogenous=train_data[exogenous_column],trend='ct', order=arima_order)
        elif exogenous_column is None:
            model = SARIMAX(train_data[target_column],trend='ct', order=arima_order)
        return model
    
    # instantiate fit model for train_data
    results = model.fit()

    # variables for start and end for predictions to evaluate against test data
    start = len(train_data)
    end = len(train_data) + len(test_data) - 1

    if exogenous_column is not None:
        predictions = results.predict(start,end,typ='endogenous').rename(f'SARIMAX{arima_order} Predictions')
    elif exogenous_column is None:
        predictions = results.predict(start,end,typ='exogenous').rename(f'SARIMAX{arima_order} Predictions')
    # ensure predictions are in DataFrame format, label index as date to match
    predictions.index.name = 'date'
    
    return predictions

# not sure I'm completely understanding how this returns an object that I was able to correctly use in my ipynb, but I did get it to work. (this is last function I need looked at! thanks, James)

def build_SARIMAX_forecast(dataframe, target_column, days, arima_order, m_periods=52, exogenous_column=None, seasonal_arima_order=None):
    '''
    #purpose: creates a SARIMA or SARIMAX model based on datetime dataframe with any target column
    must specify arima_order at least, but seasonal_arima_order is optional
    
    #inputs:
    dataframe: a dataframe of the state Covid data
        type = dataframe
    target_column: column to forecast trend
        type = str
    days: number of days into the future you wish to forecast
        type = int
    arima_order = arima order from stepwise_fit.order
        type = tuple
    m_periods: periods in season
    *exogenous_column: name of exogenous column data
    *seasonal_arima_order: if trend is seasonal, include seasonal arima order from stepwise_fit.seasonal_order
        type = tuple
        
    #outputs: train data, test data, prediction data (to evaluate against test data), and forecast data 
    '''
    length = len(dataframe)-days

    # build full dataframe model
    if seasonal_arima_order is not None:
        if exogenous_column is not None:
            model = SARIMAX(dataframe[target_column],exogenous=dataframe[exogenous_column],trend='ct', order=arima_order, seasonal_order=seasonal_arima_order)
        elif exogenous_column is None:
            model = SARIMAX(dataframe[target_column],trend='ct', order=arima_order, seasonal_order=seasonal_arima_order)
        return model
    elif seasonal_arima_order is None:
        if exogenous_column is not None:
            model = SARIMAX(dataframe[target_column],exogenous=dataframe[exogenous_column],trend='ct', order=arima_order)
        elif exogenous_column is None:
            model = SARIMAX(dataframe[target_column],trend='ct', order=arima_order)
        return model
    
    # new results forecast, use this to get predictions
    results_forecast = model.fit()
    
    # create forecast
    forecast = results_forecast.get_predict(start=length,end=len(dataframe), typ='endogenous').rename(f'SARIMAX {arima_order} {days} Days Forecast')

    # quick plots for forecast
    # predictions.plot(legend=True,figsize=(15,7))
    # dataframe[target_column].plot(legend=True,figsize=(15,7))
    # forecast.plot(legend=True,figsize=(15,7));  

    return forecast # returns model that I have to fit

def graph_predictions(dataframe, test_model, model):
    # instantiate fit model for train_data
    results = test_model.fit()
    

    # variables for start and end for predictions to evaluate against test data
    start = len(train_data)
    end = len(train_data) + len(test_data) - 1
    pass

def graph_forecast(dataframe, results_forecast):
    pass

#     # train model for forecast
#     model = sm.tsa.statespace.SARIMAX(df_state_new['death'],exog=df_state_new['holiday'],
#                                       order=arima_order, seasonal_order=seasonal_order,
#                                       enforce_invertibility=False)
#     res = model.fit(disp=False)

#     # create forecast
#     exog_forecast = df_state_new[length:][['holiday']]
#     fcast = res.predict(start=len(df_state_new),end=len(df_state_new)+days-1,exog=exog_forecast).rename(f'SARIMAX{arima_order},{seasonal_order} {days} Days Forecast')

#*************************************************************************************


#     # train model for forecast
#     model = sm.tsa.statespace.SARIMAX(df_state_new['death'],exog=df_state_new['holiday'],
#                                       order=arima_order, seasonal_order=seasonal_order,
#                                       enforce_invertibility=False)
#     res = model.fit(disp=False)

#     # create forecast
#     exog_forecast = df_state_new[length:][['holiday']]
#     fcast = res.predict(start=len(df_state_new),end=len(df_state_new)+days-1,exog=exog_forecast).rename(f'SARIMAX{arima_order},{seasonal_order} {days} Days Forecast')

#     # graph forecast deaths, breakout of train and test split is present in graph - {PLOT}
#     legend_elements = [Line2D([0], [0], color='b', lw=5, label='Actual Deaths'),
#                        Line2D([0], [0], color='#FFA500', lw=5, label='Actual Deaths'),
#                        Line2D([0], [0], color='g', lw=5, label=f'SARIMAX {arima_order} , {seasonal_order} Predictions'),
#                        Line2D([0], [0], color='r', lw=5, label=f'SARIMAX {arima_order} , {seasonal_order} {days} Day Forecast')]

#     fig, ax = plt.subplots(figsize=(20,10));
#     ax.plot(train_data['death'])
#     ax.plot(test_data['death'])
#     ax.plot(predictions_state)
#     ax.plot(fcast)
#     ax.grid(b=True,alpha=.5)
#     plt.title(f'SARIMAX {days} Day Forecast, {state_postal_code}')
#     ax.legend(handles=legend_elements)
#     for x in df_state_new.index:
#         if df_state_new['holiday'].loc[x]==1:    # for days where holiday == 1
#             ax.axvline(x=x, color='red', alpha = 0.4);   
#     plt.xlabel('Date')
#     plt.ylabel('Deaths')
#     plt.show();

#     last_predictions = len(fcast)-5
#     actual_numbers = fcast[last_predictions:]

#     return actual_numbers
    
#**************************************************************************************************************************
    
    

def create_NN_predict(df_states,state_postal_code,days,epochs):
    
    '''
    *purpose: creates a RNN model based on datetime dataframe with column 'death'
              and a state postal code under column 'state'
    
    *inputs:
    df_states: a dataframe of the state Covid data
    state_postal_code: state postal code to get state related death data
    days: number of days out you wish to forecast
    epochs: number of epochs you wish to run
    '''
    
    # create dataframe based on state_postal_code
    df_state = df_states[df_states['state']==state_postal_code]    

    # sort index, lowest index to oldest date, drop na's in death column
    df_state = df_state.sort_index()
    df_state = df_state.dropna(subset=['death'])
    df_state_new = pd.DataFrame(df_state['death'])

    length = len(df_state_new)-days

    # create train/test split based on days forecasting
    train_data = df_state_new.iloc[:length]
    test_data = df_state_new.iloc[length:]

    # create scaler
    scaler = MinMaxScaler()

    # fit on the train data
    scaler.fit(train_data)

    # scale the train and test data
    scaled_train = scaler.transform(train_data)
    scaled_test = scaler.transform(test_data)

    # define time series generator
    days
    n_features = 1
    generator = TimeseriesGenerator(scaled_train, scaled_train, 
                                    length=days, batch_size=1)

    # build LSTM model 
    model = Sequential()
    model.add(LSTM(300, activation='relu',
                   input_shape=(days,n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mse')

    # fit the model
    model.fit_generator(generator,epochs=epochs)

    # get data for loss values
    loss_per_epoch = model.history.history['loss']
    # plt.plot(range(len(loss_per_epoch)),loss_per_epoch);

    # evaluate the batch
    first_eval = scaled_train[-days:]
    first_eval = first_eval.reshape((1, days, n_features))

    scaler_predictions = []

    first_eval = scaled_train[-days:]
    current_batch = first_eval.reshape((1, days, n_features))

    # create test predictions
    for i in range(len(test_data)):
        current_pred = model.predict(current_batch)[0]
        scaler_predictions.append(current_pred) 
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

    true_predictions = scaler.inverse_transform(scaler_predictions)
    test_data['Predictions'] = true_predictions

    legend_elements = [Line2D([0], [0], color='g', lw=4, label='Actual Deaths'),
                       Line2D([0], [0], color='#FFA500', lw=4, label=f'RNN {state_postal_code} Predictions')]

    fig, ax = plt.subplots(figsize=(20,10));
    ax.plot(test_data)
    ax.plot(train_data);
    ax.grid(b=True,alpha=.5)
    plt.title(f'Test Data vs RNN, {state_postal_code}')
    ax.legend(handles=legend_elements)
    plt.xlabel('Date')
    plt.ylabel('Deaths')
    plt.show();    
    
def multivariate_nn_forecast(df_states,days_to_train,days_to_forecast,epochs):
    
    '''
    *purpose: creates a multivariate RNN model and graph forecast
              based on datetime dataframe with column 'death'
    
    *inputs:
    df_states: a dataframe of the US Covid data
    days_to_train: number of past days to train on
    days_to_forecast: number of days out you wish to forecast
    epochs: number of epochs you wish to run
    '''
    
    # remove extra, unnecessary columns 
    df_states = df_states.sort_index()
    df_states = df_states.drop(columns=['dateChecked','lastModified','hash',
                                        'pending','hospitalizedCumulative',
                                        'inIcuCumulative', 'onVentilatorCumulative',
                                        'recovered','total','deathIncrease',
                                        'hospitalized','hospitalizedIncrease',
                                        'negativeIncrease','posNeg','positiveIncrease',
                                        'states','totalTestResults','totalTestResultsIncrease',
                                        'negative'])
    
    # drop rows where at least one element is missing
    df_states = df_states.dropna() 
    
    # move death to first index position
    df_states = df_states[['death','positive', 'hospitalizedCurrently', 'inIcuCurrently',
                           'onVentilatorCurrently']] 

    # drop all but those currently on ventilators and percentage testing positive out of total test pool
    df_states = df_states.drop(columns=['positive','inIcuCurrently','hospitalizedCurrently'])
    
    # where to specificy the columns to use in multivariate NN
    columns = list(df_states)[0:2]
    print(columns) # variables, x axis is time
    
    # extract x axis dates for plotting certain graphs
    X_axis_dates = pd.to_datetime(df_states.index)
    
    # create training df, ensure float data types
    df_training = df_states[columns].astype(float)
    
    # scale the dataset
    standard_scaler = StandardScaler()
    standard_scaler.fit(df_training)
    df_training_scaled = standard_scaler.transform(df_training)
    
    # create lists to append to
    X_train = []
    y_train = []
    
    # take in input arguments from function call
    future_days = 1
    past_days = days_to_train            # number of days to train the model on
    
    for i in range(past_days, len(df_training_scaled) - future_days + 1):
        X_train.append(df_training_scaled[i-past_days:i, 0:df_training.shape[1]])
        y_train.append(df_training_scaled[i+future_days-1:i+future_days,0])
    
    # set X_train and y_train data sets to numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # save shapes of numpy arrays as variables
    shapeX = X_train.shape
    shapey = y_train.shape
    
    def make_model():
        model = Sequential()
        model.add(LSTM(100, activation='relu', return_sequences=True, 
                       input_shape=(shapeX[1],shapeX[2])))
        model.add(LSTM(50, activation='relu', return_sequences=False))
        model.add(Dense(shapey[1]))
        model.compile(optimizer='adam',loss='mse')
        return model
    
    # instantiate model (make_model function is in this .py file) and fit
    model = make_model()
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    
    # create forecast data
    days = days_to_forecast
    forecast_dates = pd.date_range(list(X_axis_dates)[-1],periods=days,freq='D').tolist()
    forecast = model.predict(X_train[-days:])

    # create target future forecast data and inverse transform
    forecast_columns = np.repeat(forecast, df_training_scaled.shape[1],axis=-1)
    y_pred_future = standard_scaler.inverse_transform(forecast_columns)[:,0]

    # append dates back into new dataframe
    forecast_dates_array = []
    for time in forecast_dates:
        forecast_dates_array.append(time.date())

    # create final forecast dataframe
    df_fcast = []
    df_fcast = pd.DataFrame({'date':np.array(forecast_dates_array),'death':y_pred_future})
    df_fcast.index=pd.to_datetime(df_fcast['date'])

    # plot the data and the forecast data
    df_fcast['death'].plot(legend=True, figsize=(15,7));
    (df_states['death']).plot(legend=True);
    
    
def make_model():
    # initialize and build sequential model
    model = Sequential()

    model.add(LSTM(100, activation='relu', return_sequences=True, 
                   input_shape=(shapeX[1],shapeX[2])))
    model.add(LSTM(50, activation='relu', return_sequences=False))
    model.add(Dense(shapey[1]))

    model.compile(optimizer='adam',loss='mse')
    model.summary()
    return model   


# def create_SARIMA_summary_forecast_usa(df_states,days): # cas_usa()
    
#     '''
#     *purpose: creates a SARIMA model based on datetime dataframe with column 'death'
    
#     *inputs:
#     df_states: a dataframe of the state Covid data
#     days: number of days out you wish to forecast
#     '''
    
#     # sort index, lowest index to oldest date, drop na's in death column
#     df_states = df_states.sort_index()
#     df_states = df_states.dropna(subset=['death'])
#     df_state_new = pd.DataFrame(df_states)

#     # create stepwise fit model, see summary
#     stepwise_fit = auto_arima(df_state_new['death'],start_p=0,start_q=0,max_p=10,
#                           max_q=10, seasonal=True, maxiter=1000, method='bfgs',
#                           n_jobs=-1,stepwise=True)
#     # auto_arima automatically differences and returns that differencing for the model in the arima_order = stepwise_fit.order below
#     ## find correct ARIMA order
    
#     arima_order = stepwise_fit.order
    
#     length = len(df_state_new)-days

#     train_data = df_state_new.iloc[:length]
#     test_data = df_state_new.iloc[length:]

#     model = sm.tsa.statespace.SARIMAX(train_data['death'],trend='ct', order=arima_order)
#     res = model.fit(disp=False)

#     start = len(train_data)
#     end = len(train_data) + len(test_data) - 1

#     predictions = res.predict(start,end,typ='endogenous').rename(f'SARIMAX {arima_order} Predictions')

#     # ensure predictions are in DataFrame format, label index as date to match df_alaska
#     predictions = pd.DataFrame(predictions)
#     predictions.index.name = 'date'

#     train_data.index.freq = 'D'
#     test_data.index.freq = 'D' # -1D is reverse index, ie most recent date is at top of dataframe
#     # perform sort_index on dataframe to correct. set frequencies to match for plotting
#     # on same visualization

#     # graph test vs. prediction data - {PLOT}
#     legend_elements = [Line2D([0], [0], color='b', lw=4, label='Actual Deaths'),
#                        Line2D([0], [0], color='#FFA500', lw=4, label=f'SARIMA {arima_order} Predictions')]

#     fig, ax = plt.subplots(figsize=(20,10));
#     ax.plot(test_data['death'])
#     ax.plot(predictions);
#     ax.grid(b=True,alpha=.5)
#     plt.title('Test Data vs SARIMA, US')
#     ax.legend(handles=legend_elements)
#     plt.xlabel('Date')
#     plt.ylabel('Deaths')
#     plt.show();

#     # train model for forecast
#     model = sm.tsa.statespace.SARIMAX(df_state_new['death'],trend='ct', order=arima_order)
#     res = model.fit(disp=False)

#     # create forecast
#     fcast = res.predict(start=len(df_state_new),end=len(df_state_new)+days, typ='endogenous').rename(f'SARIMAX {arima_order} Predictions')

#     # graph forecast deaths, breakout of train and test split is present in graph - {PLOT}
#     legend_elements = [Line2D([0], [0], color='b', lw=5, label='Actual Deaths'),
#                        Line2D([0], [0], color='#FFA500', lw=5, label='Actual Deaths'),
#                        Line2D([0], [0], color='g', lw=5, label=f'SARIMA {arima_order} Predictions'),
#                        Line2D([0], [0], color='r', lw=5, label=f'SARIMA {arima_order} {days} Day Forecast')]

#     fig, ax = plt.subplots(figsize=(20,10));
#     ax.plot(train_data['death'])
#     ax.plot(test_data['death'])
#     ax.plot(predictions)
#     ax.plot(fcast)
#     ax.grid(b=True,alpha=.5)
#     plt.title(f'SARIMA {days} Day Forecast, US')
#     ax.legend(handles=legend_elements)
#     plt.xlabel('Date')
#     plt.ylabel('Deaths')
#     plt.show();
    
#     last_predictions = len(fcast)-5
#     actual_numbers = fcast[last_predictions:]
    
#     return actual_numbers

# def create_SARIMAX_summary_forecast_usa(df_states,days): # create_SAX_usa()
    
#     '''
#     *purpose: creates a SARIMAX model based on datetime dataframe with column 'death'
#               and uses holidays as an exogenous variable
    
#     *inputs:
#     df_states: a dataframe of the state Covid data
#     days: number of days out you wish to forecast    
#     '''
#     # sort index, lowest index to oldest date, drop na's in death column
#     df_states = df_states.sort_index()
#     df_states = df_states.dropna(subset=['death'])
#     df_state_new = pd.DataFrame(df_states)

# #     ets_decomp = sd(df_state_new['death'])
# #     ets_decomp.plot();

#     # create stepwise fit model, see summary
#     stepwise_fit = auto_arima(df_state_new['death'],seasonal=True,m=52,start_p=0,start_q=0,max_p=10,
#                               max_q=10, maxiter=500, method='bfgs')

#     # auto_arima automatically differences and returns that differencing for the model in the arima_order = stepwise_fit.order below
#     ## find correct ARIMA order

#     arima_order = stepwise_fit.order
#     seasonal_order = stepwise_fit.seasonal_order

#     length = len(df_state_new)-days

#     train_data = df_state_new.iloc[:length]
#     test_data = df_state_new.iloc[length:]

#     model = sm.tsa.statespace.SARIMAX(train_data['death'], trend='ct', seasonal_order=seasonal_order, 
#                                       order=arima_order, enforce_invertibility=False)
#     res = model.fit()

#     start = len(train_data)
#     end = len(train_data) + len(test_data) - 1

#     predictions_state = res.predict(start,end,dynamic=False).rename(f'SARIMAX {arima_order} Predictions')

#     # ensure predictions are in DataFrame format, label index as date to match df_alaska
#     predictions_state = pd.DataFrame(predictions_state)
#     predictions_state.index.name = 'date'

#     train_data.index.freq = 'D'
#     test_data.index.freq = 'D' # -1D is reverse index, ie most recent date is at top of dataframe
#     # perform sort_index on dataframe to correct. set frequencies to match for plotting
#     # on same visualization

#     # graph test vs. prediction data - {PLOT}
#     legend_elements = [Line2D([0], [0], color='b', lw=4, label='Actual Deaths'),
#                        Line2D([0], [0], color='#FFA500', lw=4, label=f'SARIMA {arima_order} Predictions')]

#     fig, ax = plt.subplots(figsize=(20,10));
#     ax.plot(test_data['death'])
#     ax.plot(predictions_state);
#     ax.grid(b=True,alpha=.5)
#     plt.title(f'Test Data vs SARIMA, USA')
#     ax.legend(handles=legend_elements)
#     for x in test_data.index:
#         if test_data['holiday'].loc[x]==1:    # for days where holiday == 1
#             ax.axvline(x=x, color='red', alpha = 0.4);   
#     plt.xlabel('Date')
#     plt.ylabel('Deaths')
#     plt.show();

#     error1 = mse(test_data['death'], predictions_state)
#     error2 = rmse(test_data['death'], predictions_state)

#     # print(f'SARIMAX{arima_order}{seasonal_order} MSE Error: {error1}')
#     # print(f'SARIMAX{arima_order}{seasonal_order} RMSE Error: {error2}')

#     # train model for forecast
#     model = sm.tsa.statespace.SARIMAX(df_state_new['death'],exog=df_state_new['holiday'],
#                                       order=arima_order, seasonal_order=seasonal_order,
#                                       enforce_invertibility=False)
#     res = model.fit(disp=False)

#     # create forecast
#     exog_forecast = df_state_new[length:][['holiday']]
#     fcast = res.predict(start=len(df_state_new),end=len(df_state_new)+days-1,exog=exog_forecast).rename(f'SARIMAX{arima_order},{seasonal_order} {days} Days Forecast')

#     # graph forecast deaths, breakout of train and test split is present in graph - {PLOT}
#     legend_elements = [Line2D([0], [0], color='b', lw=5, label='Actual Deaths'),
#                        Line2D([0], [0], color='#FFA500', lw=5, label='Actual Deaths'),
#                        Line2D([0], [0], color='g', lw=5, label=f'SARIMAX {arima_order} , {seasonal_order} Predictions'),
#                        Line2D([0], [0], color='r', lw=5, label=f'SARIMAX {arima_order} , {seasonal_order} {days} Day Forecast')]

#     fig, ax = plt.subplots(figsize=(20,10));
#     ax.plot(train_data['death'])
#     ax.plot(test_data['death'])
#     ax.plot(predictions_state)
#     ax.plot(fcast)
#     ax.grid(b=True,alpha=.5)
#     plt.title(f'SARIMAX {days} Day Forecast, USA')
#     ax.legend(handles=legend_elements)
#     for x in df_state_new.index:
#         if df_state_new['holiday'].loc[x]==1:    # for days where holiday == 1
#             ax.axvline(x=x, color='red', alpha = 0.4);   
#     plt.xlabel('Date')
#     plt.ylabel('Deaths')
#     plt.show();

#     last_predictions = len(fcast)-5
#     actual_numbers = fcast[last_predictions:]

#     return actual_numbers
    
 
 # graph forecast deaths, breakout of train and test split is present in graph - {PLOT}
#     legend_elements = [Line2D([0], [0], color='b', lw=5, label='Actual Deaths'),
#                        Line2D([0], [0], color='#FFA500', lw=5, label='Actual Deaths'),
#                        Line2D([0], [0], color='g', lw=5, label=f'SARIMA {arima_order} Predictions'),
#                        Line2D([0], [0], color='r', lw=5, label=f'SARIMA {arima_order} {days} Day Forecast')]

#     fig, ax = plt.subplots(figsize=(20,10));
#     ax.plot(train_data['death']) # [Line2D([0], [0], color='b', lw=5, label='Actual Deaths')
#     ax.plot(test_data['death'])
#     ax.plot(predictions_state)
#     ax.plot(fcast)
#     ax.grid(b=True,alpha=.5)
#     plt.title(f'SARIMA {days} Day Forecast, {state_postal_code}')
# #    ax.legend(handles=legend_elements)
#     plt.xlabel('Date')
#     plt.ylabel('Deaths')
#     plt.show();
    
#     last_predictions = len(fcast)-5
#     actual_numbers = fcast[last_predictions:]
    
#     print(actual_numbers)
          
#     return fcast, res
    
    