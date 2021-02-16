# COVID-19 Data Analysis and Forecasting

#### Author: Joseph Denney
#### Email: joseph.d.denney@gmail.com
#### github: www.github.com/josephdenney/Covid_Analysis

## Introduction

### Problem and Purpose

#### This project will use forecasting to model Covid-19 deaths  based on current hospitalization, ventilator, and death data. I will be using API html links to bring in up to date data regularly. This project will use supervised learning in the form of SARIMA and SARIMAX in order to create time series death forecasts.

#### The purpose of this analysis is to provide an accurate forecast of Covid-19 related deaths as 2021 progresses.
#### Our challenges are -
#### * 1. Create multiple forecasts by creating forecasts for specific states
#### * 2. Build a forecast for the United States as a whole
#### * 3. Provide insights as to the urgency of making changes to how we are operating as a country

### The Data
#### The Covid Tracking Project was organized by the news agency The Atlantic early in 2020 in an effort to provide as much data on the pandemic as possible. Coordination of state by state Covid data required building working relationships with state officials to obtain relevant state information. Above are links to the project that can provide further information regarding Covid-19. Additionally, it is worth noting that the project is coming to its end at the beginning of March 2021 as a result of improvements to Federal collection of data. 


```python
import webbrowser

if open_links == True:
    webbrowser.open("https://covidtracking.com/")
    webbrowser.open("https://covidtracking.com/data/api")
```

### Chosen States

### Custom Libraries


```python
# %load_ext autoreload
# %autoreload 2
# %reload_ext autoreload
from Cust_Func import *
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    

### Standard Libraries


```python
from tensorflow.random import set_seed
import numpy as np

set_seed(42)
np.random.seed(42) 
```


```python
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
%matplotlib inline
import functools
from jupyter_plotly_dash import JupyterDash
import datetime as dt
from datetime import date
from datetime import datetime, timedelta
import pandas_datareader as pdr
import holidays
```


```python
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.stattools import acovf, acf, pacf, pacf_yw, pacf_ols
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import mse,rmse
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose as sd
from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
from statsmodels.tsa.ar_model import AR, ARResults
from statsmodels.tsa.arima_model import ARMA, ARIMA, ARMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint, TensorBoard
```


```python
import warnings
warnings.filterwarnings('ignore')

import itertools
import statsmodels.api as sm
from matplotlib.pylab import rcParams
plt.style.use('ggplot')
```


```python
scaler = MinMaxScaler()
standard_scaler = StandardScaler()
```

### Custom Libraries

## Explore Data

### Create New DataFrame


```python
open_links = False
```


```python
# set to true to fetch new data. 
get_data = False
```


```python
if get_data == True:
    df_states = pd.read_csv('https://api.covidtracking.com/v1/states/daily.csv',index_col='date',parse_dates=True)
    df_whole_US = pd.read_csv('https://api.covidtracking.com/v1/us/daily.csv',index_col='date',parse_dates=True)
    df_states.to_csv('StateData.csv')
    df_whole_US.to_csv('USA.csv')
else:
    df_states = pd.read_csv('StateData.csv', index_col='date', parse_dates=True)
    df_whole_US = pd.read_csv('USA.csv', index_col='date', parse_dates=True)
```


```python
df_states.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>positive</th>
      <th>probableCases</th>
      <th>negative</th>
      <th>pending</th>
      <th>totalTestResultsSource</th>
      <th>totalTestResults</th>
      <th>hospitalizedCurrently</th>
      <th>hospitalizedCumulative</th>
      <th>inIcuCurrently</th>
      <th>...</th>
      <th>posNeg</th>
      <th>deathIncrease</th>
      <th>hospitalizedIncrease</th>
      <th>hash</th>
      <th>commercialScore</th>
      <th>negativeRegularScore</th>
      <th>negativeScore</th>
      <th>positiveScore</th>
      <th>score</th>
      <th>grade</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-02-12</th>
      <td>AK</td>
      <td>54282.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>totalTestsViral</td>
      <td>1584548.0</td>
      <td>35.0</td>
      <td>1230.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>54282</td>
      <td>2</td>
      <td>3</td>
      <td>36a7bd363d7e7e136b514bdd9b6e1f20c4ee03e3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>AL</td>
      <td>478667.0</td>
      <td>103040.0</td>
      <td>1842516.0</td>
      <td>NaN</td>
      <td>totalTestsPeopleViral</td>
      <td>2218143.0</td>
      <td>1267.0</td>
      <td>44148.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>2321183</td>
      <td>159</td>
      <td>242</td>
      <td>ff04dbca52ac8a4d86e024cc0103bcd7f40bafb1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>AR</td>
      <td>311608.0</td>
      <td>64580.0</td>
      <td>2322916.0</td>
      <td>NaN</td>
      <td>totalTestsViral</td>
      <td>2569944.0</td>
      <td>712.0</td>
      <td>14278.0</td>
      <td>258.0</td>
      <td>...</td>
      <td>2634524</td>
      <td>13</td>
      <td>23</td>
      <td>58c4a514c40fb89a1ec4ace6165c7865cca79390</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>AS</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2140.0</td>
      <td>NaN</td>
      <td>totalTestsViral</td>
      <td>2140.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>2140</td>
      <td>0</td>
      <td>0</td>
      <td>062c1f214e9d596a07bd505ce605e8819a958e0c</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>AZ</td>
      <td>793532.0</td>
      <td>53218.0</td>
      <td>2864855.0</td>
      <td>NaN</td>
      <td>totalTestsViral</td>
      <td>7140917.0</td>
      <td>2396.0</td>
      <td>55413.0</td>
      <td>705.0</td>
      <td>...</td>
      <td>3658387</td>
      <td>172</td>
      <td>141</td>
      <td>5c92b090358c959185311493808db36779928457</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 54 columns</p>
</div>




```python
df_states.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 19485 entries, 2021-02-12 to 2020-01-13
    Data columns (total 54 columns):
     #   Column                       Non-Null Count  Dtype  
    ---  ------                       --------------  -----  
     0   state                        19485 non-null  object 
     1   positive                     19296 non-null  float64
     2   probableCases                8390 non-null   float64
     3   negative                     15509 non-null  float64
     4   pending                      2024 non-null   float64
     5   totalTestResultsSource       19485 non-null  object 
     6   totalTestResults             19383 non-null  float64
     7   hospitalizedCurrently        16119 non-null  float64
     8   hospitalizedCumulative       12017 non-null  float64
     9   inIcuCurrently               10568 non-null  float64
     10  inIcuCumulative              3516 non-null   float64
     11  onVentilatorCurrently        8386 non-null   float64
     12  onVentilatorCumulative       1201 non-null   float64
     13  recovered                    13992 non-null  float64
     14  dataQualityGrade             18148 non-null  object 
     15  lastUpdateEt                 18899 non-null  object 
     16  dateModified                 18899 non-null  object 
     17  checkTimeEt                  18899 non-null  object 
     18  death                        18614 non-null  float64
     19  hospitalized                 12017 non-null  float64
     20  dateChecked                  18899 non-null  object 
     21  totalTestsViral              12911 non-null  float64
     22  positiveTestsViral           7502 non-null   float64
     23  negativeTestsViral           4498 non-null   float64
     24  positiveCasesViral           13334 non-null  float64
     25  deathConfirmed               9077 non-null   float64
     26  deathProbable                6952 non-null   float64
     27  totalTestEncountersViral     4909 non-null   float64
     28  totalTestsPeopleViral        8607 non-null   float64
     29  totalTestsAntibody           4417 non-null   float64
     30  positiveTestsAntibody        3131 non-null   float64
     31  negativeTestsAntibody        1403 non-null   float64
     32  totalTestsPeopleAntibody     1734 non-null   float64
     33  positiveTestsPeopleAntibody  1002 non-null   float64
     34  negativeTestsPeopleAntibody  903 non-null    float64
     35  totalTestsPeopleAntigen      884 non-null    float64
     36  positiveTestsPeopleAntigen   564 non-null    float64
     37  totalTestsAntigen            2926 non-null   float64
     38  positiveTestsAntigen         1922 non-null   float64
     39  fips                         19485 non-null  int64  
     40  positiveIncrease             19485 non-null  int64  
     41  negativeIncrease             19485 non-null  int64  
     42  total                        19485 non-null  int64  
     43  totalTestResultsIncrease     19485 non-null  int64  
     44  posNeg                       19485 non-null  int64  
     45  deathIncrease                19485 non-null  int64  
     46  hospitalizedIncrease         19485 non-null  int64  
     47  hash                         19485 non-null  object 
     48  commercialScore              19485 non-null  int64  
     49  negativeRegularScore         19485 non-null  int64  
     50  negativeScore                19485 non-null  int64  
     51  positiveScore                19485 non-null  int64  
     52  score                        19485 non-null  int64  
     53  grade                        0 non-null      float64
    dtypes: float64(33), int64(13), object(8)
    memory usage: 8.2+ MB
    


```python
df_whole_US.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>states</th>
      <th>positive</th>
      <th>negative</th>
      <th>pending</th>
      <th>hospitalizedCurrently</th>
      <th>hospitalizedCumulative</th>
      <th>inIcuCurrently</th>
      <th>inIcuCumulative</th>
      <th>onVentilatorCurrently</th>
      <th>onVentilatorCumulative</th>
      <th>...</th>
      <th>lastModified</th>
      <th>recovered</th>
      <th>total</th>
      <th>posNeg</th>
      <th>deathIncrease</th>
      <th>hospitalizedIncrease</th>
      <th>negativeIncrease</th>
      <th>positiveIncrease</th>
      <th>totalTestResultsIncrease</th>
      <th>hash</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-02-12</th>
      <td>56</td>
      <td>27266230.0</td>
      <td>122400369.0</td>
      <td>9434.0</td>
      <td>71504.0</td>
      <td>839119.0</td>
      <td>14775.0</td>
      <td>43389.0</td>
      <td>4849.0</td>
      <td>4126.0</td>
      <td>...</td>
      <td>2021-02-12T24:00:00Z</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>5418</td>
      <td>2345</td>
      <td>567071</td>
      <td>100570</td>
      <td>1816007</td>
      <td>c402515c19b77ba9243af172a9c5799f13cd8e56</td>
    </tr>
    <tr>
      <th>2021-02-11</th>
      <td>56</td>
      <td>27165660.0</td>
      <td>121833298.0</td>
      <td>11981.0</td>
      <td>74225.0</td>
      <td>836774.0</td>
      <td>15190.0</td>
      <td>43291.0</td>
      <td>4970.0</td>
      <td>4113.0</td>
      <td>...</td>
      <td>2021-02-11T24:00:00Z</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>3873</td>
      <td>2460</td>
      <td>588596</td>
      <td>102417</td>
      <td>1872586</td>
      <td>9e06c1c2bc7906114b2dfb77c02fac6a1ff15c7c</td>
    </tr>
    <tr>
      <th>2021-02-10</th>
      <td>56</td>
      <td>27063243.0</td>
      <td>121244702.0</td>
      <td>12079.0</td>
      <td>76979.0</td>
      <td>834314.0</td>
      <td>15788.0</td>
      <td>43184.0</td>
      <td>5121.0</td>
      <td>4106.0</td>
      <td>...</td>
      <td>2021-02-10T24:00:00Z</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>3445</td>
      <td>3226</td>
      <td>385138</td>
      <td>95194</td>
      <td>1393156</td>
      <td>a821a2f23aaee791d155df7e3a2755b31c1bdd32</td>
    </tr>
    <tr>
      <th>2021-02-09</th>
      <td>56</td>
      <td>26968049.0</td>
      <td>120859564.0</td>
      <td>10516.0</td>
      <td>79179.0</td>
      <td>831088.0</td>
      <td>16129.0</td>
      <td>43000.0</td>
      <td>5216.0</td>
      <td>4092.0</td>
      <td>...</td>
      <td>2021-02-09T24:00:00Z</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2795</td>
      <td>3144</td>
      <td>492086</td>
      <td>92986</td>
      <td>1502502</td>
      <td>0ad7fb536eb23f95461201090c436ec7f76ac052</td>
    </tr>
    <tr>
      <th>2021-02-08</th>
      <td>56</td>
      <td>26875063.0</td>
      <td>120367478.0</td>
      <td>12114.0</td>
      <td>80055.0</td>
      <td>827944.0</td>
      <td>16174.0</td>
      <td>42833.0</td>
      <td>5260.0</td>
      <td>4080.0</td>
      <td>...</td>
      <td>2021-02-08T24:00:00Z</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>1309</td>
      <td>1638</td>
      <td>454325</td>
      <td>77737</td>
      <td>1434298</td>
      <td>7abf3026a5235e6761608e2971df85adb1c9bb18</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



### Plot of Ventilators and Deaths for Each State


```python
df_states.columns.unique
```




    <bound method Index.unique of Index(['state', 'positive', 'probableCases', 'negative', 'pending',
           'totalTestResultsSource', 'totalTestResults', 'hospitalizedCurrently',
           'hospitalizedCumulative', 'inIcuCurrently', 'inIcuCumulative',
           'onVentilatorCurrently', 'onVentilatorCumulative', 'recovered',
           'dataQualityGrade', 'lastUpdateEt', 'dateModified', 'checkTimeEt',
           'death', 'hospitalized', 'dateChecked', 'totalTestsViral',
           'positiveTestsViral', 'negativeTestsViral', 'positiveCasesViral',
           'deathConfirmed', 'deathProbable', 'totalTestEncountersViral',
           'totalTestsPeopleViral', 'totalTestsAntibody', 'positiveTestsAntibody',
           'negativeTestsAntibody', 'totalTestsPeopleAntibody',
           'positiveTestsPeopleAntibody', 'negativeTestsPeopleAntibody',
           'totalTestsPeopleAntigen', 'positiveTestsPeopleAntigen',
           'totalTestsAntigen', 'positiveTestsAntigen', 'fips', 'positiveIncrease',
           'negativeIncrease', 'total', 'totalTestResultsIncrease', 'posNeg',
           'deathIncrease', 'hospitalizedIncrease', 'hash', 'commercialScore',
           'negativeRegularScore', 'negativeScore', 'positiveScore', 'score',
           'grade'],
          dtype='object')>




```python
df_states.isnull().sum()
```




    state                              0
    positive                         189
    probableCases                  11095
    negative                        3976
    pending                        17461
    totalTestResultsSource             0
    totalTestResults                 102
    hospitalizedCurrently           3366
    hospitalizedCumulative          7468
    inIcuCurrently                  8917
    inIcuCumulative                15969
    onVentilatorCurrently          11099
    onVentilatorCumulative         18284
    recovered                       5493
    dataQualityGrade                1337
    lastUpdateEt                     586
    dateModified                     586
    checkTimeEt                      586
    death                            871
    hospitalized                    7468
    dateChecked                      586
    totalTestsViral                 6574
    positiveTestsViral             11983
    negativeTestsViral             14987
    positiveCasesViral              6151
    deathConfirmed                 10408
    deathProbable                  12533
    totalTestEncountersViral       14576
    totalTestsPeopleViral          10878
    totalTestsAntibody             15068
    positiveTestsAntibody          16354
    negativeTestsAntibody          18082
    totalTestsPeopleAntibody       17751
    positiveTestsPeopleAntibody    18483
    negativeTestsPeopleAntibody    18582
    totalTestsPeopleAntigen        18601
    positiveTestsPeopleAntigen     18921
    totalTestsAntigen              16559
    positiveTestsAntigen           17563
    fips                               0
    positiveIncrease                   0
    negativeIncrease                   0
    total                              0
    totalTestResultsIncrease           0
    posNeg                             0
    deathIncrease                      0
    hospitalizedIncrease               0
    hash                               0
    commercialScore                    0
    negativeRegularScore               0
    negativeScore                      0
    positiveScore                      0
    score                              0
    grade                          19485
    dtype: int64




```python
# columns to keep in dataframe
columns = ['state','death','inIcuCurrently','onVentilatorCurrently','positive','hospitalizedCurrently','deathIncrease']
```


```python
for col in columns:
    df_states[col] = df_states[col].fillna(0)
```


```python
df_states = sort_and_clean_df(dataframe=df_states, target_columns=columns, percent_data_threshold=.05)
```


```python
df_states.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 19485 entries, 2020-01-13 to 2021-02-12
    Data columns (total 7 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   state                  19485 non-null  object 
     1   death                  19485 non-null  float64
     2   inIcuCurrently         19485 non-null  float64
     3   onVentilatorCurrently  19485 non-null  float64
     4   positive               19485 non-null  float64
     5   hospitalizedCurrently  19485 non-null  float64
     6   deathIncrease          19485 non-null  int64  
    dtypes: float64(5), int64(1), object(1)
    memory usage: 1.2+ MB
    


```python
df_states.iloc[-50:].sort_values(by='death',ascending=False)
# # only graph the top 7 
# that keep state ventilator data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>death</th>
      <th>inIcuCurrently</th>
      <th>onVentilatorCurrently</th>
      <th>positive</th>
      <th>hospitalizedCurrently</th>
      <th>deathIncrease</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-02-12</th>
      <td>CA</td>
      <td>46002.0</td>
      <td>2930.0</td>
      <td>0.0</td>
      <td>3381615.0</td>
      <td>10505.0</td>
      <td>546</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>TX</td>
      <td>40095.0</td>
      <td>2582.0</td>
      <td>0.0</td>
      <td>2541845.0</td>
      <td>8607.0</td>
      <td>324</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>NY</td>
      <td>36882.0</td>
      <td>1358.0</td>
      <td>941.0</td>
      <td>1512690.0</td>
      <td>7068.0</td>
      <td>139</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>FL</td>
      <td>29061.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1781450.0</td>
      <td>4825.0</td>
      <td>190</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>PA</td>
      <td>22959.0</td>
      <td>496.0</td>
      <td>286.0</td>
      <td>888256.0</td>
      <td>2548.0</td>
      <td>99</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>NJ</td>
      <td>22393.0</td>
      <td>525.0</td>
      <td>336.0</td>
      <td>740062.0</td>
      <td>2565.0</td>
      <td>64</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>MI</td>
      <td>16027.0</td>
      <td>293.0</td>
      <td>123.0</td>
      <td>628012.0</td>
      <td>1024.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>GA</td>
      <td>15708.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>958985.0</td>
      <td>3362.0</td>
      <td>195</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>MA</td>
      <td>15358.0</td>
      <td>300.0</td>
      <td>180.0</td>
      <td>553812.0</td>
      <td>1223.0</td>
      <td>89</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>OH</td>
      <td>15136.0</td>
      <td>482.0</td>
      <td>307.0</td>
      <td>934742.0</td>
      <td>1799.0</td>
      <td>2559</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>AZ</td>
      <td>14834.0</td>
      <td>705.0</td>
      <td>466.0</td>
      <td>793532.0</td>
      <td>2396.0</td>
      <td>172</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>TN</td>
      <td>10893.0</td>
      <td>338.0</td>
      <td>184.0</td>
      <td>754279.0</td>
      <td>1332.0</td>
      <td>81</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>NC</td>
      <td>10376.0</td>
      <td>506.0</td>
      <td>0.0</td>
      <td>814594.0</td>
      <td>2151.0</td>
      <td>82</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>LA</td>
      <td>9276.0</td>
      <td>0.0</td>
      <td>151.0</td>
      <td>418585.0</td>
      <td>1001.0</td>
      <td>37</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>AL</td>
      <td>9180.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>478667.0</td>
      <td>1267.0</td>
      <td>159</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>SC</td>
      <td>7894.0</td>
      <td>310.0</td>
      <td>174.0</td>
      <td>480157.0</td>
      <td>1375.0</td>
      <td>57</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>MD</td>
      <td>7503.0</td>
      <td>326.0</td>
      <td>0.0</td>
      <td>368977.0</td>
      <td>1225.0</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>MO</td>
      <td>7442.0</td>
      <td>328.0</td>
      <td>208.0</td>
      <td>470107.0</td>
      <td>1466.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>CT</td>
      <td>7381.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>267337.0</td>
      <td>674.0</td>
      <td>27</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>VA</td>
      <td>6966.0</td>
      <td>430.0</td>
      <td>268.0</td>
      <td>544209.0</td>
      <td>2117.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>WI</td>
      <td>6734.0</td>
      <td>120.0</td>
      <td>0.0</td>
      <td>605785.0</td>
      <td>461.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>MS</td>
      <td>6429.0</td>
      <td>168.0</td>
      <td>112.0</td>
      <td>285648.0</td>
      <td>659.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>CO</td>
      <td>5790.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>411774.0</td>
      <td>470.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>IA</td>
      <td>5223.0</td>
      <td>59.0</td>
      <td>33.0</td>
      <td>273936.0</td>
      <td>249.0</td>
      <td>27</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>AR</td>
      <td>5212.0</td>
      <td>258.0</td>
      <td>123.0</td>
      <td>311608.0</td>
      <td>712.0</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>NV</td>
      <td>4663.0</td>
      <td>219.0</td>
      <td>137.0</td>
      <td>287023.0</td>
      <td>847.0</td>
      <td>26</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>WA</td>
      <td>4633.0</td>
      <td>160.0</td>
      <td>79.0</td>
      <td>326159.0</td>
      <td>704.0</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>OK</td>
      <td>3959.0</td>
      <td>235.0</td>
      <td>0.0</td>
      <td>410818.0</td>
      <td>806.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>NM</td>
      <td>3502.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>179724.0</td>
      <td>365.0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>RI</td>
      <td>2290.0</td>
      <td>39.0</td>
      <td>20.0</td>
      <td>120821.0</td>
      <td>222.0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>WV</td>
      <td>2200.0</td>
      <td>80.0</td>
      <td>39.0</td>
      <td>126887.0</td>
      <td>348.0</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>OR</td>
      <td>2056.0</td>
      <td>54.0</td>
      <td>27.0</td>
      <td>149082.0</td>
      <td>236.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>NE</td>
      <td>2002.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>195485.0</td>
      <td>216.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>PR</td>
      <td>1907.0</td>
      <td>45.0</td>
      <td>45.0</td>
      <td>96924.0</td>
      <td>240.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>SD</td>
      <td>1831.0</td>
      <td>13.0</td>
      <td>11.0</td>
      <td>110068.0</td>
      <td>84.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>UT</td>
      <td>1785.0</td>
      <td>118.0</td>
      <td>0.0</td>
      <td>359641.0</td>
      <td>345.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>ND</td>
      <td>1461.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>98466.0</td>
      <td>39.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>MT</td>
      <td>1324.0</td>
      <td>23.0</td>
      <td>13.0</td>
      <td>97063.0</td>
      <td>100.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>DE</td>
      <td>1269.0</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>82263.0</td>
      <td>247.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>NH</td>
      <td>1117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>69612.0</td>
      <td>138.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>DC</td>
      <td>976.0</td>
      <td>57.0</td>
      <td>32.0</td>
      <td>38670.0</td>
      <td>207.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>WY</td>
      <td>647.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>53086.0</td>
      <td>44.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>ME</td>
      <td>643.0</td>
      <td>24.0</td>
      <td>9.0</td>
      <td>42259.0</td>
      <td>102.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>HI</td>
      <td>425.0</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>27460.0</td>
      <td>40.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>AK</td>
      <td>282.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>54282.0</td>
      <td>35.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>VT</td>
      <td>189.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>13415.0</td>
      <td>49.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>GU</td>
      <td>130.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>7689.0</td>
      <td>8.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>VI</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2524.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>MP</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>134.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>AS</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_states['state'].unique() # list of states to iterate through
```




    array(['WA', 'MA', 'VA', 'FL', 'NJ', 'NE', 'IN', 'MI', 'RI', 'WY', 'NY',
           'PA', 'TX', 'VT', 'WI', 'IL', 'HI', 'NC', 'CO', 'CA', 'AZ', 'GA',
           'NH', 'OR', 'SC', 'MD', 'DC', 'NM', 'TN', 'OH', 'NV', 'IA', 'KY',
           'KS', 'AR', 'DE', 'AK', 'MN', 'WV', 'ID', 'LA', 'CT', 'AL', 'MO',
           'ME', 'MT', 'MS', 'UT', 'SD', 'ND', 'OK', 'GU', 'AS', 'MP', 'VI',
           'PR'], dtype=object)




```python
# for loop iterates through shortened list and prints ventilator usage
# for the trailing 180 day period. 
state_postal = ['NY', 'PA', 'NJ','IL','MI','MA','OH']

fig = plt.figure(figsize=(14,7));

for state in state_postal:
    df_individual = df_states[df_states['state']==state]['onVentilatorCurrently'].sort_index()
    df_plot = df_individual.iloc[(df_individual.index.argmax()-180):(df_individual.index.argmax())]
    plt.plot(df_plot,label=f'{state}');
    plt.title('Number of People on Ventilators')
    plt.xlabel('Date')
    plt.ylabel('People')
    plt.legend();
```


    
![png](Covid_Notebook_files/Covid_Notebook_35_0.png)
    



```python
# same as above graph - all states have death data, this is a graph of the
# states with the highest covid mortality 
state_postal = ['CA', 'NY', 'TX', 'FL', 'PA', 'NJ','IL','MI','MA','OH'] # highest death count states
# some do not have ventilator data reported. 

fig = plt.figure(figsize=(14,7));

for state in state_postal:
    df_individual = df_states[df_states['state']==state].death.sort_index()
    df_plot = df_individual.loc['2020-03':'2020-05']
    plt.plot(df_plot,label=f'{state}');
    plt.title('Number of Total Covid Related Deaths')
    plt.xlabel('Date')
    plt.ylabel('Deaths')
    plt.legend();
```


    
![png](Covid_Notebook_files/Covid_Notebook_36_0.png)
    


### Plot Alaska Death Count


```python
df_AK = df_states[df_states['state']=='AK'] # just look at Alaska for now 
```


```python
fig = plt.figure(figsize=(15,7));

df_AK['death'].plot(legend=True,title='Current Ventilator Usage and Death Totals');
df_AK['onVentilatorCurrently'].plot(legend=True);
df_AK['deathIncrease'].plot(legend=True);
```


    
![png](Covid_Notebook_files/Covid_Notebook_39_0.png)
    


Ventilator usage in Alaska peaks right before the end of December. deathIncrease is the rate of death, or 'volume' of death. Spikes in that line correspond to a steeper increase in deaths along the red trend. 

# Modeling and Forecasts

### Alaska SARIMA Model - Initial Modeling


```python
df_AK = df_AK.sort_index()
```


```python
df_AK = df_AK.dropna(subset=['death'])
df_AK = df_AK.dropna(subset=['onVentilatorCurrently'])
```


```python
df_alaska = pd.DataFrame(df_AK)
```


```python
print(df_alaska.index.min())
print(df_alaska.index.max())
print('Length of dataframe: ' , len(df_alaska))
```

    2020-03-06 00:00:00
    2021-02-12 00:00:00
    Length of dataframe:  344
    


```python
sd(df_alaska['death'], model='additive').plot(); # alaska = seasonal
```


    
![png](Covid_Notebook_files/Covid_Notebook_47_0.png)
    



```python
stepwise_fit = auto_arima(df_alaska['death'],start_p=0,start_q=0,max_p=10,
                          max_q=10, seasonal=True, seasonal_test='ocsb', maxiter=75, method='lbfgs',
                          n_jobs=-1,stepwise=True)
```


```python
model = SARIMAX(df_alaska['death'], order=stepwise_fit.order,seasonal_order=stepwise_fit.seasonal_order).fit()
model.summary()
```




<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>death</td>      <th>  No. Observations:  </th>    <td>344</td>  
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(0, 2, 1)</td> <th>  Log Likelihood     </th> <td>-771.585</td>
</tr>
<tr>
  <th>Date:</th>            <td>Tue, 16 Feb 2021</td> <th>  AIC                </th> <td>1547.170</td>
</tr>
<tr>
  <th>Time:</th>                <td>08:24:46</td>     <th>  BIC                </th> <td>1554.840</td>
</tr>
<tr>
  <th>Sample:</th>             <td>03-06-2020</td>    <th>  HQIC               </th> <td>1550.225</td>
</tr>
<tr>
  <th></th>                   <td>- 02-12-2021</td>   <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ma.L1</th>  <td>   -0.9661</td> <td>    0.010</td> <td>  -94.202</td> <td> 0.000</td> <td>   -0.986</td> <td>   -0.946</td>
</tr>
<tr>
  <th>sigma2</th> <td>    5.2930</td> <td>    0.095</td> <td>   55.531</td> <td> 0.000</td> <td>    5.106</td> <td>    5.480</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>      <td>0.07</td>  <th>  Jarque-Bera (JB):  </th> <td>19731.69</td>
</tr>
<tr>
  <th>Prob(Q):</th>                 <td>0.79</td>  <th>  Prob(JB):          </th>   <td>0.00</td>  
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>102.42</td> <th>  Skew:              </th>   <td>5.22</td>  
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>     <td>0.00</td>  <th>  Kurtosis:          </th>   <td>38.72</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).




```python
# dont forget get_predict
predictions_AK = model.get_forecast(30)
```


```python
predictions_AK.predicted_mean
```




    2021-02-13    283.716071
    2021-02-14    285.432141
    2021-02-15    287.148212
    2021-02-16    288.864283
    2021-02-17    290.580353
    2021-02-18    292.296424
    2021-02-19    294.012495
    2021-02-20    295.728565
    2021-02-21    297.444636
    2021-02-22    299.160706
    2021-02-23    300.876777
    2021-02-24    302.592848
    2021-02-25    304.308918
    2021-02-26    306.024989
    2021-02-27    307.741060
    2021-02-28    309.457130
    2021-03-01    311.173201
    2021-03-02    312.889272
    2021-03-03    314.605342
    2021-03-04    316.321413
    2021-03-05    318.037484
    2021-03-06    319.753554
    2021-03-07    321.469625
    2021-03-08    323.185695
    2021-03-09    324.901766
    2021-03-10    326.617837
    2021-03-11    328.333907
    2021-03-12    330.049978
    2021-03-13    331.766049
    2021-03-14    333.482119
    Freq: D, Name: predicted_mean, dtype: float64




```python
predictions_AK.predicted_mean
alaska_predictions = predictions_AK.conf_int(alpha=.05) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lower death</th>
      <th>upper death</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-02-13</th>
      <td>279.206865</td>
      <td>288.225277</td>
    </tr>
    <tr>
      <th>2021-02-14</th>
      <td>278.946285</td>
      <td>291.917998</td>
    </tr>
    <tr>
      <th>2021-02-15</th>
      <td>279.070661</td>
      <td>295.225763</td>
    </tr>
    <tr>
      <th>2021-02-16</th>
      <td>279.381611</td>
      <td>298.346954</td>
    </tr>
    <tr>
      <th>2021-02-17</th>
      <td>279.803728</td>
      <td>301.356978</td>
    </tr>
    <tr>
      <th>2021-02-18</th>
      <td>280.299039</td>
      <td>304.293809</td>
    </tr>
    <tr>
      <th>2021-02-19</th>
      <td>280.845396</td>
      <td>307.179593</td>
    </tr>
    <tr>
      <th>2021-02-20</th>
      <td>281.428635</td>
      <td>310.028495</td>
    </tr>
    <tr>
      <th>2021-02-21</th>
      <td>282.039103</td>
      <td>312.850169</td>
    </tr>
    <tr>
      <th>2021-02-22</th>
      <td>282.669908</td>
      <td>315.651505</td>
    </tr>
    <tr>
      <th>2021-02-23</th>
      <td>283.315951</td>
      <td>318.437603</td>
    </tr>
    <tr>
      <th>2021-02-24</th>
      <td>283.973353</td>
      <td>321.212343</td>
    </tr>
    <tr>
      <th>2021-02-25</th>
      <td>284.639092</td>
      <td>323.978744</td>
    </tr>
    <tr>
      <th>2021-02-26</th>
      <td>285.310776</td>
      <td>326.739202</td>
    </tr>
    <tr>
      <th>2021-02-27</th>
      <td>285.986476</td>
      <td>329.495643</td>
    </tr>
    <tr>
      <th>2021-02-28</th>
      <td>286.664619</td>
      <td>332.249642</td>
    </tr>
    <tr>
      <th>2021-03-01</th>
      <td>287.343907</td>
      <td>335.002495</td>
    </tr>
    <tr>
      <th>2021-03-02</th>
      <td>288.023260</td>
      <td>337.755284</td>
    </tr>
    <tr>
      <th>2021-03-03</th>
      <td>288.701768</td>
      <td>340.508916</td>
    </tr>
    <tr>
      <th>2021-03-04</th>
      <td>289.378665</td>
      <td>343.264161</td>
    </tr>
    <tr>
      <th>2021-03-05</th>
      <td>290.053296</td>
      <td>346.021671</td>
    </tr>
    <tr>
      <th>2021-03-06</th>
      <td>290.725103</td>
      <td>348.782005</td>
    </tr>
    <tr>
      <th>2021-03-07</th>
      <td>291.393603</td>
      <td>351.545646</td>
    </tr>
    <tr>
      <th>2021-03-08</th>
      <td>292.058383</td>
      <td>354.313008</td>
    </tr>
    <tr>
      <th>2021-03-09</th>
      <td>292.719080</td>
      <td>357.084452</td>
    </tr>
    <tr>
      <th>2021-03-10</th>
      <td>293.375384</td>
      <td>359.860289</td>
    </tr>
    <tr>
      <th>2021-03-11</th>
      <td>294.027022</td>
      <td>362.640793</td>
    </tr>
    <tr>
      <th>2021-03-12</th>
      <td>294.673755</td>
      <td>365.426201</td>
    </tr>
    <tr>
      <th>2021-03-13</th>
      <td>295.315376</td>
      <td>368.216722</td>
    </tr>
    <tr>
      <th>2021-03-14</th>
      <td>295.951701</td>
      <td>371.012537</td>
    </tr>
  </tbody>
</table>
</div>




```python
stepwise_fit.order
```




    (0, 2, 1)




```python
stepwise_fit.summary()
```




<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>y</td>        <th>  No. Observations:  </th>    <td>344</td>  
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(0, 2, 1)</td> <th>  Log Likelihood     </th> <td>-770.318</td>
</tr>
<tr>
  <th>Date:</th>            <td>Tue, 16 Feb 2021</td> <th>  AIC                </th> <td>1546.637</td>
</tr>
<tr>
  <th>Time:</th>                <td>08:24:51</td>     <th>  BIC                </th> <td>1558.141</td>
</tr>
<tr>
  <th>Sample:</th>                  <td>0</td>        <th>  HQIC               </th> <td>1551.220</td>
</tr>
<tr>
  <th></th>                      <td> - 344</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>    0.0067</td> <td>    0.004</td> <td>    1.851</td> <td> 0.064</td> <td>   -0.000</td> <td>    0.014</td>
</tr>
<tr>
  <th>ma.L1</th>     <td>   -0.9870</td> <td>    0.011</td> <td>  -88.631</td> <td> 0.000</td> <td>   -1.009</td> <td>   -0.965</td>
</tr>
<tr>
  <th>sigma2</th>    <td>    5.2394</td> <td>    0.186</td> <td>   28.104</td> <td> 0.000</td> <td>    4.874</td> <td>    5.605</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>0.00</td>  <th>  Jarque-Bera (JB):  </th> <td>20653.57</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>1.00</td>  <th>  Prob(JB):          </th>   <td>0.00</td>  
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>76.47</td> <th>  Skew:              </th>   <td>5.36</td>  
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.00</td>  <th>  Kurtosis:          </th>   <td>39.53</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).




```python
length = len(df_alaska)-45
```


```python
train_data = df_alaska.iloc[:length]
test_data = df_alaska.iloc[length:]
```


```python
model = sm.tsa.statespace.SARIMAX(train_data['death'], order=stepwise_fit.order)
res = model.fit(disp=False)
print(res.summary()) # high p values indicate difficulty in modeling.
```

                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                  death   No. Observations:                  299
    Model:               SARIMAX(0, 2, 1)   Log Likelihood                -594.695
    Date:                Tue, 16 Feb 2021   AIC                           1193.389
    Time:                        08:24:53   BIC                           1200.777
    Sample:                    03-06-2020   HQIC                          1196.347
                             - 12-29-2020                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ma.L1         -0.9494      0.014    -66.756      0.000      -0.977      -0.922
    sigma2         3.1868      0.093     34.116      0.000       3.004       3.370
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.47   Jarque-Bera (JB):             13291.62
    Prob(Q):                              0.49   Prob(JB):                         0.00
    Heteroskedasticity (H):              70.71   Skew:                             4.78
    Prob(H) (two-sided):                  0.00   Kurtosis:                        34.34
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    


```python
start = len(train_data)
end = len(train_data) + len(test_data) - 1
```


```python
predictions_AK = res.predict(start,end,typ='endogenous').rename('SARIMAX(0,2,1) Predictions')
```


```python
train_data.index
```




    DatetimeIndex(['2020-03-06', '2020-03-07', '2020-03-08', '2020-03-09',
                   '2020-03-10', '2020-03-11', '2020-03-12', '2020-03-13',
                   '2020-03-14', '2020-03-15',
                   ...
                   '2020-12-20', '2020-12-21', '2020-12-22', '2020-12-23',
                   '2020-12-24', '2020-12-25', '2020-12-26', '2020-12-27',
                   '2020-12-28', '2020-12-29'],
                  dtype='datetime64[ns]', name='date', length=299, freq=None)




```python
test_data.index
```




    DatetimeIndex(['2020-12-30', '2020-12-31', '2021-01-01', '2021-01-02',
                   '2021-01-03', '2021-01-04', '2021-01-05', '2021-01-06',
                   '2021-01-07', '2021-01-08', '2021-01-09', '2021-01-10',
                   '2021-01-11', '2021-01-12', '2021-01-13', '2021-01-14',
                   '2021-01-15', '2021-01-16', '2021-01-17', '2021-01-18',
                   '2021-01-19', '2021-01-20', '2021-01-21', '2021-01-22',
                   '2021-01-23', '2021-01-24', '2021-01-25', '2021-01-26',
                   '2021-01-27', '2021-01-28', '2021-01-29', '2021-01-30',
                   '2021-01-31', '2021-02-01', '2021-02-02', '2021-02-03',
                   '2021-02-04', '2021-02-05', '2021-02-06', '2021-02-07',
                   '2021-02-08', '2021-02-09', '2021-02-10', '2021-02-11',
                   '2021-02-12'],
                  dtype='datetime64[ns]', name='date', freq=None)




```python
predictions_AK = pd.DataFrame(predictions_AK)
```


```python
predictions_AK.index.name = 'date'
```

#### Compare Test Data with Predictions


```python
train_data.index.freq = 'D'
test_data.index.freq = 'D' # -1D is reverse index, ie most recent date is at top of dataframe
# perform sort_index on dataframe to correct. set frequencies to match for plotting
# on same visualization
```


```python
pd.DataFrame(test_data['death']).info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 45 entries, 2020-12-30 to 2021-02-12
    Freq: D
    Data columns (total 1 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   death   45 non-null     float64
    dtypes: float64(1)
    memory usage: 720.0 bytes
    


```python
predictions_AK.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 45 entries, 2020-12-30 to 2021-02-12
    Freq: D
    Data columns (total 1 columns):
     #   Column                      Non-Null Count  Dtype  
    ---  ------                      --------------  -----  
     0   SARIMAX(0,2,1) Predictions  45 non-null     float64
    dtypes: float64(1)
    memory usage: 720.0 bytes
    


```python
pd.DataFrame(test_data['death']).plot(figsize=(16,8),legend=True,title='Test Data vs SARIMA',grid=True)
plt.plot(pd.DataFrame(predictions_AK))
plt.show()
```


    
![png](Covid_Notebook_files/Covid_Notebook_68_0.png)
    



```python
model = sm.tsa.statespace.SARIMAX(df_alaska['death'], order=stepwise_fit.order)
res = model.fit(disp=False)
print(res.summary())
```

                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                  death   No. Observations:                  344
    Model:               SARIMAX(0, 2, 1)   Log Likelihood                -771.585
    Date:                Tue, 16 Feb 2021   AIC                           1547.170
    Time:                        08:25:04   BIC                           1554.840
    Sample:                    03-06-2020   HQIC                          1550.225
                             - 02-12-2021                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ma.L1         -0.9661      0.010    -94.202      0.000      -0.986      -0.946
    sigma2         5.2930      0.095     55.531      0.000       5.106       5.480
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.07   Jarque-Bera (JB):             19731.69
    Prob(Q):                              0.79   Prob(JB):                         0.00
    Heteroskedasticity (H):             102.42   Skew:                             5.22
    Prob(H) (two-sided):                  0.00   Kurtosis:                        38.72
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    


```python
fcast = res.predict(start=len(df_AK),end=len(df_AK)+45, typ='endogenous').rename('SARIMAX FORECAST')
```


```python
fig, ax = plt.subplots()

train_data['death'].plot(figsize=(16,8),legend=True,ylabel='Deaths',title='Forecast Deaths, Alaska',grid=True);
test_data['death'].plot(grid=True);
fcast.plot(legend=True,figsize=(18,8)); 
ax.grid();
plt.show();
```


    
![png](Covid_Notebook_files/Covid_Notebook_71_0.png)
    


#### Below graph will show prediction data against test data as well as a separate future forecast.


```python
fig, ax = plt.subplots()

test_data['death'].plot(figsize=(16,8),legend=True,title='Forecast Deaths, Alaska',grid=True, label='Test Data');
train_data['death'].plot(figsize=(16,8),legend=True,ylabel='Deaths',grid=True, label='Train Data');
plt.plot(predictions_AK, linewidth=2); # 'PREDICTIONS' FROM END OF TRAINING DATA
fcast.plot(legend=True,figsize=(18,8)); # SARIMA FORECAST
ax.grid();
plt.show();
```


    
![png](Covid_Notebook_files/Covid_Notebook_73_0.png)
    


#### Using Auto Arima to determine order and seasonal order for the SARIMA model is pretty effective here and provides a valid forecast. Given this run through the process, future forecasts will forecast an exogenous variable and plug that forecast back into the model. 

## SARIMAX Modeling

### New York State Ventilator Usage Forecast

#### SARIMAX modeling steps, annotated only for New York forecast
##### * create state specific dataframe
##### * obtain seasonality periods with seasonal decomp
##### * some states provide ventilator data, some provide icu, and others only hospitalization figures. when choosing an exogenous variable to forecast to supplement the death forecast, ventilator data should be prioritized, followed by icu and then hospitalization figures if nothing else is available. 
##### * gridsearch optimization using auto arima. auto arima function is built into custom library and is called by arima_tune() function
##### * evaluate_predictions(), evaluates the predictions against the test data. 
##### * build_SARIMAX_forecast() graphs and returns the first forward looking forecast with the target variable to forecast being the exogenous data we wish to use to enhance the final model. 
##### * get_exogenous_forecast_dataframe() builds and returns a workable dataframe with that forward looking forecast for input into final step
##### * build_SARIMAX_forecast() one more time, specifying the target (death) and the exogenous columns (onVentilatorCurrently)

#### NY State was one of the first states to experience the worst of the pandemic. In addition to this, they are one of the most populous states in the country, and have had a spike in Covid cases post-holiday season 2020. California just passed barely passed New York state in total deaths *related (not directly caused) by COVID-19, making New York the state with the second most deaths in the United States.  


```python
# change to True, run cell to follow link
open_links = False

import webbrowser

if open_links == True:
    webbrowser.open("https://deadline.com/2021/02/california-south-african-covid-19-variant-found-gavin-newsom-1234691514/")
    webbrowser.open("https://www.cityandstateny.com/articles/politics/new-york-state/new-coronavirus-numbers.html")
    webbrowser.open("http://www.op.nysed.gov/COVID-19_EO.html#") # new york state has issued executive orders to increase
    # the number of healtchare workers
```


```python
df = state_dataframe(df_states, 'NY')
```

    Successfully returned indexed dataframe for NY
    


```python
df_ref = state_dataframe(df_states, 'NY')
```

    Successfully returned indexed dataframe for NY
    


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 348 entries, 2020-03-02 to 2021-02-12
    Freq: D
    Data columns (total 7 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   state                  348 non-null    object 
     1   death                  348 non-null    float64
     2   inIcuCurrently         348 non-null    float64
     3   onVentilatorCurrently  348 non-null    float64
     4   positive               348 non-null    float64
     5   hospitalizedCurrently  348 non-null    float64
     6   deathIncrease          348 non-null    int64  
    dtypes: float64(5), int64(1), object(1)
    memory usage: 21.8+ KB
    


```python
# death increase seasonal decomp plot shows that we have near weekly seasonality. 
```


```python
plt.rcParams['figure.figsize']=(15,10);
sd(df.loc['04-2020':'06-2020']['onVentilatorCurrently']).plot();
# m_periods = 7, or 7 days in each cycle since our frequency for the 
# time series index is 'D', or one day. 
```


    
![png](Covid_Notebook_files/Covid_Notebook_84_0.png)
    



```python
# seasonality peaks match the peaks of the deathIncrease seasonal decomp.
```


```python
stepwise_fit, stepwise_full, results, results_full = arima_tune(df, 'onVentilatorCurrently', 
                                                                days_to_forecast=30, m_periods=7, 
                                                                verbose=True) 
# train days arg defaults to 270 days, but can be changed. seasonality can be adjusted as well
# see docstring for further details
# forecasting 30 days out into the future with a seasonality length of 7 days
# verbose = true returns orders and summary as well as plotting diagnostics
```

    ARIMA order is:  (0, 2, 2)
    Seasonal ARIMA order is:  (0, 0, 1, 7)
    Use ARIMA object stepwise_fit to store ARIMA and seasonal ARIMA orders in variables.
    


<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>onVentilatorCurrently</td>      <th>  No. Observations:  </th>    <td>240</td>  
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(0, 2, 2)x(0, 0, [1], 7)</td> <th>  Log Likelihood     </th> <td>-881.528</td>
</tr>
<tr>
  <th>Date:</th>                   <td>Tue, 16 Feb 2021</td>         <th>  AIC                </th> <td>1771.057</td>
</tr>
<tr>
  <th>Time:</th>                       <td>10:24:07</td>             <th>  BIC                </th> <td>1784.774</td>
</tr>
<tr>
  <th>Sample:</th>                    <td>05-19-2020</td>            <th>  HQIC               </th> <td>1776.591</td>
</tr>
<tr>
  <th></th>                          <td>- 01-13-2021</td>           <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>               <td>opg</td>               <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ma.L1</th>   <td>   -0.9036</td> <td>    0.051</td> <td>  -17.733</td> <td> 0.000</td> <td>   -1.003</td> <td>   -0.804</td>
</tr>
<tr>
  <th>ma.L2</th>   <td>    0.0592</td> <td>    0.051</td> <td>    1.167</td> <td> 0.243</td> <td>   -0.040</td> <td>    0.159</td>
</tr>
<tr>
  <th>ma.S.L7</th> <td>    0.3439</td> <td>    0.057</td> <td>    6.011</td> <td> 0.000</td> <td>    0.232</td> <td>    0.456</td>
</tr>
<tr>
  <th>sigma2</th>  <td>  133.0677</td> <td>    7.314</td> <td>   18.195</td> <td> 0.000</td> <td>  118.733</td> <td>  147.402</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>0.01</td> <th>  Jarque-Bera (JB):  </th> <td>204.70</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.92</td> <th>  Prob(JB):          </th>  <td>0.00</td> 
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>1.71</td> <th>  Skew:              </th>  <td>-0.19</td>
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.02</td> <th>  Kurtosis:          </th>  <td>7.63</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).



    
![png](Covid_Notebook_files/Covid_Notebook_86_2.png)
    


#### Reasonable q-q plot. Model summary is good overall. 


```python
evaluate_predictions(results, df, 'onVentilatorCurrently', stepwise_fit=stepwise_fit, 
                     alpha=.05, days_to_forecast=30)
# plot training time and test time
# this evaluates the model using a train test split while also providing
# a forecast of confidence intervals with an alpha of .05. 
```


    
![png](Covid_Notebook_files/Covid_Notebook_88_0.png)
    



```python
exog_forecast, forecast_obj = build_SARIMAX_forecast(model=results_full, 
                                                     dataframe=df, 
                                                     target_column='onVentilatorCurrently', 
                                                     days_to_forecast=30, 
                                                     stepwise_fit=stepwise_full, 
                                                     alpha=.05, state_postal_code='NY')

# this is a forecast of those people who are currently on ventilators in cases
# involving Covid-19. This forecast data will be used to enhance the overall 
# forecast of death or deathIncrease (rate of death)
```


    
![png](Covid_Notebook_files/Covid_Notebook_89_0.png)
    



```python
forecast_obj.conf_int()[-3:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lower onVentilatorCurrently</th>
      <th>upper onVentilatorCurrently</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-03-12</th>
      <td>254.437515</td>
      <td>1222.552583</td>
    </tr>
    <tr>
      <th>2021-03-13</th>
      <td>224.304039</td>
      <td>1238.460202</td>
    </tr>
    <tr>
      <th>2021-03-14</th>
      <td>193.822435</td>
      <td>1254.715949</td>
    </tr>
  </tbody>
</table>
</div>



#### New York ventilator data has shown an improvement more recently. The recent downward trend here is encouraging, and it is likely to continue.
#### The initial spike in ventilator usage in New York (seen below) was likely a result of not knowing how to properly treat the virus. As time has progressed we have learned that ventilators should only be used in the most severe cases. Additional methods like keeping Covid patients on their stomachs while using the ventilator has been claimed to be more effective. 
#### Finally, this forecast will be used to influence a forecast of deaths in New York.


```python
df['onVentilatorCurrently'].plot(figsize=(12,4)); # see spike here coming down 
# from initial May 2020 reporting. 
```


    
![png](Covid_Notebook_files/Covid_Notebook_92_0.png)
    


### New York State Deaths Forecast 
#### Modeled using New York's Ventilator Usage Forecast


```python
# returns df_forecast for input into the final build sarimax forecast below
stepwise_fit, df_forecast = get_exogenous_forecast_dataframe(dataframe=df,
                                                             original_dataframe=df_ref,
                                                             exog_forecast=exog_forecast, 
                                                             target_column='death',
                                                             exogenous_column='onVentilatorCurrently',
                                                             days_to_forecast=30,
                                                             m_periods=7)
```


```python
df_forecast.tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>death</th>
      <th>inIcuCurrently</th>
      <th>onVentilatorCurrently</th>
      <th>positive</th>
      <th>hospitalizedCurrently</th>
      <th>deathIncrease</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-03-10</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>752.720906</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2021-03-11</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>745.607978</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2021-03-12</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>738.495049</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2021-03-13</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>731.382121</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2021-03-14</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>724.269192</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create final model using sarimax
full_exog_model = SARIMAX(df['death'],df['onVentilatorCurrently'],
                          order=stepwise_fit.order,seasonal_order=stepwise_fit.seasonal_order)

# fit model before plugging into below function
model = full_exog_model.fit()
```


```python
exog_forecast, results_forecast = build_SARIMAX_forecast(model=model,
                                                         dataframe=df_forecast, 
                                                         target_column='death', 
                                                         days_to_forecast=30, 
                                                         stepwise_fit=stepwise_fit, 
                                                         alpha=.05,
                                                         original_df=df_ref,
                                                         exogenous_column='onVentilatorCurrently',
                                                         state_postal_code='NY')
```


    
![png](Covid_Notebook_files/Covid_Notebook_97_0.png)
    



```python
# actual numbers of 95% confidence interval. alpha defaults to .05
results_forecast.conf_int()[-3:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lower death</th>
      <th>upper death</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-03-12</th>
      <td>33784.347268</td>
      <td>46876.594343</td>
    </tr>
    <tr>
      <th>2021-03-13</th>
      <td>33462.303924</td>
      <td>47440.180518</td>
    </tr>
    <tr>
      <th>2021-03-14</th>
      <td>33128.260645</td>
      <td>48015.668384</td>
    </tr>
  </tbody>
</table>
</div>



##### Above graph implements usage of a function that flattens the lower portion of the confidence interval instead of allowing it to decrease. Deaths remain flat in a best case scenario instead of 'decreasing' in the forecast. 

### New York Analysis and Recommendations

#### New York flattened its curve from the beginning of the pandemic until the Thanksgiving holidays when everything suddenly became more difficult. There is a clear increase in deaths beginning around mid November and into Christmas and the New Year, likely a result of family gatherings, social gatherings, and increased survivability in the cold. 
#### There is every possibility that without continuing social distancing and enforcing the wearing of masks that the rate of death will once again increase.

##### Recommendations for the state include the following:
##### *Lower the number of people allowed at indoor private gatherings from the current amount of 10. Social gatherings are not economically essential. Certain states have limits of one or two households per private residence which has proven to limit the spread of the disease. 
##### * Increase effort to improve awareness. Covering the mouth but not the nose does not limit the spread of Covid-19. Mandate signage depicting proper mask usage at public establishments. 
##### * Continue social distancing policies and reduce the number of outdoor events. The pandemic in New York is not under control. 

### California Deaths Forecast
#### Modeled using ICU Forecast (ICU Forecast not shown)


```python
# change to True, run cell to follow link
open_links = False

import webbrowser

if open_links == True:
    webbrowser.open("https://deadline.com/2021/02/california-south-african-covid-19-variant-found-gavin-newsom-1234691514/")
    webbrowser.open("https://covid19.ca.gov/?utm_source=google&utm_medium=cpc&utm_campaign=ca-covid19response-august2020&utm_term=covid%2019&gclid=CjwKCAiAjp6BBhAIEiwAkO9WukC-31gfIfspHCyf7FgUt_vAh4OrFSPfX0QXfbyLfQRvVDnlIJPhKxoCZqQQAvD_BwE")
    
```


```python
df_ref = state_dataframe(df_states, 'CA')
```

    Successfully returned indexed dataframe for CA
    


```python
plt.rcParams['figure.figsize']=(15,10);
sd(df_ref.loc['04-2020':'06-2020']['inIcuCurrently']).plot();
```


    
![png](Covid_Notebook_files/Covid_Notebook_106_0.png)
    



```python
stepwise_fit, stepwise_full, results, results_full = arima_tune(df_ref, 'inIcuCurrently', 
                                                                days_to_forecast=30, m_periods=7, 
                                                                verbose=True) 
# train days arg defaults to 270 days, but can be changed. seasonality can be adjusted as well
# see docstring for further details
# forecasting 30 days out into the future with a seasonality length of 7 days
```

    ARIMA order is:  (1, 2, 3)
    Seasonal ARIMA order is:  (2, 0, 1, 7)
    Use ARIMA object stepwise_fit to store ARIMA and seasonal ARIMA orders in variables.
    


<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>inIcuCurrently</td>          <th>  No. Observations:  </th>    <td>240</td>   
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(1, 2, 3)x(2, 0, [1], 7)</td> <th>  Log Likelihood     </th> <td>-1106.950</td>
</tr>
<tr>
  <th>Date:</th>                   <td>Tue, 16 Feb 2021</td>         <th>  AIC                </th> <td>2229.900</td> 
</tr>
<tr>
  <th>Time:</th>                       <td>08:28:19</td>             <th>  BIC                </th> <td>2257.157</td> 
</tr>
<tr>
  <th>Sample:</th>                    <td>05-19-2020</td>            <th>  HQIC               </th> <td>2240.904</td> 
</tr>
<tr>
  <th></th>                          <td>- 01-13-2021</td>           <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>               <td>opg</td>               <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ar.L1</th>    <td>   -0.9631</td> <td>    0.049</td> <td>  -19.747</td> <td> 0.000</td> <td>   -1.059</td> <td>   -0.868</td>
</tr>
<tr>
  <th>ma.L1</th>    <td>   -0.0191</td> <td>    0.089</td> <td>   -0.214</td> <td> 0.830</td> <td>   -0.194</td> <td>    0.156</td>
</tr>
<tr>
  <th>ma.L2</th>    <td>   -0.7361</td> <td>    0.078</td> <td>   -9.422</td> <td> 0.000</td> <td>   -0.889</td> <td>   -0.583</td>
</tr>
<tr>
  <th>ma.L3</th>    <td>    0.1093</td> <td>    0.075</td> <td>    1.465</td> <td> 0.143</td> <td>   -0.037</td> <td>    0.256</td>
</tr>
<tr>
  <th>ar.S.L7</th>  <td>    0.2439</td> <td>    0.325</td> <td>    0.750</td> <td> 0.453</td> <td>   -0.393</td> <td>    0.881</td>
</tr>
<tr>
  <th>ar.S.L14</th> <td>    0.1039</td> <td>    0.071</td> <td>    1.465</td> <td> 0.143</td> <td>   -0.035</td> <td>    0.243</td>
</tr>
<tr>
  <th>ma.S.L7</th>  <td>   -0.2487</td> <td>    0.336</td> <td>   -0.740</td> <td> 0.460</td> <td>   -0.908</td> <td>    0.410</td>
</tr>
<tr>
  <th>sigma2</th>   <td> 1191.5610</td> <td>  102.142</td> <td>   11.666</td> <td> 0.000</td> <td>  991.366</td> <td> 1391.756</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>0.03</td> <th>  Jarque-Bera (JB):  </th> <td>23.81</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.86</td> <th>  Prob(JB):          </th> <td>0.00</td> 
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.87</td> <th>  Skew:              </th> <td>-0.62</td>
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.55</td> <th>  Kurtosis:          </th> <td>4.01</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).



    
![png](Covid_Notebook_files/Covid_Notebook_107_2.png)
    



```python
exog_forecast, forecast_obj = build_SARIMAX_forecast(model=results_full, 
                                                     dataframe=df_ref, 
                                                     target_column= 'inIcuCurrently', 
                                                     days_to_forecast=30, 
                                                     stepwise_fit=stepwise_full, 
                                                     alpha=.05, state_postal_code='CA')

# this is a forecast of those people who are currently on ventilators in cases
# involving Covid-19. This forecast data will be used to enhance the overall 
# forecast of death or deathIncrease (rate of death)
```


    
![png](Covid_Notebook_files/Covid_Notebook_108_0.png)
    



```python
forecast_obj.conf_int()[-3:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lower inIcuCurrently</th>
      <th>upper inIcuCurrently</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-03-12</th>
      <td>-1316.197735</td>
      <td>1812.058878</td>
    </tr>
    <tr>
      <th>2021-03-13</th>
      <td>-1493.942065</td>
      <td>1783.965606</td>
    </tr>
    <tr>
      <th>2021-03-14</th>
      <td>-1657.600905</td>
      <td>1770.970581</td>
    </tr>
  </tbody>
</table>
</div>




```python
state_dataframe, exog_forecast = create_exog_forecast(df_states, 'inIcuCurrently', 
                                                      m_periods=7, state_postal_code='CA')
```

    Successfully returned indexed dataframe for CA
    ARIMA order is:  (1, 2, 3)
    Seasonal ARIMA order is:  (2, 0, 1, 7)
    Use ARIMA object stepwise_fit to store ARIMA and seasonal ARIMA orders in variables.
    


<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>inIcuCurrently</td>          <th>  No. Observations:  </th>    <td>240</td>   
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(1, 2, 3)x(2, 0, [1], 7)</td> <th>  Log Likelihood     </th> <td>-1106.950</td>
</tr>
<tr>
  <th>Date:</th>                   <td>Tue, 16 Feb 2021</td>         <th>  AIC                </th> <td>2229.900</td> 
</tr>
<tr>
  <th>Time:</th>                       <td>08:29:05</td>             <th>  BIC                </th> <td>2257.157</td> 
</tr>
<tr>
  <th>Sample:</th>                    <td>05-19-2020</td>            <th>  HQIC               </th> <td>2240.904</td> 
</tr>
<tr>
  <th></th>                          <td>- 01-13-2021</td>           <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>               <td>opg</td>               <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ar.L1</th>    <td>   -0.9631</td> <td>    0.049</td> <td>  -19.747</td> <td> 0.000</td> <td>   -1.059</td> <td>   -0.868</td>
</tr>
<tr>
  <th>ma.L1</th>    <td>   -0.0191</td> <td>    0.089</td> <td>   -0.214</td> <td> 0.830</td> <td>   -0.194</td> <td>    0.156</td>
</tr>
<tr>
  <th>ma.L2</th>    <td>   -0.7361</td> <td>    0.078</td> <td>   -9.422</td> <td> 0.000</td> <td>   -0.889</td> <td>   -0.583</td>
</tr>
<tr>
  <th>ma.L3</th>    <td>    0.1093</td> <td>    0.075</td> <td>    1.465</td> <td> 0.143</td> <td>   -0.037</td> <td>    0.256</td>
</tr>
<tr>
  <th>ar.S.L7</th>  <td>    0.2439</td> <td>    0.325</td> <td>    0.750</td> <td> 0.453</td> <td>   -0.393</td> <td>    0.881</td>
</tr>
<tr>
  <th>ar.S.L14</th> <td>    0.1039</td> <td>    0.071</td> <td>    1.465</td> <td> 0.143</td> <td>   -0.035</td> <td>    0.243</td>
</tr>
<tr>
  <th>ma.S.L7</th>  <td>   -0.2487</td> <td>    0.336</td> <td>   -0.740</td> <td> 0.460</td> <td>   -0.908</td> <td>    0.410</td>
</tr>
<tr>
  <th>sigma2</th>   <td> 1191.5610</td> <td>  102.142</td> <td>   11.666</td> <td> 0.000</td> <td>  991.366</td> <td> 1391.756</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>0.03</td> <th>  Jarque-Bera (JB):  </th> <td>23.81</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.86</td> <th>  Prob(JB):          </th> <td>0.00</td> 
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.87</td> <th>  Skew:              </th> <td>-0.62</td>
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.55</td> <th>  Kurtosis:          </th> <td>4.01</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).



    
![png](Covid_Notebook_files/Covid_Notebook_110_2.png)
    



```python
# normal q-q plot
```


```python
forecast_object = graph_exog_forecast(dataframe=state_dataframe, 
                                      target_column='death', 
                                      exogenous_column='inIcuCurrently', 
                                      exog_forecast=exog_forecast,
                                      df_ref=df_ref, 
                                      alpha=.05, days_to_forecast=30, 
                                      train_days=270, m_periods=7,
                                      state_postal_code='CA')
```


    
![png](Covid_Notebook_files/Covid_Notebook_112_0.png)
    



```python

```


```python
forecast_object.predicted_mean[-5:] # projected mean deaths
# by March 14th, 2021 stand at 57,730.  
```




    2021-03-10    56063.879182
    2021-03-11    56515.194316
    2021-03-12    56978.468058
    2021-03-13    57444.262203
    2021-03-14    57729.645022
    Freq: D, Name: predicted_mean, dtype: float64




```python
forecast_object.conf_int()[-5:] # upper confidence interval of 95% forecasts
# deaths of over 61,000 by March 14th, 2021. 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lower death</th>
      <th>upper death</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-03-10</th>
      <td>52931.486803</td>
      <td>59196.271562</td>
    </tr>
    <tr>
      <th>2021-03-11</th>
      <td>53183.717245</td>
      <td>59846.671386</td>
    </tr>
    <tr>
      <th>2021-03-12</th>
      <td>53443.382517</td>
      <td>60513.553599</td>
    </tr>
    <tr>
      <th>2021-03-13</th>
      <td>53689.901401</td>
      <td>61198.623006</td>
    </tr>
    <tr>
      <th>2021-03-14</th>
      <td>53749.443886</td>
      <td>61709.846157</td>
    </tr>
  </tbody>
</table>
</div>



### California Analysis and Recommendations

#### California has experienced an even more drastic increase in deaths since the holidays than any other state. What is happening is an emergency with over 540 Covid deaths on Feb. 11th, 2021 alone. The California state website talks about helping to slow the spread of Covid. Given the circumstances, this language is not urgently and clearly conveying the message that the situation needs immediate attention from each and every individual living in the state. 

##### Recommendations for the state include the following:
##### * Require wearing a mask if an individual is not in or on their private property. Allow no exceptions. 
##### * Prohibit private and public gatherings of 5 or more people unless from the same household. 
##### * The spread of this disease in this state will continue to take lives if people are not made to understand the consequences of selfish behavior. Introduce visual evidence of the rammifications of the virus with an emphasis on personal stories on public social media and television. 

### Texas Hospitalized Forecast


```python
# change to True, run cell to follow link(s)
open_links = False

import webbrowser

if open_links == True:
    webbrowser.open("https://www.kvue.com/article/news/health/coronavirus/austin-texas-stage-4-covid-coronavirus-stage-5-feb-9/269-07d19225-d943-496a-8246-8940489fbc65")
    webbrowser.open("https://www.khou.com/article/news/local/texas/texas-covid-hospitalizations-record-houston-harris-county/285-2549ca26-ebf6-40e0-b29e-e4c945e08a50")
    webbrowser.open("https://apps.texastribune.org/features/2020/texas-coronavirus-cases-map/?_ga=2.53372705.880277519.1613246925-1270976238.1613246925")
    
```

#### Texas is also a populous state and has been inclined to open back up quickly to keep the economy going. The state has not reported ICU or Ventilator numbers to the Covid Tracking Project. A KHOU article cites a decrease in hospitalizations while warning about the continued threat of breaching hospital capacity. 


```python
df = state_dataframe(df_states, 'TX')
```

    Successfully returned indexed dataframe for TX
    


```python
df_ref = state_dataframe(df_states, 'TX')
```

    Successfully returned indexed dataframe for TX
    


```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>death</th>
      <th>inIcuCurrently</th>
      <th>onVentilatorCurrently</th>
      <th>positive</th>
      <th>hospitalizedCurrently</th>
      <th>deathIncrease</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-02-08</th>
      <td>TX</td>
      <td>38700.0</td>
      <td>2667.0</td>
      <td>0.0</td>
      <td>2491227.0</td>
      <td>9401.0</td>
      <td>57</td>
    </tr>
    <tr>
      <th>2021-02-09</th>
      <td>TX</td>
      <td>39001.0</td>
      <td>2777.0</td>
      <td>0.0</td>
      <td>2504556.0</td>
      <td>9401.0</td>
      <td>301</td>
    </tr>
    <tr>
      <th>2021-02-10</th>
      <td>TX</td>
      <td>39386.0</td>
      <td>2740.0</td>
      <td>0.0</td>
      <td>2517453.0</td>
      <td>9165.0</td>
      <td>385</td>
    </tr>
    <tr>
      <th>2021-02-11</th>
      <td>TX</td>
      <td>39771.0</td>
      <td>2703.0</td>
      <td>0.0</td>
      <td>2529343.0</td>
      <td>8933.0</td>
      <td>385</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>TX</td>
      <td>40095.0</td>
      <td>2582.0</td>
      <td>0.0</td>
      <td>2541845.0</td>
      <td>8607.0</td>
      <td>324</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.rcParams['figure.figsize']=(15,10);
sd(df.loc['04-2020':'06-2020']['hospitalizedCurrently']).plot();
```


    
![png](Covid_Notebook_files/Covid_Notebook_125_0.png)
    



```python
stepwise_fit, stepwise_full, results, results_full = arima_tune(df, 'hospitalizedCurrently', 
                                                                days_to_forecast=30, m_periods=7, 
                                                                verbose=True) 
# train days arg defaults to 270 days, but can be changed. seasonality can be adjusted as well
# see docstring for further details
```

    ARIMA order is:  (0, 2, 1)
    Seasonal ARIMA order is:  (0, 0, 0, 7)
    Use ARIMA object stepwise_fit to store ARIMA and seasonal ARIMA orders in variables.
    


<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>   <td>hospitalizedCurrently</td> <th>  No. Observations:  </th>    <td>240</td>   
</tr>
<tr>
  <th>Model:</th>             <td>SARIMAX(0, 2, 1)</td>    <th>  Log Likelihood     </th> <td>-1560.634</td>
</tr>
<tr>
  <th>Date:</th>              <td>Tue, 16 Feb 2021</td>    <th>  AIC                </th> <td>3125.269</td> 
</tr>
<tr>
  <th>Time:</th>                  <td>10:35:56</td>        <th>  BIC                </th> <td>3132.197</td> 
</tr>
<tr>
  <th>Sample:</th>               <td>05-19-2020</td>       <th>  HQIC               </th> <td>3128.061</td> 
</tr>
<tr>
  <th></th>                     <td>- 01-13-2021</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>          <td>opg</td>          <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ma.L1</th>  <td>   -0.8240</td> <td>    0.031</td> <td>  -26.164</td> <td> 0.000</td> <td>   -0.886</td> <td>   -0.762</td>
</tr>
<tr>
  <th>sigma2</th> <td> 3.242e+04</td> <td> 1014.928</td> <td>   31.942</td> <td> 0.000</td> <td> 3.04e+04</td> <td> 3.44e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>0.07</td> <th>  Jarque-Bera (JB):  </th> <td>2986.65</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.79</td> <th>  Prob(JB):          </th>  <td>0.00</td>  
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.94</td> <th>  Skew:              </th>  <td>-2.10</td> 
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.78</td> <th>  Kurtosis:          </th>  <td>19.91</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).



    
![png](Covid_Notebook_files/Covid_Notebook_126_2.png)
    


#### The ends of the q-q plot are not quite in line.


```python
evaluate_predictions(results, df, 'hospitalizedCurrently', 
                     stepwise_fit=stepwise_fit, alpha=.05, days_to_forecast=30)
# plot training time and test time
```


    
![png](Covid_Notebook_files/Covid_Notebook_128_0.png)
    



```python
exog_forecast, forecast_obj = build_SARIMAX_forecast(model=results_full, 
                                                     dataframe=df, 
                                                     target_column='hospitalizedCurrently', 
                                                     days_to_forecast=30, stepwise_fit=stepwise_full, 
                                                     alpha=.05,
                                                     state_postal_code='TX')
```


    
![png](Covid_Notebook_files/Covid_Notebook_129_0.png)
    



```python
# graph of those currently hospitalized and subsequent forecast above shows
# a declining rate of hospitalization with a chance to remain flat. 
```

### Texas Deaths Forecast
#### Modeled using Texas's Hospitalized Forecast


```python
stepwise_fit, df_forecast = get_exogenous_forecast_dataframe(dataframe=df,
                                                             original_dataframe=df_ref,
                                                             exog_forecast=exog_forecast, 
                                                             target_column='death',
                                                             exogenous_column='hospitalizedCurrently',
                                                             days_to_forecast=30,
                                                             m_periods=7)
```


```python
# get exogenous forecast dataframe will return an extended dataframe
# containing the forecasted exogenous column from build_SARIMAX_forecast
# above after taking in the variable exog_forecast
```


```python
full_exog_model = SARIMAX(df['death'],df['hospitalizedCurrently'],
                          order=stepwise_fit.order,seasonal_order=stepwise_fit.seasonal_order)
```


```python
# fit model 
model = full_exog_model.fit()
```


```python
exog_forecast, results_forecast = build_SARIMAX_forecast(model=model,
                                                         dataframe=df_forecast, 
                                                         target_column='death', 
                                                         days_to_forecast=30, 
                                                         stepwise_fit=stepwise_fit, 
                                                         alpha=.05,
                                                         original_df=df_ref,
                                                         exogenous_column='hospitalizedCurrently',
                                                         state_postal_code='TX')
```


    
![png](Covid_Notebook_files/Covid_Notebook_136_0.png)
    



```python
results_forecast.conf_int()[-5:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lower death</th>
      <th>upper death</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-03-10</th>
      <td>44698.502289</td>
      <td>49384.094774</td>
    </tr>
    <tr>
      <th>2021-03-11</th>
      <td>44967.253255</td>
      <td>49898.574043</td>
    </tr>
    <tr>
      <th>2021-03-12</th>
      <td>45185.608779</td>
      <td>50364.084997</td>
    </tr>
    <tr>
      <th>2021-03-13</th>
      <td>45360.771459</td>
      <td>50804.571114</td>
    </tr>
    <tr>
      <th>2021-03-14</th>
      <td>45382.988130</td>
      <td>51095.233892</td>
    </tr>
  </tbody>
</table>
</div>



### Texas Analysis and Recommendations

#### The number of people currently hospitalized has decreased in recent weeks and I am forecasting that to continue. However, deaths are forecasted to slow only slightly over the next 30 days. There is some improvement, but it isn't enough.

##### Recommendations for the state include the following:
##### * Limit private social gatherings. 
##### * There are those in Texas who believe that the virus is a joke, that the rules don't apply to them, and that there are no consequences. Like California, an awareness campaign with personal stories and visual evidence of what Covid does could help contain the virus. 
##### * Make wearing a mask outside of one's private property required. 

### Florida Deaths Forecast
#### Modeled using Hospitalized Currently Forecast


```python
# change to True, run cell to follow link
open_links = False

import webbrowser

if open_links == True:
    webbrowser.open("https://www.newsweek.com/covid-florida-travel-advice-state-new-uk-variant-cases-1568572")
```

#### Florida currently has 300 new cases of the UK variant of Covid with 0 travel restrictions in place. The spread of the virus within Florida and throughout the rest of the United States as a result of a lack of travel bans and restraint is a serious issue. This section will forecast rate of death, total deaths, and will use the number of people currently hospitalized as an exogeous forecast. (data as of 2-10-2021)


```python
df_ref = state_dataframe(df_states, 'FL')
```

    Successfully returned indexed dataframe for FL
    


```python
df_ref.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>death</th>
      <th>inIcuCurrently</th>
      <th>onVentilatorCurrently</th>
      <th>positive</th>
      <th>hospitalizedCurrently</th>
      <th>deathIncrease</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-02-08</th>
      <td>FL</td>
      <td>28287.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1751343.0</td>
      <td>5381.0</td>
      <td>126</td>
    </tr>
    <tr>
      <th>2021-02-09</th>
      <td>FL</td>
      <td>28526.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1758254.0</td>
      <td>5307.0</td>
      <td>239</td>
    </tr>
    <tr>
      <th>2021-02-10</th>
      <td>FL</td>
      <td>28691.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1765659.0</td>
      <td>5129.0</td>
      <td>165</td>
    </tr>
    <tr>
      <th>2021-02-11</th>
      <td>FL</td>
      <td>28871.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1774013.0</td>
      <td>4906.0</td>
      <td>180</td>
    </tr>
    <tr>
      <th>2021-02-12</th>
      <td>FL</td>
      <td>29061.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1781450.0</td>
      <td>4825.0</td>
      <td>190</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.rcParams['figure.figsize']=(15,10);
sd(df_ref.loc['10-2020':'12-2020']['hospitalizedCurrently']).plot(); # seasonality of about 6-7 days
```


    
![png](Covid_Notebook_files/Covid_Notebook_146_0.png)
    



```python
state_dataframe, exog_forecast = create_exog_forecast(df_states, 'hospitalizedCurrently', 
                                                      days_to_forecast=45, m_periods=7, 
                                                      state_postal_code='FL')
```

    Successfully returned indexed dataframe for FL
    ARIMA order is:  (0, 1, 0)
    Seasonal ARIMA order is:  (0, 0, 1, 6)
    Use ARIMA object stepwise_fit to store ARIMA and seasonal ARIMA orders in variables.
    


<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>hospitalizedCurrently</td>      <th>  No. Observations:  </th>    <td>225</td>   
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(0, 1, 0)x(0, 0, [1], 6)</td> <th>  Log Likelihood     </th> <td>-1650.364</td>
</tr>
<tr>
  <th>Date:</th>                   <td>Tue, 16 Feb 2021</td>         <th>  AIC                </th> <td>3304.729</td> 
</tr>
<tr>
  <th>Time:</th>                       <td>08:34:47</td>             <th>  BIC                </th> <td>3311.489</td> 
</tr>
<tr>
  <th>Sample:</th>                    <td>05-19-2020</td>            <th>  HQIC               </th> <td>3307.460</td> 
</tr>
<tr>
  <th></th>                          <td>- 12-29-2020</td>           <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>               <td>opg</td>               <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ma.S.L6</th> <td>    0.1485</td> <td>    0.234</td> <td>    0.634</td> <td> 0.526</td> <td>   -0.311</td> <td>    0.608</td>
</tr>
<tr>
  <th>sigma2</th>  <td> 2.362e+05</td> <td> 2300.288</td> <td>  102.690</td> <td> 0.000</td> <td> 2.32e+05</td> <td> 2.41e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>0.71</td> <th>  Jarque-Bera (JB):  </th> <td>335172.89</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.40</td> <th>  Prob(JB):          </th>   <td>0.00</td>   
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.01</td> <th>  Skew:              </th>   <td>13.52</td>  
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.00</td> <th>  Kurtosis:          </th>  <td>193.63</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).



    
![png](Covid_Notebook_files/Covid_Notebook_147_2.png)
    



```python
forecast_object = graph_exog_forecast(dataframe=state_dataframe, 
                                      target_column='deathIncrease', 
                                      exogenous_column='hospitalizedCurrently', 
                                      exog_forecast=exog_forecast,
                                      df_ref=df_ref, 
                                      alpha=.05, days_to_forecast=45, 
                                      train_days=270, m_periods=7,
                                      state_postal_code='FL')
```


    
![png](Covid_Notebook_files/Covid_Notebook_148_0.png)
    



```python
forecast_object = graph_exog_forecast(dataframe=state_dataframe, 
                                      target_column='death', 
                                      exogenous_column='hospitalizedCurrently', 
                                      exog_forecast=exog_forecast,
                                      df_ref=df_ref, 
                                      alpha=.05, days_to_forecast=45, 
                                      train_days=270, m_periods=7,
                                      state_postal_code='FL')
```


    
![png](Covid_Notebook_files/Covid_Notebook_149_0.png)
    



```python
forecast_object.predicted_mean[-5:] # projected mean deaths for
# by the end of March 
```




    2021-03-25    35837.119536
    2021-03-26    36003.597226
    2021-03-27    36168.891724
    2021-03-28    36333.435035
    2021-03-29    36498.392480
    Freq: D, Name: predicted_mean, dtype: float64



### Florida Analysis and Recommendations

#### Florida does not have any travel restrictions in place. There have been and will continue to be a steady rate of Covid deaths, likely reaching over 35000 by the end of March. 

##### Recommendations for the state include the following:
##### * Florida is fully opened - implement travel restrictions and reduce the number of people allowed to privately gather.
##### * Pre and post-Super Bowl footage showed business as usual with zero mask usage. Implement and enforce laws requiring masks in public. There are plenty of states that are responsibly open that are mitigating the spread of this virus. Florida seems to be encouraging the spread. 
##### * Common sense is easy. Don't breathe into peoples' faces, wear a mask, and keep your distance. 

### United States Death Forecast


```python
# change to True, run cell to follow link(s)
open_links = False

import webbrowser

if open_links == True:
    webbrowser.open("")
```


```python
df_whole_US.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>states</th>
      <th>positive</th>
      <th>negative</th>
      <th>pending</th>
      <th>hospitalizedCurrently</th>
      <th>hospitalizedCumulative</th>
      <th>inIcuCurrently</th>
      <th>inIcuCumulative</th>
      <th>onVentilatorCurrently</th>
      <th>onVentilatorCumulative</th>
      <th>...</th>
      <th>lastModified</th>
      <th>recovered</th>
      <th>total</th>
      <th>posNeg</th>
      <th>deathIncrease</th>
      <th>hospitalizedIncrease</th>
      <th>negativeIncrease</th>
      <th>positiveIncrease</th>
      <th>totalTestResultsIncrease</th>
      <th>hash</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-02-12</th>
      <td>56</td>
      <td>27266230.0</td>
      <td>122400369.0</td>
      <td>9434.0</td>
      <td>71504.0</td>
      <td>839119.0</td>
      <td>14775.0</td>
      <td>43389.0</td>
      <td>4849.0</td>
      <td>4126.0</td>
      <td>...</td>
      <td>2021-02-12T24:00:00Z</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>5418</td>
      <td>2345</td>
      <td>567071</td>
      <td>100570</td>
      <td>1816007</td>
      <td>c402515c19b77ba9243af172a9c5799f13cd8e56</td>
    </tr>
    <tr>
      <th>2021-02-11</th>
      <td>56</td>
      <td>27165660.0</td>
      <td>121833298.0</td>
      <td>11981.0</td>
      <td>74225.0</td>
      <td>836774.0</td>
      <td>15190.0</td>
      <td>43291.0</td>
      <td>4970.0</td>
      <td>4113.0</td>
      <td>...</td>
      <td>2021-02-11T24:00:00Z</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>3873</td>
      <td>2460</td>
      <td>588596</td>
      <td>102417</td>
      <td>1872586</td>
      <td>9e06c1c2bc7906114b2dfb77c02fac6a1ff15c7c</td>
    </tr>
    <tr>
      <th>2021-02-10</th>
      <td>56</td>
      <td>27063243.0</td>
      <td>121244702.0</td>
      <td>12079.0</td>
      <td>76979.0</td>
      <td>834314.0</td>
      <td>15788.0</td>
      <td>43184.0</td>
      <td>5121.0</td>
      <td>4106.0</td>
      <td>...</td>
      <td>2021-02-10T24:00:00Z</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>3445</td>
      <td>3226</td>
      <td>385138</td>
      <td>95194</td>
      <td>1393156</td>
      <td>a821a2f23aaee791d155df7e3a2755b31c1bdd32</td>
    </tr>
    <tr>
      <th>2021-02-09</th>
      <td>56</td>
      <td>26968049.0</td>
      <td>120859564.0</td>
      <td>10516.0</td>
      <td>79179.0</td>
      <td>831088.0</td>
      <td>16129.0</td>
      <td>43000.0</td>
      <td>5216.0</td>
      <td>4092.0</td>
      <td>...</td>
      <td>2021-02-09T24:00:00Z</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2795</td>
      <td>3144</td>
      <td>492086</td>
      <td>92986</td>
      <td>1502502</td>
      <td>0ad7fb536eb23f95461201090c436ec7f76ac052</td>
    </tr>
    <tr>
      <th>2021-02-08</th>
      <td>56</td>
      <td>26875063.0</td>
      <td>120367478.0</td>
      <td>12114.0</td>
      <td>80055.0</td>
      <td>827944.0</td>
      <td>16174.0</td>
      <td>42833.0</td>
      <td>5260.0</td>
      <td>4080.0</td>
      <td>...</td>
      <td>2021-02-08T24:00:00Z</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>1309</td>
      <td>1638</td>
      <td>454325</td>
      <td>77737</td>
      <td>1434298</td>
      <td>7abf3026a5235e6761608e2971df85adb1c9bb18</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
columns =  ['death',
            'inIcuCurrently',
            'onVentilatorCurrently',
            'positive',
            'hospitalizedCurrently',
            'deathIncrease']
            # sort_and_clean_df(df_whole_US, columns)
```


```python
df_whole_US = sort_and_clean_df(df_whole_US, columns)
```


```python
sd(df_whole_US['hospitalizedCurrently']).plot();
```


    
![png](Covid_Notebook_files/Covid_Notebook_159_0.png)
    



```python
dataframe, exog_forecast = create_exog_forecast(df_whole_US, 'hospitalizedCurrently', 
                                                days_to_forecast=30, m_periods=7)
```

    ARIMA order is:  (1, 2, 1)
    Seasonal ARIMA order is:  (2, 0, 0, 7)
    Use ARIMA object stepwise_fit to store ARIMA and seasonal ARIMA orders in variables.
    


<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>hospitalizedCurrently</td>     <th>  No. Observations:  </th>    <td>240</td>   
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(1, 2, 1)x(2, 0, [], 7)</td> <th>  Log Likelihood     </th> <td>-1825.622</td>
</tr>
<tr>
  <th>Date:</th>                   <td>Tue, 16 Feb 2021</td>        <th>  AIC                </th> <td>3661.243</td> 
</tr>
<tr>
  <th>Time:</th>                       <td>08:35:43</td>            <th>  BIC                </th> <td>3678.279</td> 
</tr>
<tr>
  <th>Sample:</th>                    <td>05-19-2020</td>           <th>  HQIC               </th> <td>3668.121</td> 
</tr>
<tr>
  <th></th>                          <td>- 01-13-2021</td>          <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>               <td>opg</td>              <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ar.L1</th>    <td>    0.1604</td> <td>    0.071</td> <td>    2.265</td> <td> 0.023</td> <td>    0.022</td> <td>    0.299</td>
</tr>
<tr>
  <th>ma.L1</th>    <td>   -0.8665</td> <td>    0.054</td> <td>  -16.195</td> <td> 0.000</td> <td>   -0.971</td> <td>   -0.762</td>
</tr>
<tr>
  <th>ar.S.L7</th>  <td>    0.3457</td> <td>    0.032</td> <td>   10.649</td> <td> 0.000</td> <td>    0.282</td> <td>    0.409</td>
</tr>
<tr>
  <th>ar.S.L14</th> <td>    0.1347</td> <td>    0.054</td> <td>    2.509</td> <td> 0.012</td> <td>    0.029</td> <td>    0.240</td>
</tr>
<tr>
  <th>sigma2</th>   <td> 7.502e+05</td> <td> 2.72e+04</td> <td>   27.613</td> <td> 0.000</td> <td> 6.97e+05</td> <td> 8.03e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>0.14</td> <th>  Jarque-Bera (JB):  </th> <td>3442.77</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.70</td> <th>  Prob(JB):          </th>  <td>0.00</td>  
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.70</td> <th>  Skew:              </th>  <td>1.75</td>  
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.12</td> <th>  Kurtosis:          </th>  <td>21.93</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).



    
![png](Covid_Notebook_files/Covid_Notebook_160_2.png)
    



```python
stepwise_fit, stepwise_full, results, results_full = arima_tune(df_whole_US, 'hospitalizedCurrently', 
                                                                days_to_forecast=30, m_periods=7, 
                                                                verbose=True) 
```

    ARIMA order is:  (1, 2, 1)
    Seasonal ARIMA order is:  (2, 0, 0, 7)
    Use ARIMA object stepwise_fit to store ARIMA and seasonal ARIMA orders in variables.
    


<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>hospitalizedCurrently</td>     <th>  No. Observations:  </th>    <td>240</td>   
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(1, 2, 1)x(2, 0, [], 7)</td> <th>  Log Likelihood     </th> <td>-1825.622</td>
</tr>
<tr>
  <th>Date:</th>                   <td>Tue, 16 Feb 2021</td>        <th>  AIC                </th> <td>3661.243</td> 
</tr>
<tr>
  <th>Time:</th>                       <td>08:36:07</td>            <th>  BIC                </th> <td>3678.279</td> 
</tr>
<tr>
  <th>Sample:</th>                    <td>05-19-2020</td>           <th>  HQIC               </th> <td>3668.121</td> 
</tr>
<tr>
  <th></th>                          <td>- 01-13-2021</td>          <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>               <td>opg</td>              <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ar.L1</th>    <td>    0.1604</td> <td>    0.071</td> <td>    2.265</td> <td> 0.023</td> <td>    0.022</td> <td>    0.299</td>
</tr>
<tr>
  <th>ma.L1</th>    <td>   -0.8665</td> <td>    0.054</td> <td>  -16.195</td> <td> 0.000</td> <td>   -0.971</td> <td>   -0.762</td>
</tr>
<tr>
  <th>ar.S.L7</th>  <td>    0.3457</td> <td>    0.032</td> <td>   10.649</td> <td> 0.000</td> <td>    0.282</td> <td>    0.409</td>
</tr>
<tr>
  <th>ar.S.L14</th> <td>    0.1347</td> <td>    0.054</td> <td>    2.509</td> <td> 0.012</td> <td>    0.029</td> <td>    0.240</td>
</tr>
<tr>
  <th>sigma2</th>   <td> 7.502e+05</td> <td> 2.72e+04</td> <td>   27.613</td> <td> 0.000</td> <td> 6.97e+05</td> <td> 8.03e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>0.14</td> <th>  Jarque-Bera (JB):  </th> <td>3442.77</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.70</td> <th>  Prob(JB):          </th>  <td>0.00</td>  
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.70</td> <th>  Skew:              </th>  <td>1.75</td>  
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.12</td> <th>  Kurtosis:          </th>  <td>21.93</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).



    
![png](Covid_Notebook_files/Covid_Notebook_161_2.png)
    



```python
exog_forecast, results_forecast = build_SARIMAX_forecast(model=results_full, 
                     dataframe=df_whole_US, 
                     target_column='hospitalizedCurrently', 
                     days_to_forecast=30, stepwise_fit=stepwise_full, 
                     alpha=.05)
```


    
![png](Covid_Notebook_files/Covid_Notebook_162_0.png)
    



```python
results_forecast.conf_int()[-5:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lower hospitalizedCurrently</th>
      <th>upper hospitalizedCurrently</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-03-10</th>
      <td>-37162.653996</td>
      <td>56833.546329</td>
    </tr>
    <tr>
      <th>2021-03-11</th>
      <td>-42675.077866</td>
      <td>56981.481885</td>
    </tr>
    <tr>
      <th>2021-03-12</th>
      <td>-48225.243116</td>
      <td>57207.320274</td>
    </tr>
    <tr>
      <th>2021-03-13</th>
      <td>-53868.515937</td>
      <td>57624.618009</td>
    </tr>
    <tr>
      <th>2021-03-14</th>
      <td>-59633.796094</td>
      <td>58077.962479</td>
    </tr>
  </tbody>
</table>
</div>




```python
forecast_object_deaths = graph_exog_forecast(dataframe=dataframe, 
                                      target_column='death', 
                                      exogenous_column='hospitalizedCurrently', 
                                      exog_forecast=exog_forecast,
                                      df_ref=df_ref, 
                                      alpha=.05, days_to_forecast=30, 
                                      train_days=270, m_periods=7)
```


    
![png](Covid_Notebook_files/Covid_Notebook_164_0.png)
    



```python
forecast_object_deaths.conf_int(alpha=.05)[-5:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lower death</th>
      <th>upper death</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-03-10</th>
      <td>535041.892460</td>
      <td>689517.817213</td>
    </tr>
    <tr>
      <th>2021-03-11</th>
      <td>536047.723241</td>
      <td>699351.622777</td>
    </tr>
    <tr>
      <th>2021-03-12</th>
      <td>536973.512268</td>
      <td>709267.354209</td>
    </tr>
    <tr>
      <th>2021-03-13</th>
      <td>537822.057496</td>
      <td>719264.943468</td>
    </tr>
    <tr>
      <th>2021-03-14</th>
      <td>538590.395094</td>
      <td>729338.707643</td>
    </tr>
  </tbody>
</table>
</div>




```python
forecast_object = graph_exog_forecast(dataframe=dataframe, 
                                      target_column='deathIncrease', 
                                      exogenous_column='hospitalizedCurrently', 
                                      exog_forecast=exog_forecast,
                                      df_ref=df_ref, 
                                      alpha=.025, days_to_forecast=30, 
                                      train_days=270, m_periods=7)
```


    
![png](Covid_Notebook_files/Covid_Notebook_166_0.png)
    


### United States Analysis and Recommendations

#### The holidays were literal killers. There is no getting around that fact, and without accelerated vaccinations we will continue to have increased deaths around future holidays. We have an opportunity to stop the spread of this virus with everyone's cooperation and resolve. My forecast for the country predicts there's a chance to truly slow and potentially stop the pandemic and Covid deaths by mid-March. There is also the possibility we continue to be irresponsible as a whole and deaths accelerate into the end of March.

##### I recommend that we come together as Americans in this one endeavor - to protect one another in what is a time of unmitigated risk. 
##### * WEAR A MASK
##### * STAY AWAY FROM PEOPLE YOU DONT LIVE WITH and respect their space. If you are someone who believes this is a hoax and you see someone with a mask on, just stay away from them. 
##### * Use the drive-thru whenever possible, have food delivered. Let's do our best to keep the food service industry going without sacrificing common sense. 
##### * Have your groceries delivered - grocery stores continue to stay extremely busy but there are services that are helping individuals pay their bills that can help limit the spread of this disease. 
##### * Limit your time indoors with friends and family not in your household and wear a mask when taking that time with them. 

# Summary Recommendations

## State by State Recommendations

### New York

##### Recommendations for the state include the following:
##### *Lower the number of people allowed at indoor private gatherings from the current amount of 10. Social gatherings are not economically essential. Certain states have limits of one or two households per private residence which has proven to limit the spread of the disease. 
##### * Increase effort to improve awareness. Covering the mouth but not the nose does not limit the spread of Covid-19. Mandate signage depicting proper mask usage at public establishments. 
##### * Continue social distancing policies and reduce the number of outdoor events. The pandemic in New York is not under control. 

### California

##### Recommendations for the state include the following:
##### * Require wearing a mask if an individual is not in or on their private property. Allow no exceptions. 
##### * Prohibit private and public gatherings of 5 or more people unless from the same household. 
##### * The spread of this disease in this state will continue to take lives if people are not made to understand the consequences of selfish behavior. Introduce visual evidence of the rammifications of the virus with an emphasis on personal stories on public social media and television. 

### Texas

##### Recommendations for the state include the following:
##### * Limit private social gatherings. 
##### * There are those in Texas who believe that the virus is a joke, that the rules don't apply to them, and that there are no consequences. Like California, an awareness campaign with personal stories and visual evidence of what Covid does could help contain the virus. 
##### * Make wearing a mask outside of one's private property required. 

### Florida

##### Recommendations for the state include the following:
##### * Florida is fully opened - implement travel restrictions and reduce the number of people allowed to privately gather.
##### * Pre and post-Super Bowl footage showed business as usual with zero mask usage. Implement and enforce laws requiring masks in public. There are plenty of states that are responsibly open that are mitigating the spread of this virus. Florida seems to be encouraging the spread. 
##### * Common sense is easy. Don't breathe into peoples' faces, wear a mask, and keep your distance. 

### United States

##### I recommend that we come together as Americans in this one endeavor - to protect one another in what is a time of unmitigated risk. 
##### * WEAR A MASK
##### * STAY AWAY FROM PEOPLE YOU DONT LIVE WITH and respect their space. If you are someone who believes this is a hoax and you see someone with a mask on, just stay away from them. 
##### * Use the drive-thru whenever possible, have food delivered. Let's do our best to keep the food service industry going without sacrificing common sense. 
##### * Have your groceries delivered - grocery stores continue to stay extremely busy but there are services that are helping individuals pay their bills that can help limit the spread of this disease. 
##### * Limit your time indoors with friends and family not in your household and wear a mask when taking that time with them. 

## Conclusions

I chose to undertake this project for several reasons. It is relevant to what is happening now, and it has real implications in peoples' lives. On a much more personal level, it is frustrating to have three grandparents in their 90's whom I am unable to see at this time. Additionally, my mother has an auto-immune deficiency, which makes her risk around irresponsible individuals that much more real for me.

Most people seem to be respectful enough to wear a mask, but after nearly a year many are growing tired of this simple task. I hope that respect for others will prevail over the desire for personal freedom. The covenant that we enter into as citizens is with each other. It's to protect each other, and it requires that we be considerate and understand that one person's wants (the desire to not wear a mask, to party, to have a good time) do not supercede the responsibility to protect our fellow Americans.

With vaccine distribution occurring, we don't have much longer to endure the difficulties.

## Future Work

#### * Continue to update the analysis until the project ceases functioning on March 7th, 2021. 
#### * Find a future source of data to actively pull in and compare future actual data with the forecasts in this notebook.

# Appendix and Ancillary Code

## Using an Exogenous Variable with SARIMAX


```python
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

calendar = calendar()
holidays = calendar.holidays(start=df_states.index.min(), end=df_states.index.max())

df_states['holiday'] = df_states.index.isin(holidays)

df_whole_US['holiday'] = df_whole_US.index.isin(holidays)
```


```python
df_states = sort_and_clean_df(df_states,.05)
```


```python
state_postal_code = 'TX'
days = 30

df_state = df_states[df_states['state']==state_postal_code]    

# sort index, lowest index to oldest date, drop na's in death column
df_state = df_state.sort_index()
df_state = df_state.dropna(subset=['death'])
df_state_new = pd.DataFrame(df_state)

#     ets_decomp = sd(df_state_new['death'])
#     ets_decomp.plot();

# create stepwise fit model, see summary
stepwise_fit = auto_arima(df_state_new['death'],seasonal=True,m=52,maxiter=2)

# auto_arima automatically differences and returns that differencing for the model in the arima_order = stepwise_fit.order below
## find correct ARIMA order

arima_order = stepwise_fit.order
seasonal_order = stepwise_fit.seasonal_order

length = len(df_state_new)-days

train_data = df_state_new.iloc[:length]
test_data = df_state_new.iloc[length:]

model = sm.tsa.statespace.SARIMAX(train_data['death'], trend='ct', seasonal_order=seasonal_order, 
                                  order=arima_order, enforce_invertibility=False)
res = model.fit()

start = len(train_data)
end = len(train_data) + len(test_data) - 1

predictions_state = res.predict(start,end,dynamic=False).rename(f'SARIMAX {arima_order} Predictions')

# ensure predictions are in DataFrame format, label index as date to match df_alaska
predictions_state = pd.DataFrame(predictions_state)
predictions_state.index.name = 'date'

train_data.index.freq = 'D'
test_data.index.freq = 'D' # -1D is reverse index, ie most recent date is at top of dataframe
# perform sort_index on dataframe to correct. set frequencies to match for plotting
# on same visualization

# graph test vs. prediction data - {PLOT}
legend_elements = [Line2D([0], [0], color='b', lw=4, label='Actual Deaths'),
                   Line2D([0], [0], color='#FFA500', lw=4, label=f'SARIMAX {arima_order} Predictions')]

fig, ax = plt.subplots(figsize=(20,10));
ax.plot(test_data['death'])
ax.plot(predictions_state);
ax.grid(b=True,alpha=.5)
plt.title(f'Test Data vs SARIMA, {state_postal_code}')
ax.legend(handles=legend_elements)
for x in test_data.index:
    if test_data['holiday'].loc[x]==1:    # for days where holiday == 1
        ax.axvline(x=x, color='red', alpha = 0.4);   
plt.xlabel('Date')
plt.ylabel('Deaths')
plt.show();

error1 = mse(test_data['death'], predictions_state)
error2 = rmse(test_data['death'], predictions_state)

# print(f'SARIMAX{arima_order}{seasonal_order} MSE Error: {error1}')
# print(f'SARIMAX{arima_order}{seasonal_order} RMSE Error: {error2}')

# train model for forecast
model = sm.tsa.statespace.SARIMAX(df_state_new['death'],exog=df_state_new['holiday'],
                                  order=arima_order, seasonal_order=seasonal_order,
                                  enforce_invertibility=False)
res = model.fit(disp=False)

# create forecast
exog_forecast = df_state_new[length:][['holiday']]
fcast = res.predict(start=len(df_state_new),end=len(df_state_new)+days-1,exog=exog_forecast).rename(f'SARIMAX{arima_order},{seasonal_order} {days} Days Forecast')

# graph forecast deaths, breakout of train and test split is present in graph - {PLOT}
legend_elements = [Line2D([0], [0], color='b', lw=5, label='Actual Deaths'),
                   Line2D([0], [0], color='#FFA500', lw=5, label='Actual Deaths'),
                   Line2D([0], [0], color='g', lw=5, label=f'SARIMAX {arima_order} , {seasonal_order} Predictions'),
                   Line2D([0], [0], color='r', lw=5, label=f'SARIMAX {arima_order} , {seasonal_order} {days} Day Forecast')]

fig, ax = plt.subplots(figsize=(20,10));
ax.plot(train_data['death'])
ax.plot(test_data['death'])
ax.plot(predictions_state)
ax.plot(fcast)
ax.grid(b=True,alpha=.5)
plt.title(f'SARIMAX {days} Day Forecast, {state_postal_code}')
ax.legend(handles=legend_elements)
for x in df_state_new.index:
    if df_state_new['holiday'].loc[x]==1:    # for days where holiday == 1
        ax.axvline(x=x, color='red', alpha = 0.4);   
plt.xlabel('Date')
plt.ylabel('Deaths')
plt.show();

last_predictions = len(fcast)-5
actual_numbers = fcast[last_predictions:]

return actual_numbers
```

## Univariate Forecast with RNN

### Texas


```python
create_NN_predict(df_states=df_states,state_postal_code='TX',days=25,epochs=4) 
```

### Florida


```python
create_NN_predict(df_states,'FL',20,epochs=4)
```

### California


```python
create_NN_predict(df_states,'CA',20,epochs=4)
```

## Multivariate Forecast with RNN

### Data Import and Ventilator/Death Plot


```python
# initialize Df
df_whole_US = pd.read_csv('https://api.covidtracking.com/v1/us/daily.csv',index_col='date',parse_dates=True)
```


```python
# we can see some lag here, common sense tells us we will probably see a 
#decrease in death rate after ICU and Ventilator populations fall. 
df_whole_US['inIcuCurrently'].plot(legend=True, figsize=(15,7))
df_whole_US['onVentilatorCurrently'].plot(legend=True, figsize=(15,7))
(df_whole_US['death']/10).plot(legend=True);  # to scale to other graphs
```


```python
mv_forecast(df_whole_US,days_to_train=50,days_to_forecast=20,epochs=100) 
#hyperparameters to optimize days to train?
```


```python

```
