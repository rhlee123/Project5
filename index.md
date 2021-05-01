# Predicting S&P 500 Daily Returns

The goal of this project is to evaluate the performance of different regressors when predicting the daily S&P 500 returns for a day using the daily returns of other international stock market indices that day as well as the prior day. The marktet indices used as features for predicting S&P 500 daily return are the Istanbul Stock market return index, the Istanbul stock exchange national 100 index, the stock market return index of Germany, stock market return index of UK, stock market return index of Japan, stock market return index of Brazil, MSCI European index, and MSCI emerging markets index. The performance of models examined in this project will be evaluated by comparing the 10 split k-fold cross-validated mean absolute error for each of the models. 

## Data

Let us take a look at the data. Below I imported relevant packages as well as preprocessed my data. 
```python 
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from nadaraya_watson import NadarayaWatson, NadarayaWatsonCV
from pygam import LinearGAM
import numpy as np
import pandas as pd
from math import ceil
from scipy import linalg
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split as tts
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import operator
```
```python 
df = pd.read_csv('FinanceDataS.csv')
features = ['date','ISE1','ISE2','DAX','NIKKEI','BOVESPA','EU','EM','Prev_ISE1','Prev_ISE2','Prev_SP','Prev_DAX','Prev_NIKKEI','Prev_BOVESPA','Prev_EU','Prev_EM']
X = np.array(df[features])
y = np.array(df['SP']).reshape(-1,1)
```
Now let us take a look at how our dataset looks. Although there is a date column, that column will be ignored for the purposes of our analysis. 
![image](https://user-images.githubusercontent.com/55299814/116769362-86ec6900-aa09-11eb-802f-bc7ba32e75ad.png)

To further take a look at the stock market dataset, here is a heatmap showing the correlations between features in the dataset in which we will be using to predict daily S&P 500 returns:

![image](https://user-images.githubusercontent.com/55299814/116769394-c915aa80-aa09-11eb-8675-2a557ad5b74b.png)

