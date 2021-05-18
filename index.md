# Predicting S&P 500 Daily Returns

The goal of this project is to evaluate the performance of different regressors when predicting the daily S&P 500 returns for a day using the daily returns of other international stock market indices that day as well as the prior day in addition to daily 10-year treasury returns that day and the day prior. The market indices used as features for predicting S&P 500 daily return are the Istanbul Stock market return index, the Istanbul stock exchange national 100 index, the stock market return index of Germany, stock market return index of UK, stock market return index of Japan, stock market return index of Brazil, MSCI European index, and MSCI emerging markets index. The performance of models examined in this project will be evaluated by comparing the 10 split k-fold cross-validated mean absolute error for each of the models. For our analysis, we will compare our models to a simple baseline linear regression model. The particular models explored are neural networks, random forest, xgboost, gaussian loess regression, and regularization methods such as LASSO regression, RIDGE regression, Elastic Net regression, and SQRT LASSO regression. 

## Background

For the analysis, we will first focus on the evaluating performance of regularized regressions versus a baseline linear model in the situation of predicting daily S&P 500 stock market index return. Regularization methods are used to determine the weights for features within a model, and depending on the regularization technique, features can be excluded altogether from the model by having a weight of 0. Further, regularization is the process of of regularizing the parameters that constrains or coefficients estimates towards zeros, in otherwords discouraging learning a too complex or too flexible model, and ultimately helping to reduce the risk of overfitting. Regularization helps to choose the preferred model complexity, so that the model is better at predicting. Regularization is essentially adding a penalty term to the objective function, and controlling the model complexity using that penalty term. Regularization ultimately attempts to reduce the variance of the estimator by simplifying it, something that will increase the bias, in such a way that will decrease the expected error of the model's predictions. Additionally, Regularization is useful in tackling issues of multicolinearity among features becauase it incorporates the need to learn additional information for our model other than the observations in the data set.
  
Regularization techniques are especially useful in situations where there are large number of features or a low number of observations to number of features, where there is multicolinearity between the features, trying to seek a sparse solution, or accounting for for variable groupings in high dimensions. Regularization ultimately seeks to tackle the shortcomings of Ordinary Least Square models. 
  
### Ridge 
L2 regularization, or commonly known as ridge regularization, is a type of regularization that controls for the sizes of each coefficient or estimators. Not only does ridge regularization encorporate principles of OLS by reducing the sum of squared residuals, it also penalizes models with the regularization term of L2 Norm or
![\alpha \sum_{i=1}^p \beta_i^2](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Calpha+%5Csum_%7Bi%3D1%7D%5Ep+%5Cbeta_i%5E2).

Ridge regularization is especially useful when there is multicolinearity within data, and further, ridge regularization seeks to ultimately minimze the cost function:

![image](https://user-images.githubusercontent.com/55299814/116770213-c1590480-aa0f-11eb-9d0b-44690d598453.png)


(α) in this instance is the hyperparameter that determines the strength of regularization, or the strength of the penalty on the model. 
### LASSO
L1 regularization, or commonly known as Least Absolute Shrinkage and Selection Operator (LASSO) regularization, determines the weight of features by penalizing the model with the regularization term of L1 Norm or ![\alpha \sum_{i=1}^p |\beta_i| ](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Calpha+%5Csum_%7Bi%3D1%7D%5Ep+%7C%5Cbeta_i%7C+). 

Further, LASSO regularization seeks to ultimately minimze the cost function: 

![Capture](https://user-images.githubusercontent.com/55299814/111152630-4f178800-8567-11eb-9f12-49e834aa0f2a.PNG)

LASSO differs from ridge in that the way that LASSO penalizes high coefficients, as instead of squaring coefficients, LASSO takes the absolute value of the coefficients. Ultimately the weights of features can go to 0 using L1 norm, as opposed to L2 norm that ridge regularization in which weights can not go to 0. Ridge regularization will shrink the coefficients for least important features, very close to zero, however, will never make them exactly zero, resulting in the final model including all predictors. However, in the case of the LASSO, the L1 norm penalty has the eﬀect of forcing some of the coeﬃcient estimates to be exactly equal to zero when the tuning parameter (α) is suﬃciently large. Therefore, the lasso method, not only performs variable selection but is generally said to yield sparse models.
### Elastic Net
Elastic net regularization combines aspects of both ridge and LASSO regularization by including both L1 norm and L2 norm penalties. Elastic net determines the weights of features by minimizing the cost funciton where λ between 0 and 1: 

![\hat{\beta} = argmin_\beta \left\Vert  y-X\beta \right\Vert ^2 + \lambda_2\left\Vert  \beta \right\Vert ^2 + \lambda_1\left\Vert  \beta\right\Vert_1
![jkiyutyfcgvhbkjnkihyguh](https://user-images.githubusercontent.com/55299814/111017541-3d28c000-8382-11eb-9681-03df13e00f9f.png)
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Chat%7B%5Cbeta%7D+%3D+argmin_%5Cbeta+%5Cleft%5CVert++y-X%5Cbeta+%5Cright%5CVert+%5E2+%2B+%5Clambda_2%5Cleft%5CVert++%5Cbeta+%5Cright%5CVert+%5E2+%2B+%5Clambda_1%5Cleft%5CVert++%5Cbeta%5Cright%5CVert_1%0A) 

Elastic net regularization is a good middle ground between the other techniques, ridge and LASSO, because the technique allow for the model to learns weights that fit the multicolinearity and sparsity pattern within the data. 

### Square Root LASSO: 

Square Root LASSO slightly adjusts the LASSO method, in which it takes the square root of the LASSO cost function. It is important to note that L1 norm is still used for its penalty. Ultimately, lasso weights features by minimizing the cost function: 

![image](https://user-images.githubusercontent.com/55299814/118607213-5127de00-b786-11eb-87d3-9d1fba62fcca.png)

### Kernel Weighted Regression (LOESS)

Kernels use their respective functions to determine the weights of our data points for our locally weighted regression. Kernel weighted regressions work well for data that does not show linear qualities. For this regression we will be using statsmodels kernelreg function for the gaussian kernel. 

### Neural Network 

Neural Networks are models that look to recognize underlying relationships in a data set through a process that is similar to the way that the human brain works. Neural networks use activation functions to transform inputs. In a neural network, a neuron is a mathematical function that collects and classifies information according to a specific structure or architecture. The neural network ultimately goes through a learning process in which it fine tunes the connection strengths and relationships between neurons in the network to optimize the neural networks performance in solving a particular problem, which in our case is predicting the S&P return.

### Random Forest

Random Forests is a type of regression algorithm that expanands on the more basic Decision Trees algorithm. Random Forest algorithms create a 'forest' of Decision Trees that will be randomly sampled with replacement. Further, random forests will then give weights to each Decision Tree in order to control for some of the overfitting issues that are prevelant in normal Decision Tree algorithm.

### XGBOOST 

Boosting refers to a family of algorithms that look to turn weak learners into strong learners. In boosting, the individual models are built sequentially by putting more weight on instances where there are wrong predictions and high magnitudes of errors. The model will focus during learning on instances which are hard to predict correctly, so that the model in a sense learns from past mistakes. Extreme gradient boost is a decision-tree based algorithm that uses advanced gradient boosting and regularization to prevent overfitting.

### GridSearchCV

The Grid Search algorithm is method that is useful in adjusting the parameters in supervised learning models and aim improve the performance of a model by selecting a competitive hyperparameter value. The Grid Search algorithm works by examining all possible combinations of the parameters of interest and finds the best ones. I will use this algorithm to aim to find the best hyperparameter that yields the lowest MAE for my regularized regression. 

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
features = ['date','ISE1','ISE2','DAX','NIKKEI','BOVESPA','EU','EM','Treasury','Prev_ISE1','Prev_ISE2','Prev_SP','Prev_DAX','Prev_NIKKEI','Prev_BOVESPA','Prev_EU','Prev_EM','Prev_Treasury']
X = np.array(df[features])
y = np.array(df['SP']).reshape(-1,1)
```
Now let us take a look at how our dataset looks. Although there is a date column, that column will be ignored for the purposes of our analysis. 
![image](https://user-images.githubusercontent.com/55299814/116769362-86ec6900-aa09-11eb-802f-bc7ba32e75ad.png)

To further take a look at the stock market dataset, here is a heatmap showing the correlations between features in the dataset in which we will be using to predict daily S&P 500 returns:

![image](https://user-images.githubusercontent.com/55299814/118596819-e07ac480-b779-11eb-84fa-eb5ee5c37e24.png)

Clearly, there is strong correlation between the features, which indicates that regularization might be helpful in effectively estimating coefficients despite the multicollinearity present. Further, it indicates that regularization might reduce the MAE of a model that looks to predict price using these features. It should be noted as well that features are mostly negatively correlated with one another and because of the use of the previous day returns in the dataset as well, many features are also completely uncorrelated with one another. However, through in through, the correlation matrix above suggest that the use of regularization techniques may be helpful in more accurately predicting S&P 500 returns than the baseline linear model due to the excessive multicollinearity presaent. 

## Initializing Models

Below is code for intializing the a function to calculate the 10 fold K-fold cross-validated MAE for the different models used in this preliminary analysis.  

### Linear Regression 
```python 
def DoKFold(X,y,n):
  kf = KFold(n_splits=n,shuffle=True, random_state = 1234)
  mae = []
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    y_test  = y[idxtest]
    model = LinearRegression()
    model.fit(X_train, y_train)
    yhat_linear = model.predict(X_test)
    mae.append(mean_absolute_error(y_test,yhat_linear))
  return print("MAE Linear Model = {:,.4f}%".format(np.mean(mae)*100))
  ```
 
### Ridge 

```python 
def DoKFold(X,y,alpha,n):
  kf = KFold(n_splits=n,shuffle=True, random_state = 1234)
  mae = []
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    y_test  = y[idxtest]
    model = Ridge(alpha = alpha)
    model.fit(X_train, y_train)
    yhat_ridge = model.predict(X_test)
    mae.append(mean_absolute_error(y_test,yhat_ridge))
  return print("MAE Ridge Model = {:,.4f}%".format(np.mean(mae)*100))
  ```

### LASSO 

```python 
def DoKFold(X,y,alpha,n):
  kf = KFold(n_splits=n,shuffle=True, random_state = 1234)
  mae = []
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    y_test  = y[idxtest]
    model = Lasso(alpha = alpha)
    model.fit(X_train, y_train)
    yhat_lasso = model.predict(X_test)
    mae.append(mean_absolute_error(y_test,yhat_lasso))
  return print("MAE Lasso Model = {:,.4f}%".format(np.mean(mae)*100))
  ```

### Elastic Net

```python 
def DoKFold(X,y,alpha,l,n):
  kf = KFold(n_splits=n,shuffle=True, random_state = 1234)
  mae = []
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    y_test  = y[idxtest]
    model = ElasticNet(alpha = alpha, l1_ratio=l)
    model.fit(X_train, y_train)
    yhat_EN = model.predict(X_test)
    mae.append(mean_absolute_error(y_test,yhat_EN))
  return print("MAE Elastic Net Model = {:,.4f}%".format(np.mean(mae)*100))
```

### SQRT LASSO

```python 
def sqrtlasso_model(X,y,alpha):
  n = X.shape[0]
  p = X.shape[1]
  #we add an extra columns of 1 for the intercept
  #X = np.c_[np.ones((n,1)),X]
  def sqrtlasso(beta):
    beta = beta.flatten()
    beta = beta.reshape(-1,1)
    n = len(y)
    return np.sqrt(1/n*np.sum((y-X.dot(beta))**2)) + alpha*np.sum(np.abs(beta))
  
  
  def dsqrtlasso(beta):
    beta = beta.flatten()
    beta = beta.reshape(-1,1)
    n = len(y)
    return np.array((-1/np.sqrt(n))*np.transpose(X).dot(y-X.dot(beta))/np.sqrt(np.sum((y-X.dot(beta))**2))+alpha*np.sign(beta)).flatten()
  b0 = np.ones((p,1))
  output = minimize(sqrtlasso, b0, method='L-BFGS-B', jac=dsqrtlasso,options={'gtol': 1e-8, 'maxiter': 1e8,'maxls': 25,'disp': True})
  return output.x

def DoKFoldSqrt(X,y,a,k,d):
  PE = []
  kf = KFold(n_splits=k,shuffle=True,random_state=1234)
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    y_test  = y[idxtest]
    beta_sqrt = sqrtlasso_model(X_train,y_train,a)
    n = X_test.shape[0]
    p = X_test.shape[1]
    yhat_sqrt = X_test.dot(beta_sqrt)
    PE.append(MAE(y_test,yhat_sqrt))
  return 100*np.mean(PE)
```
### Kernel Weighted Regression 

```python def Gaussian(x):
  return np.where(np.abs(x)>1,0,np.exp(-1/2*x**2))
  ```

```python 
def DoKFold(X,y,n):
  kf = KFold(n_splits=n,shuffle=True, random_state = 1234)
  mae_Kern = []
  for idxtrain, idxtest in kf.split(X):
  X_train = X[idxtrain,:]
  y_train = y[idxtrain]
  X_test  = X[idxtest,:]
  y_test  = y[idxtest]
  model_KernReg = KernelReg(endog=y_train,exog=X_train,var_type='ccccccccccc',ckertype='gaussian')
  yhat_sm_test, y_std = model_KernReg.fit(X_test)
  mae_kern.append(mean_absolute_error(y_test, yhat_sm_test))
  print("Validated MAE Gaussian Regression = = {:,.4f}%".format(np.mean(mae_nn)*100)))
```
### Neural Network 

```python
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import r2_score
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
model = Sequential()
model.add(Dense(128, activation="relu", input_dim=1))
model.add(Dense(32, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="linear"))
model.compile(loss='mean_absolute_error', optimizer=Adam(lr=1e-3, decay=1e-3 / 200))
```
```python 
def DoKFold(X,y,n):
  kf = KFold(n_splits=n,shuffle=True, random_state = 1234)
  mae_nn = []
  for idxtrain, idxtest in kf.split(X):
  X_train = dat[idxtrain,:]
  y_train = dat[idxtrain,]
  X_test  = dat[idxtest,:]
  y_test = dat[idxtest,]
  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
  model.fit(X_train,y_train,validation_split=0.3, epochs=1000, batch_size=100, verbose=0, callbacks=[es])
  yhat_nn = model.predict(X_test.reshape(-1,1))
  mae_nn.append(mean_absolute_error(y_test, yhat_nn))
  print("Validated MAE Neural Network Regression = = {:,.4f}%".format(np.mean(mae_nn)*100)))
```

### XGBoost

```python 
import xgboost as xgb
model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
mae_xgb = []

def DoKFold(X,y,n):
  kf = KFold(n_splits=n,shuffle=True, random_state = 1234)
  mae_xgb = []
  model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
  for idxtrain, idxtest in kf.split(X):
  X_train = dat[idxtrain,:]
  y_train = dat[idxtrain]
  X_test  = dat[idxtest,:]
  y_test = dat[idxtest]
  model_xgb.fit(X_train.reshape(-1,1),y_train)
  yhat_xgb = model_xgb.predict(X_test.reshape(-1,1))
  mae_xgb.append(mean_absolute_error(y_test, yhat_xgb))
  return print("MAE XGBoost Regression = {:,.4f}%".format(np.mean(mae_xgb)*100))
```

### Random Forest 
```python 
  def DoKFold(X,y,n):
  kf = KFold(n_splits=n,shuffle=True, random_state = 1234)
  rf = RandomForestRegressor(n_estimators=1000,max_depth=3)
  mae_rf = []

  for idxtrain, idxtest in kf.split(X):
  X_train = X[idxtrain,:]
  y_train = y[idxtrain]
  X_test  = X[idxtest,:]
  y_test  = y[idxtest]
  rf.fit(X_train,y_train.ravel())
  yhat_rf = rf.predict(X_test)
  mae_rf.append(mean_absolute_error(y_test, yhat_rf))
  return print("MAE Random Forest = {:,.4f}%".format(np.mean(mae_rf)*100))
```

## Analysis/Conclusion 

Below are the 10 fold cross-validated MAE for the models used to predict the S&P 500 daily returns as well as the optimal hyperparameters. 

| Model                          | MAE               | Optimal α | Optimal L1 Ratio |
|--------------------------------|-------------------|---------- |------------------|
| MAE Linear Model               | 0.5915            |           |                  |              
| MAE Ridge Regression Model     | 0.5770            | 0.012     |                  |
| MAE LASSO Model                | 0.5742            | 0.000004  |                  |
| MAE Elastic Net Model          | 0.5745            | 0.0000015 | 2                | 
| MAE Square Root Lasso          | 0.5721            | 0.0000001 | 1.5              |
| MAE Gaussian Kernel Regression | 0.5788            |           |                  |
| MAE Neural Network             | 0.5775            |           |                  |
| MAE Random Forest              | 0.5791            |           |                  |
| MAE XGBoost                    | 0.5785            |           |                  |

It seems that all of the regularization methods outperformed the baseline linear model when predicing S&P 500 daily returns. All the regularization techniques performed better at predicting coefficients that would result in predictions that produce lower MAE's because the regularization techniques were configured to take into account data with multicolinearity present. In particular, the Square Root Lasso regression performed the best with the lowest MAE of .5728%, and performed especially well because the Square Root LASSO accounts for multicolinearity and penalizes in a way that produces sparse solutions.  Clearly there is merit to using feature selection or regularization techniques as opposed to a simple linear model for predicting daily S&P returns. The regular LASSO model performed second best, with Elastic Net the third, and ridge regression the 4th. It should also be noted that Neural networks also performed particular well in comparison to the baseline linear model and other non regularization techniques in obtaining predictions with relatively low MAE. Although the neural network model produced a competitive result, neural networks are very computational expensive and require a lot of computation power and time to run. Overall, it can be seen that using machine learning techniques, in particular regualarized regression could be potentially beneficial in attempting to predict daily S&P returns using the intraday returns of other major benchmark indices. It is evident that leveraging regularization techniques could help fit models that perform better at predicting S&P returns compared to baseline model.  


