# Predicting S&P 500 Daily Returns

The goal of this project is to evaluate the performance of different regressors when predicting the daily S&P 500 returns for a day using the daily returns of other international stock market indices that day as well as the prior day. The marktet indices used as features for predicting S&P 500 daily return are the Istanbul Stock market return index, the Istanbul stock exchange national 100 index, the stock market return index of Germany, stock market return index of UK, stock market return index of Japan, stock market return index of Brazil, MSCI European index, and MSCI emerging markets index. The performance of models examined in this project will be evaluated by comparing the 10 split k-fold cross-validated mean absolute error for each of the models. For our preliminary analysis, we will be looking at a simple baseline linear regression model and regularization methods such as LASSO regression, RIDGE regression, Elastic Net regression, and SQRT LASSO regression. 

## Background

For the preliminary analysis, we will mainly focusing on the evalutaing performance of regularized regressions versus a baseline linear model in the situation of predicting daily S&P 500 stock market index return. Regularization methods are used to determine the weights for features within a model, and depending on the regularization technique, features can be excluded altogether from the model by having a weight of 0. Further, regularization is the process of of regularizing the parameters that constrains or coefficients estimates towards zeros, in otherwords discouraging learning a too complex or too flexible model, and ultimately helping to reduce the risk of overfitting. Regularization helps to choose the preferred model complexity, so that the model is better at predicting. Regularization is essentially adding a penalty term to the objective function, and controlling the model complexity using that penalty term. Regularization ultmately attempts to reduce the variance of the estimator by simplifying it, something that will increase the bias, in such a way that will decrease the expected error of the model's predictions. Additionally, Regularization is useful in tackling issues of multicolinearity among features becauase it incorporates the need to learn additional information for our model other than the observations in the data set.
  
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

![\sqrt{\frac{1}{n}\sum\lim_{i=1}^{n}(y_i-\hat{y}_i)^2} +\alpha\sum\lim_{i=1}^{p}|\beta_i|
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Csqrt%7B%5Cfrac%7B1%7D%7Bn%7D%5Csum%5Clim_%7Bi%3D1%7D%5E%7Bn%7D%28y_i-%5Chat%7By%7D_i%29%5E2%7D+%2B%5Calpha%5Csum%5Clim_%7Bi%3D1%7D%5E%7Bp%7D%7C%5Cbeta_i%7C%0A)


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


## Analysis/Conclusion 

Below are the 10 fold cross-validated MAE for the models used to predict the S&P 500 daily returns as well as the optimal hyperparameters. 

| Model                          | MAE               | Optimal α | Optimal L1 Ratio |
|--------------------------------|-------------------|---------- |------------------|
| MAE Linear Model               | 0.5909%           |           |                  |              
| MAE Ridge Regression Model     | 0.5775%           | 0.011     |                  |
| MAE LASSO Model                | 0.5744%           | 0.000003  |                  |
| MAE Elastic Net Model          | 0.5750%           | 0.0000014 | 2                |          
| MAE Square Root LASSO          | 0.5728%           | 0.0004    |                  |

It seems that all of the regularization methods outperformed the baseline linear model when predicing S&P 500 daily returns. All the regularization techniques performed better at predicting coefficients that would result in predictions that produce lower MAE's because the regularization techniques were configured to take into account data with multicolinearity present. In particular, the Square Root Lasso regression performed the best with the lowest MAE of .5728%, and performed especially well because the Square Root LASSO accounts for multicolinearity and penalizes in a way that produces sparse solutions.  Clearly there is merit to using feature selection or regularization techniques as opposed to a simple linear model for predicting daily S&P returns. 

For further future analysis, I plan to expand on this project by also examining the performance of Kernel Weighted Regressions (Loess), stepwise regression, neural networks, XGBoost, SVR, Random Forests, as well as evaluate the potential use of polynomial features for polynomial regression. I wouild also like to evaluate why certain models perform better than others. 
