# Predicting S&P 500 Daily Returns

The goal of this project is to evaluate the performance of different regressors when predicting the daily S&P 500 returns for a day using the daily returns of other international stock market indices that day as well as the prior day. The marktet indices used as features for predicting S&P 500 daily return are the Istanbul Stock market return index, the Istanbul stock exchange national 100 index, the stock market return index of Germany, stock market return index of UK, stock market return index of Japan, stock market return index of Brazil, MSCI European index, and MSCI emerging markets index. The performance of models examined in this project will be evaluated by comparing the 10 split k-fold cross-validated mean absolute error for each of the models. For our preliminary analysis, we will be looking at a simple baseline linear regression model and regularization methods such as LASSO regression, RIDGE regression, Elastic Net regression, and SQRT LASSO regression. 

## Background

For the preliminary analysis, we will mainly focusing on the evalutaing performance of regularized regressions versus a baseline linear model in the situation of predicting daily S&P 500 stock market index return. Regularization methods are used to determine the weights for features within a model, and depending on the regularization technique, features can be excluded altogether from the model by having a weight of 0. Further, regularization is the process of of regularizing the parameters that constrains or coefficients estimates towards zeros, in otherwords discouraging learning a too complex or too flexible model, and ultimately helping to reduce the risk of overfitting. Regularization helps to choose the preferred model complexity, so that the model is better at predicting. Regularization is essentially adding a penalty term to the objective function, and controlling the model complexity using that penalty term. Regularization ultmately attempts to reduce the variance of the estimator by simplifying it, something that will increase the bias, in such a way that will decrease the expected error of the model's predictions. Additionally, Regularization is useful in tackling issues of multicolinearity among features becauase it incorporates the need to learn additional information for our model other than the observations in the data set.
  
 Regularization techniques are especially useful in situations where there are large number of features or a low number of observations to number of features, where there is multicolinearity between the features, trying to seek a sparse solution, or accounting for for variable groupings in high dimensions. Regularization ultimately seeks to tackle the shortcomings of Ordinary Least Square models. 
  
### Ridge 
L2 regularization, or commonly known as ridge regularization, is a type of regularization that controls for the sizes of each coefficient or estimators. Not only does ridge regularization encorporate principles of OLS by reducing the sum of squared residuals, it also penalizes models with the regularization term of L2 Norm or
![\alpha \sum_{i=1}^p \beta_i^2](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Calpha+%5Csum_%7Bi%3D1%7D%5Ep+%5Cbeta_i%5E2).

Ridge regularization is especially useful when there is multicolinearity within data, and further, ridge regularization seeks to ultimately minimze the cost function:

![\sum_{i=1}^N (y_i - \hat{y}_i)^2 + \alpha \sum_{i=1}^p |\beta_i^2| ](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%21%5B%5Csum_%7Bi%3D1%7D%5EN+%28y_i+-+%5Chat%7By%7D_i%29%5E2+%2B+%5Calpha+%5Csum_%7Bi%3D1%7D%5Ep+%7C%5Cbeta_i%5E2%7C+)

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



