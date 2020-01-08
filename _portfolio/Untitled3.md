---
title: "Russian Ruble and Oil Prices; Machine Learning Model based on the OLS to predict the exchange rate."
excerpt: "First International Match and 42000 games later<br/><img src='/images/Elvira_boss.png'>"
collection: portfolio
---
<br/><img src='/images/Elvira_boss.png'>

```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
color = sns.color_palette()
from scipy import stats
import statsmodels.api as sm
from statsmodels.distributions.mixture_rvs import mixture_rvs
```

We will make a hypothesis that the currency rate
of ruble depends on world oil prices. We will construct the model
using MLR showing interrelation between the currency rate of
dollar to ruble and Brent brand oil prices during the period from
2010 to 2016, based on the daily, monthly, quortarely and yearly data. 


```python
macro = pd.read_csv("/Users/aldiyar/Desktop/python app1/russiahousing/sberbank-russian-housing-market/macro.csv")
```


```python
#Putin invades Ukraine to annex Crimea
macro['crimea'] = (macro['timestamp'] >= '2014-02-21')
```

This is an important dummy variable that separates the data into before and after an invasion of Crimea. Following the events, Russia dropped the pegged exchange rate control and international sanctions drastically restricted access to the international capital markets.   

Regression 1 on Daily values USD EUR BRENT URALS and controlled for follwing var's
oil_urals cpi ppi gdp_deflator balance_trade balance_trade_growth brent 
rts micex  micex_rgbi_tr micex_cbi_tr deposits_rate income_per_cap 
real_dispos_income_per_cap_growth
unemployment pop_migration


```python
# Exogenous var's
x =  macro[['usdrub','eurrub',  'oil_urals', 'cpi', 'ppi', 'gdp_deflator', 'balance_trade', 
'balance_trade_growth', 'brent', 'rts', 'micex',  'micex_rgbi_tr', 
'micex_cbi_tr', 'deposits_rate', 'income_per_cap',
'real_dispos_income_per_cap_growth',
'unemployment', 'crimea']]
x_after = x[x['crimea'] == True]
del x_after['crimea']
# Outcome var's
usdrub = x_after['usdrub']
eurrrub = x_after['eurrub']

```


```python
sns.lmplot(x="oil_urals", y="eurrub", hue="crimea", data=macro, markers=["o", "x"], palette="Set1")
```




    <seaborn.axisgrid.FacetGrid at 0x1a16ef0d10>





<br/><img src='/images/output_7_1.png'>

Case closed, a nearly perfect correlation between oil price and an exchange rate following the events in Crimea. On the mechanics of the money flow later in the conclusion. Also, I will create a variable of real exchange rate adjusting to the inflation to prevent endogeneity trap. 


```python
x_after = x[x['crimea'] == True]
del x_after['crimea']
macro_after = macro[macro['crimea'] == True]
del macro_after['crimea']
sns.jointplot(x="oil_urals", y="usdrub",  data=macro_after, kind="reg")
```




    <seaborn.axisgrid.JointGrid at 0x1a194705d0>





<br/><img src='/images/output_9_1.png'>

ML or MLR starts here. First, I run a regular regression with all selected variables. Then I will start backwards elimination. 


```python
#Missing Values 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x_after)
x_after = imputer.transform(x_after)
```


```python
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_after_train, x_after_test, usdrub_train, usdrub_test = train_test_split(x_after, usdrub, test_size = 0.2, random_state = 0)

```


```python
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_after_train, usdrub_train)

# Predicting the Test set results
y_pred = regressor.predict(x_after_test)
```


```python
diff = (y_pred - usdrub_test) 

_ = plt.hist(diff, bins= 5)  
plt.title("Histogram of difference between predicted and test values ")
plt.show()
```



<br/><img src='/images/output_14_0.png'>

Initial model seems to be alright, most of the values are 0. Next step is to check the p-values. 


```python
import statsmodels.regression.linear_model as sm
x_after = np.append(arr= np.ones((972, 1)).astype(int), values = x_after, axis = 1)


x_after_opt = x_after[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 ]]
regressor_OLS = sm.OLS(endog = usdrub,  exog = x_after_opt).fit()
regressor_OLS.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>usdrub</td>      <th>  R-squared:         </th>  <td>   1.000</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   1.000</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>6.914e+25</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 08 Jan 2020</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>01:42:14</td>     <th>  Log-Likelihood:    </th>  <td>  23027.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   972</td>      <th>  AIC:               </th> <td>-4.602e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   955</td>      <th>  BIC:               </th> <td>-4.594e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    16</td>      <th>                     </th>      <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>-2.434e-11</td> <td> 4.56e-11</td> <td>   -0.534</td> <td> 0.594</td> <td>-1.14e-10</td> <td> 6.51e-11</td>
</tr>
<tr>
  <th>x1</th>    <td>    1.0000</td> <td> 4.87e-13</td> <td> 2.05e+12</td> <td> 0.000</td> <td>    1.000</td> <td>    1.000</td>
</tr>
<tr>
  <th>x2</th>    <td>-1.113e-13</td> <td> 3.64e-13</td> <td>   -0.306</td> <td> 0.760</td> <td>-8.25e-13</td> <td> 6.03e-13</td>
</tr>
<tr>
  <th>x3</th>    <td>-1.166e-14</td> <td> 1.89e-13</td> <td>   -0.062</td> <td> 0.951</td> <td>-3.84e-13</td> <td>  3.6e-13</td>
</tr>
<tr>
  <th>x4</th>    <td>-8.873e-14</td> <td> 1.38e-13</td> <td>   -0.645</td> <td> 0.519</td> <td>-3.59e-13</td> <td> 1.81e-13</td>
</tr>
<tr>
  <th>x5</th>    <td> 2.638e-14</td> <td>    6e-14</td> <td>    0.439</td> <td> 0.660</td> <td>-9.14e-14</td> <td> 1.44e-13</td>
</tr>
<tr>
  <th>x6</th>    <td> 3.561e-13</td> <td> 4.25e-13</td> <td>    0.839</td> <td> 0.402</td> <td>-4.77e-13</td> <td> 1.19e-12</td>
</tr>
<tr>
  <th>x7</th>    <td>-6.023e-15</td> <td> 2.77e-13</td> <td>   -0.022</td> <td> 0.983</td> <td>-5.49e-13</td> <td> 5.37e-13</td>
</tr>
<tr>
  <th>x8</th>    <td>-1.396e-14</td> <td> 6.06e-14</td> <td>   -0.230</td> <td> 0.818</td> <td>-1.33e-13</td> <td> 1.05e-13</td>
</tr>
<tr>
  <th>x9</th>    <td> 1.343e-14</td> <td> 2.01e-13</td> <td>    0.067</td> <td> 0.947</td> <td>-3.81e-13</td> <td> 4.08e-13</td>
</tr>
<tr>
  <th>x10</th>   <td>-5.091e-16</td> <td> 1.66e-14</td> <td>   -0.031</td> <td> 0.976</td> <td>-3.32e-14</td> <td> 3.21e-14</td>
</tr>
<tr>
  <th>x11</th>   <td>  7.65e-16</td> <td> 1.07e-14</td> <td>    0.072</td> <td> 0.943</td> <td>-2.02e-14</td> <td> 2.17e-14</td>
</tr>
<tr>
  <th>x12</th>   <td>-5.473e-14</td> <td> 2.46e-13</td> <td>   -0.222</td> <td> 0.824</td> <td>-5.38e-13</td> <td> 4.28e-13</td>
</tr>
<tr>
  <th>x13</th>   <td> 4.649e-16</td> <td> 5.13e-14</td> <td>    0.009</td> <td> 0.993</td> <td>   -1e-13</td> <td> 1.01e-13</td>
</tr>
<tr>
  <th>x14</th>   <td>-1.568e-13</td> <td> 7.81e-13</td> <td>   -0.201</td> <td> 0.841</td> <td>-1.69e-12</td> <td> 1.38e-12</td>
</tr>
<tr>
  <th>x15</th>   <td> 8.412e-17</td> <td> 7.21e-17</td> <td>    1.167</td> <td> 0.244</td> <td>-5.74e-17</td> <td> 2.26e-16</td>
</tr>
<tr>
  <th>x16</th>   <td> 1.915e-12</td> <td>  3.6e-12</td> <td>    0.532</td> <td> 0.595</td> <td>-5.15e-12</td> <td> 8.98e-12</td>
</tr>
<tr>
  <th>x17</th>   <td>-5.684e-13</td> <td> 1.14e-09</td> <td>   -0.000</td> <td> 1.000</td> <td>-2.24e-09</td> <td> 2.24e-09</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>148.033</td> <th>  Durbin-Watson:     </th> <td>   0.000</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 234.745</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-1.006</td>  <th>  Prob(JB):          </th> <td>1.06e-51</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.322</td>  <th>  Cond. No.          </th> <td>6.44e+20</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 8.32e-30. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.



All of the p_values, but the Urals brand price are insignificant. Now, I will eliminate all the redundant values using the 10 per cent level. Amazingly, the R squared is perfect.


```python
#Automatic elimination of variables based on the 10 percent p-value.
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(usdrub, x_after_opt).fit()
        maxVar = max(regressor_OLS.pvalues)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.10
x_after_opt = x_after[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 ]]
X_Modeled = backwardElimination(x_after_opt, SL)
```


```python
regressor_OLS = sm.OLS(endog = usdrub,  exog = X_Modeled).fit()
regressor_OLS.summary()
```

 OLS Regression Results                            
==============================================================================
Dep. Variable:                 usdrub   R-squared:                       1.000
Model:                            OLS   Adj. R-squared:                  1.000
Method:                 Least Squares   F-statistic:                 6.229e+31
Date:                Wed, 08 Jan 2020   Prob (F-statistic):               0.00
Time:                        13:20:34   Log-Likelihood:                 28870.
No. Observations:                 972   AIC:                        -5.773e+04
Df Residuals:                     968   BIC:                        -5.771e+04
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.129e-14   1.09e-14      4.697      0.000    2.99e-14    7.27e-14
x1          6.001e-14   1.09e-14      5.495      0.000    3.86e-14    8.14e-14
x2           4.43e-14   1.09e-14      4.056      0.000    2.29e-14    6.57e-14
x3             1.0000   7.97e-16   1.25e+15      0.000       1.000       1.000
x4          1.665e-16    5.6e-16      0.298      0.766   -9.32e-16    1.26e-15
x5         -1.839e-16   2.19e-16     -0.841      0.401   -6.13e-16    2.45e-16
==============================================================================
Omnibus:                      406.072   Durbin-Watson:                   0.016
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               80.491
Skew:                          -0.436   Prob(JB):                     3.32e-18
Kurtosis:                       1.892   Cond. No.                     4.76e+19
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 4.85e-33. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
"""

As expected, all variables but the micex_rgbi_tr  are positive. I will come back to it later. The oil price constitutes the largest proportion of Russian imports. When oil companies are collecting the revenue in USD, they convert most of the proceeds to the Russian Ruble. Thus creating the largest demand. As oil prices fall, the demand for rubles falls, thus reducing the price of the Russian ruble. The model also controls for the deposit rates in RUB, which is a more accurate proxy for peoples behaviour. As RUB falls people rush to withdraws savings in Rub to buy USD, further escalating the devaluation. Although other positive variables are significant and positive, in their total magnitude they are insignificant. 

micex_rgbi_tr is a Moscow index for Russians Government Bonds rate. In theory, to stay competitive and bonds must offer higher rewards in the time of small demand for the. The mechanism of that effect is puzzling, but the variable is small in magnitude and insignificant. 

Overall the model offers very high degree of predictability. 


```python
diff = (y_pred - usdrub_test) 

_ = plt.hist(diff, bins= 5)  
plt.title("Histogram of difference between predicted and test values ")
plt.show()
```



<br/><img src='/images/output_22_0.png'>


```python

```
