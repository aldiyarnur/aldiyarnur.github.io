---
title: "International Football Results"
excerpt: "First International Match and 42000 games later<br/><img src='/images/England_v_scotland_1872_ad.png'>"
collection: portfolio
---

<br/><img src='/images/England_v_scotland_1872_ad.png'>

Hello Football fans,

This is one of my first ever kernels on this website. 

Firstly, upload the most fundamental libraries for Python. However, as we go there might be a need for an additional libraries. 

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import poisson,skellam
```
Read the csv file and see first and last 5 observations. 

```python
results = pd.read_csv('results.csv')
results.head(5)
```
| NaN | date       | home_team | away_team | home_score | away_score | tournament | city    | country  | neutral |
|-----|------------|-----------|-----------|------------|------------|------------|---------|----------|---------|
| 0   | 1872-11-30 | Scotland  | England   | 0          | 0          | Friendly   | Glasgow | Scotland | False   |
| 1   | 1873-03-08 | England   | Scotland  | 4          | 2          | Friendly   | London  | England  | False   |
| 2   | 1874-03-07 | Scotland  | England   | 2          | 1          | Friendly   | Glasgow | Scotland | False   |
| 3   | 1875-03-06 | England   | Scotland  | 2          | 2          | Friendly   | London  | England  | False   |
| 4   | 1876-03-04 | Scotland  | England   | 3          | 0          | Friendly   | Glasgow | Scotland | False   |


```python
# Date adjstments 
results['date'] = pd.to_datetime(results['date'])
results['date1_year'] = results['date'].dt.year
results['date2_month'] = results['date'].dt.month
results['date3_day'] = results['date'].dt.day
# Total score will used to compute average score by game
results['total_score'] = results.home_score + results.away_score  
results['diff_score'] = abs(results.home_score - results.away_score)
```
