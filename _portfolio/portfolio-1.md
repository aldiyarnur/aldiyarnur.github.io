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
Now I upload the data and generate new calendar and score variables that will be useful later.

```python
results = pd.read_csv('results.csv')

"""DATE ADJUSTMENT FOR RESULTS"""
results['date'] = pd.to_datetime(results['date'])
results['date1_year'] = results['date'].dt.year
results['date2_month'] = results['date'].dt.month
results['date3_day'] = results['date'].dt.day
# Total score will used to compute average score by game
results['total_score'] = results.home_score + results.away_score  
results['diff_score'] = abs(results.home_score - results.away_score)
results.head(5)
```
