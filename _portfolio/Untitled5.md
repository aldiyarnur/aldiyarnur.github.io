The Binomial and Poisson distributions are similar, but they are different. Also, the fact that they are both discrete does not mean that they are the same. 

The difference between the two is that while both measure the number of certain random events (or "successes") within a certain frame, the Binomial is based on discrete events, while the Poisson is based on continuous events. That is, with a binomial distribution you have a certain number, 𝑛, of "attempts," each of which has probability of success 𝑝. With a Poisson distribution, you essentially have infinite attempts, with infinitesimal chance of success. That is, given a Binomial distribution with some 𝑛,𝑝, if you let 𝑛→∞ and 𝑝→0 in such a way that 𝑛𝑝→𝜆, then that distribution approaches a Poisson distribution with parameter 𝜆.

Assuming each team has an equal possession of the ball ( Which is not true thank to Tiki Taka), there is a small probability p that the team is going to score in that window of possession. Also, let's assume that each team will posses a ball n times. At the same time, home teams on average have a higher possession rate, thus there is a castle advantage. 

Because of this limiting effect, Poisson distributions are used to model occurences of events that could happen a very large number of times, but happen rarely. As goals do. That is, they are used in situations that would be more properly represented by a Binomial distribution with a very large 𝑛 and small 𝑝, especially when the exact values of 𝑛 and 𝑝 are unknown. (Rare goals in the EPL).

Now,suppose we want to build a model to predict the outcomes of games from the English Premier League. 20 teams, all play each other twice during a season. Each team plays 38 matches, 380 games per season in total. An early attempt at building a statistical model is described by Maher in 1982, using a Poisson Distribuion. 

Maher, essentially, stated that there are simultaneously there 760 ( 380 (20\*19) by 2) games at the same time. One away game and one home game. So, he measured $\alpha$ a rate of goals scored by the home team as strength of the home attack. $\beta$ is the weakness of team j playing away, measured by conceded goals. $\xi$ is a measure of the team i playing at home, measured by conceded goals. And $\delta$ is a measure of team j playing away, measured by scored away goals.

However, lets assume there is a home factor k. If we multiply $\alpha$ by factor k and devide $\beta$ by k then:

 $$\sum_{i=1}^{} \alpha_i  = \sum_{j=1}^{} \beta_j  $$

  $$ \sum_{i=1}^{} \xi_i  = \sum_{j=1}^{} \delta_j $$

So, our probability mass fanctions is:

$$ P(x) = \frac{{e}^{-\lambda}{\lambda}^{x}}{x!}, \lambda > 0 $$


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson,skellam
from scipy import stats
```


```python
epl_1819 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1819/E0.csv")
epl= epl_1819.iloc[:, :23]


epl['SHHG'] = epl['FTHG'] - epl['HTHG'] 
epl['SHAT'] = epl['FTAG']-epl['HTAG']

epl['FTHG'].describe()
```




    count    380.000000
    mean       1.568421
    std        1.312836
    min        0.000000
    25%        1.000000
    50%        1.000000
    75%        2.000000
    max        6.000000
    Name: FTHG, dtype: float64



EPL Season 18-19, final table:
===


```python
epl['H'] = epl['HomeTeam']
epl['A'] = epl['AwayTeam']

vars_to_keep = ['HomeTeam','AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 
                'AS', 'HST', 'AST', 'HF','AF', 'HC', 'AC', 'HY', 'AY', 'HR', 
                'AR']

game_results = pd.melt(epl, id_vars = vars_to_keep, value_vars = ['H', 'A'],
                    var_name = 'Home/Away', value_name='Team')

game_results['Opponent'] = np.where(game_results['Team'] == game_results['HomeTeam'],
                                    game_results['AwayTeam'],
                                    game_results['HomeTeam'])

points_map = {
    'W': 3,
    'D': 1,
    'L': 0
}

def get_result(score, score_opp):
    if score == score_opp:
        return 'D'
    elif score > score_opp:
        return 'W'
    else:
        return 'L'



game_results['Goals'] = np.where(game_results['Team'] == game_results['HomeTeam'],
                                 game_results['FTHG'],
                                 game_results['FTAG'])

game_results['Goals_Opp'] = np.where(game_results['Team'] != game_results['HomeTeam'],
                                     game_results['FTHG'],
                                     game_results['FTAG'])

game_results['Result'] = np.vectorize(get_result)(game_results['Goals'], game_results['Goals_Opp'])

game_results['Home_Str'] = np.where(game_results['Team'] != game_results['HomeTeam'],
                                     game_results['HS'],
                                     game_results['AS'])
game_results['Home_Str_T'] = np.where(game_results['Team'] != game_results['HomeTeam'],
                                     game_results['HST'],
                                     game_results['AST'])

game_results['Home_Corer'] = np.where(game_results['Team'] != game_results['HomeTeam'],
                                     game_results['HC'],
                                     game_results['AC'])

game_results['Home_F'] = np.where(game_results['Team'] != game_results['HomeTeam'],
                                     game_results['HF'],
                                     game_results['AF'])
game_results['Points'] = game_results['Result'].map(points_map)

league_Home_Str = (game_results.groupby('Team').sum()['Home_Str'].sort_values(ascending=False))
league_Home_Str_T = (game_results.groupby('Team').sum()['Home_Str_T'].sort_values(ascending=False))
league_Home_Home_Corer = (game_results.groupby('Team').sum()['Home_Corer'].sort_values(ascending=False))
league_Home_Home_F = (game_results.groupby('Team').sum()['Home_F'].sort_values(ascending=False))

league = pd.DataFrame(league)
league_goals = pd.DataFrame(league_goals)
league_cons = pd.DataFrame(league_cons)
league_Home_Str = pd.DataFrame(league_Home_Str)
league_Home_Str_T = pd.DataFrame(league_Home_Str_T )
league_Home_Home_Corer = pd.DataFrame(league_Home_Home_Corer)
league_Home_Home_F = pd.DataFrame(league_Home_Home_F)
frames = [league, league_goals, league_cons, league_Home_Str, league_Home_Str_T, 
          league_Home_Home_Corer, league_Home_Home_F]

tab = pd.concat(frames, ignore_index=False, axis=1, sort=True)

tab['Goal_diff'] = tab['Goals'] - tab['Goals_Opp']
tab['Rank'] = tab['Points'].rank(ascending = False)

# A bit of a brute force
```


```python
tab # A table of Points, Goals Scored, Goals Conceded, Home Attacks, Home Corners, and Home Fouls
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
      <th>Points</th>
      <th>Goals</th>
      <th>Goals_Opp</th>
      <th>Home_Str</th>
      <th>Home_Str_T</th>
      <th>Home_Corer</th>
      <th>Home_F</th>
      <th>Goal_diff</th>
      <th>Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Arsenal</td>
      <td>70</td>
      <td>73</td>
      <td>51</td>
      <td>494</td>
      <td>183</td>
      <td>184</td>
      <td>469</td>
      <td>22</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>Bournemouth</td>
      <td>45</td>
      <td>56</td>
      <td>70</td>
      <td>521</td>
      <td>175</td>
      <td>225</td>
      <td>455</td>
      <td>-14</td>
      <td>13.5</td>
    </tr>
    <tr>
      <td>Brighton</td>
      <td>36</td>
      <td>35</td>
      <td>60</td>
      <td>582</td>
      <td>175</td>
      <td>216</td>
      <td>322</td>
      <td>-25</td>
      <td>17.0</td>
    </tr>
    <tr>
      <td>Burnley</td>
      <td>40</td>
      <td>45</td>
      <td>68</td>
      <td>650</td>
      <td>210</td>
      <td>245</td>
      <td>376</td>
      <td>-23</td>
      <td>15.0</td>
    </tr>
    <tr>
      <td>Cardiff</td>
      <td>34</td>
      <td>34</td>
      <td>69</td>
      <td>571</td>
      <td>212</td>
      <td>260</td>
      <td>415</td>
      <td>-35</td>
      <td>18.0</td>
    </tr>
    <tr>
      <td>Chelsea</td>
      <td>72</td>
      <td>63</td>
      <td>39</td>
      <td>347</td>
      <td>127</td>
      <td>141</td>
      <td>431</td>
      <td>24</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>Crystal Palace</td>
      <td>49</td>
      <td>51</td>
      <td>53</td>
      <td>525</td>
      <td>164</td>
      <td>216</td>
      <td>408</td>
      <td>-2</td>
      <td>12.0</td>
    </tr>
    <tr>
      <td>Everton</td>
      <td>54</td>
      <td>54</td>
      <td>46</td>
      <td>402</td>
      <td>137</td>
      <td>179</td>
      <td>440</td>
      <td>8</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>Fulham</td>
      <td>26</td>
      <td>34</td>
      <td>81</td>
      <td>583</td>
      <td>225</td>
      <td>226</td>
      <td>354</td>
      <td>-47</td>
      <td>19.0</td>
    </tr>
    <tr>
      <td>Huddersfield</td>
      <td>16</td>
      <td>22</td>
      <td>76</td>
      <td>520</td>
      <td>191</td>
      <td>170</td>
      <td>384</td>
      <td>-54</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>Leicester</td>
      <td>52</td>
      <td>51</td>
      <td>48</td>
      <td>426</td>
      <td>139</td>
      <td>202</td>
      <td>401</td>
      <td>3</td>
      <td>9.5</td>
    </tr>
    <tr>
      <td>Liverpool</td>
      <td>97</td>
      <td>89</td>
      <td>22</td>
      <td>307</td>
      <td>97</td>
      <td>126</td>
      <td>366</td>
      <td>67</td>
      <td>2.0</td>
    </tr>
    <tr>
      <td>Man City</td>
      <td>98</td>
      <td>95</td>
      <td>23</td>
      <td>236</td>
      <td>83</td>
      <td>82</td>
      <td>320</td>
      <td>72</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Man United</td>
      <td>66</td>
      <td>65</td>
      <td>54</td>
      <td>493</td>
      <td>173</td>
      <td>186</td>
      <td>409</td>
      <td>11</td>
      <td>6.0</td>
    </tr>
    <tr>
      <td>Newcastle</td>
      <td>45</td>
      <td>42</td>
      <td>48</td>
      <td>489</td>
      <td>148</td>
      <td>231</td>
      <td>349</td>
      <td>-6</td>
      <td>13.5</td>
    </tr>
    <tr>
      <td>Southampton</td>
      <td>39</td>
      <td>45</td>
      <td>65</td>
      <td>523</td>
      <td>179</td>
      <td>213</td>
      <td>351</td>
      <td>-20</td>
      <td>16.0</td>
    </tr>
    <tr>
      <td>Tottenham</td>
      <td>71</td>
      <td>67</td>
      <td>39</td>
      <td>463</td>
      <td>159</td>
      <td>187</td>
      <td>386</td>
      <td>28</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>Watford</td>
      <td>50</td>
      <td>52</td>
      <td>59</td>
      <td>490</td>
      <td>184</td>
      <td>210</td>
      <td>403</td>
      <td>-7</td>
      <td>11.0</td>
    </tr>
    <tr>
      <td>West Ham</td>
      <td>52</td>
      <td>52</td>
      <td>55</td>
      <td>528</td>
      <td>200</td>
      <td>209</td>
      <td>401</td>
      <td>-3</td>
      <td>9.5</td>
    </tr>
    <tr>
      <td>Wolves</td>
      <td>57</td>
      <td>47</td>
      <td>46</td>
      <td>456</td>
      <td>148</td>
      <td>190</td>
      <td>334</td>
      <td>1</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Mean ", np.mean(epl['FTHG']))
print(" Var ", np.var(epl['FTHG']))
```

    Mean  1.568421052631579
     Var  1.7190027700830994


However, the assumption is essentially relaxed as a model itself an approximation for the Binomial. 

Graphs
---

I will start with an obvious graph that predicts the rank team can achieve at the end of the season. The goal difference is proxy for a team's strength in attack and defence. The more one scores and less concedes the higher rank team will eventually achieve. 


```python

g = sns.jointplot("Goal_diff", "Rank", data=tab, kind = 'reg',
                  xlim=(-60, 80), ylim=(0, 20), color="m", height=10)
```


![png](output_17_0.png)



```python
d = sns.jointplot("Home_Str", "Rank", data=tab, kind = 'reg',
                  xlim=(230, 650), ylim=(0, 20), color="m", height=10)
# A number of attacks do slightly affect the rank position. 
```


![png](output_18_0.png)



```python
t = sns.jointplot("Home_Str", "Goals", data=tab, kind = 'reg',
                  xlim=(230, 650), ylim=(20, 100), color="m", height=10)
#This one is a completely unexpected result. The more team attacks, fewer goals they score. 
#The next graph will address the issue of quality of attacks.
```


![png](output_19_0.png)



```python
p = sns.jointplot("Home_Str_T", "Goals", data=tab, kind = 'reg',
                  xlim=(80, 230), ylim=(20, 100), color="m", height=10)
```


![png](output_20_0.png)


Fascinatingly, it encourages more research on goalkeepers quality and attack quality. There is no doubt that players such as Aguero are extremely clinical in front of the goals, but, to have on average less than 3 attempts on goal per game tell more about energy conservation mode of top teams, that have to play twice a week. That is why it is clever to buy tickets for games with lower band teams, there is more action to witness.   


```python
p = sns.jointplot("Home_Corer", "Goals", data=tab, kind = 'reg',
                  xlim=(80, 230), ylim=(20, 100), color="m", height=10)
#Total Corners
#Also, tells a lot about quality of teams
```


![png](output_22_0.png)



```python
p = sns.jointplot("Home_F", "Goals", data=tab, kind = 'reg',
                  xlim=(300, 500), ylim=(20, 100), color="m", height=10)
#This graph tells there is no correlation between faults and goals. However, top-left observation is 
#Man City, which has the highest possession of the ball, thus less opportunity to commit faults.
```


![png](output_23_0.png)



```python
means_g = epl[['FTHG','FTAG']].mean() # For the Model


alpha = epl.filter(["HomeTeam", 'FTHG','HTHG', 'HS', 'HST', 'SHHG'], axis=1)
beta = epl.filter(["AwayTeam", 'FTAG','HTAG', 'AS', 'AST', 'SHAT'], axis=1)

```


```python

```


```python
plt.figure(figsize=(15,10))
sns.set_style("white")
# construct Poisson  for each mean goals value
poisson_pred = np.column_stack([[poisson.pmf(k, means[j]) for k in range(10)] for j in range(2)])
# plot histogram of actual goals
plt.hist(epl[['FTHG', 'FTAG']].values, range(11), alpha=0.8,
         label=['Home', 'Away'],normed=True, color=["#3498db", "#e74c3c"])

# add lines for the Poisson distributions
pois1, = plt.plot([i-0.5 for i in range(1,11)], poisson_pred[:,0],
                  linestyle='-', marker='o',label="Home", color = '#2980b9')
pois2, = plt.plot([i-0.5 for i in range(1,11)], poisson_pred[:,1],
                  linestyle='-', marker='o',label="Away", color = '#c0392b')

leg=plt.legend(loc='upper right', fontsize=16, ncol=2)
leg.set_title("Poisson        Actual      ", prop = {'size':'18', 'weight':'bold'})

plt.xticks([i-0.5 for i in range(1,11)],[i for i in range(11)])
plt.xlabel("Goals per Match",size=18)
plt.ylabel("Proportion of Matches",size=18)
plt.title("Number of Goals per Match",size=20,fontweight='bold')
plt.show()

```

    /Users/aldiyar/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:7: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      import sys



![png](output_26_1.png)


It looks like data visually fits the Poisson Distribution.


```python
means_s = epl[['HS','AS']].mean()
```


```python
plt.figure(figsize=(15,10))
sns.set_style("white")

# plot histogram of actual goals
plt.hist(epl[['HS', 'AS']].values, range(30), alpha=0.8,
         label=['Home', 'Away'],normed=True, color=["#3498db", "#e74c3c"])

leg=plt.legend(loc='upper right', fontsize=16, ncol=2)
leg.set_title("Color Code     ", prop = {'size':'18', 'weight':'bold'})

plt.xticks([i-0.5 for i in range(1,30)],[i for i in range(30)])
plt.xlabel("Strikes per Match",size=18)
plt.ylabel("Proportion of Matches",size=18)
plt.title("Number of Strikes per Match",size=20,fontweight='bold')
plt.show()
```

    /Users/aldiyar/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:6: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      



![png](output_29_1.png)


Looks like a number of attempts or strikes on goals is normally distributed. The home team has a castle advantage, thus it has a higher number of attempts on goals. Thus the right tail is fatter for a home team rather than a home team.


```python
means_s_t = epl[['HST','AST']].mean()
```


```python
plt.figure(figsize=(15,10))
sns.set_style("white")

# plot histogram of actual goals
plt.hist(epl[['HST', 'AST']].values, range(15), alpha=0.8,
         label=['Home', 'Away'],normed=True, color=["#3498db", "#e74c3c"])

leg=plt.legend(loc='upper right', fontsize=16, ncol=2)
leg.set_title("Color Code     ", prop = {'size':'18', 'weight':'bold'})

plt.xticks([i-0.5 for i in range(1,15)],[i for i in range(15)])
plt.xlabel("Strikes on Target per Match",size=18)
plt.ylabel("Proportion of Matches",size=18)
plt.title("Number of Strikes on Target per Match",size=20,fontweight='bold')
plt.show()
```

    /Users/aldiyar/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:6: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      



![png](output_32_1.png)


Number of strikes on target looks like a Poisson distribution with a lambda 3 or 4. There is a light skewness present. 


```python
means_red= epl[['HF','AF']].mean()
```


```python
plt.figure(figsize=(15,10))
sns.set_style("white")

# plot histogram of actual goals
plt.hist(epl[['HF', 'AF']].values, range(25), alpha=0.8,
         label=['Home', 'Away'],normed=True, color=["#3498db", "#e74c3c"])

leg=plt.legend(loc='upper right', fontsize=16, ncol=2)
leg.set_title("Color Code     ", prop = {'size':'18', 'weight':'bold'})

plt.xticks([i-0.5 for i in range(1,25)],[i for i in range(25)])
plt.xlabel("Fouls per Match",size=18)
plt.ylabel("Proportion of Matches",size=18)
plt.title("Number Fouls per Match",size=20,fontweight='bold')
plt.show()
```

    /Users/aldiyar/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:6: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      



![png](output_35_1.png)


Fouls per game look very normally distributed. Now, lets go back to goals


```python
plt.figure(figsize=(15,10))
sns.set_style("white")
# construct Poisson  for each mean goals value
poisson_pred = np.column_stack([[poisson.pmf(k, means[j]) for k in range(10)] for j in range(2)])
# plot histogram of actual goals
plt.hist(epl[['FTHG', 'FTAG']].values, range(11), alpha=0.8,
         label=['Home', 'Away'],normed=True, color=["#3498db", "#e74c3c"])

# add lines for the Poisson distributions
pois1, = plt.plot([i-0.5 for i in range(1,11)], poisson_pred[:,0],
                  linestyle='-', marker='o',label="Home", color = '#2980b9')
pois2, = plt.plot([i-0.5 for i in range(1,11)], poisson_pred[:,1],
                  linestyle='-', marker='o',label="Away", color = '#c0392b')

leg=plt.legend(loc='upper right', fontsize=16, ncol=2)
leg.set_title("Poisson        Actual      ", prop = {'size':'18', 'weight':'bold'})

plt.xticks([i-0.5 for i in range(1,11)],[i for i in range(11)])
plt.xlabel("Goals per Match",size=18)
plt.ylabel("Proportion of Matches",size=18)
plt.title("Number of Goals per Match",size=20,fontweight='bold')
plt.show()
```

    /Users/aldiyar/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:7: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      import sys



![png](output_37_1.png)



```python
print("Mean ", np.mean(epl['FTHG']))
print(" Var ", np.var(epl['FTHG']))
```

    Mean  1.568421052631579
     Var  1.7190027700830994


With a mean value almost the same as a variance. I will go ahead with a predictive model. I will try to predict results of a last 38 round of EPL.

Predictions 
===


```python
epl2=epl.iloc[:370, :] #Excluded the last round of the EPL
```


```python
print("Mean ", np.mean(epl['FTHG']))
print(" Var ", np.var(epl['FTHG']))
```

    Mean  1.568421052631579
     Var  1.7190027700830994



```python

```
