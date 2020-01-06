---
title: "International Football Results part II"
excerpt: "First International Match and 42000 games later<br/><img src='/images/England_v_scotland_1872_ad.png'>"
collection: portfolio
---

<br/><img src='/images/England_v_scotland_1872_ad.png'>




```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson,skellam
results = pd.read_csv('/Users/aldiyar/Desktop/python app1/ml/Untitled Folder/results.csv')

# New date variables
results['date'] = pd.to_datetime(results['date'])
results['date1_year'] = results['date'].dt.year
results['date2_month'] = results['date'].dt.month
results['date3_day'] = results['date'].dt.day
# Total score is used to compute average score by game
results['total_score'] = results.home_score + results.away_score  
results['diff_score'] = abs(results.home_score - results.away_score)

results.head(5)
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
      <th>date</th>
      <th>home_team</th>
      <th>away_team</th>
      <th>home_score</th>
      <th>away_score</th>
      <th>tournament</th>
      <th>city</th>
      <th>country</th>
      <th>neutral</th>
      <th>date1_year</th>
      <th>date2_month</th>
      <th>date3_day</th>
      <th>total_score</th>
      <th>diff_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1872-11-30</td>
      <td>Scotland</td>
      <td>England</td>
      <td>0</td>
      <td>0</td>
      <td>Friendly</td>
      <td>Glasgow</td>
      <td>Scotland</td>
      <td>False</td>
      <td>1872</td>
      <td>11</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1873-03-08</td>
      <td>England</td>
      <td>Scotland</td>
      <td>4</td>
      <td>2</td>
      <td>Friendly</td>
      <td>London</td>
      <td>England</td>
      <td>False</td>
      <td>1873</td>
      <td>3</td>
      <td>8</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1874-03-07</td>
      <td>Scotland</td>
      <td>England</td>
      <td>2</td>
      <td>1</td>
      <td>Friendly</td>
      <td>Glasgow</td>
      <td>Scotland</td>
      <td>False</td>
      <td>1874</td>
      <td>3</td>
      <td>7</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1875-03-06</td>
      <td>England</td>
      <td>Scotland</td>
      <td>2</td>
      <td>2</td>
      <td>Friendly</td>
      <td>London</td>
      <td>England</td>
      <td>False</td>
      <td>1875</td>
      <td>3</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1876-03-04</td>
      <td>Scotland</td>
      <td>England</td>
      <td>3</td>
      <td>0</td>
      <td>Friendly</td>
      <td>Glasgow</td>
      <td>Scotland</td>
      <td>False</td>
      <td>1876</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
allteam=results['home_team'].unique()
allteam
```




    array(['Scotland', 'England', 'Wales', 'Northern Ireland',
           'United States', 'Uruguay', 'Austria', 'Hungary', 'Argentina',
           'Belgium', 'France', 'Netherlands', 'Czechoslovakia',
           'Switzerland', 'Sweden', 'Germany', 'Italy', 'Chile', 'Norway',
           'Finland', 'Luxembourg', 'Russia', 'Denmark', 'Catalonia',
           'Basque Country', 'Brazil', 'Japan', 'Paraguay', 'Canada',
           'Estonia', 'Costa Rica', 'Guatemala', 'Spain', 'Brittany',
           'Poland', 'Yugoslavia', 'New Zealand', 'Romania', 'Latvia',
           'Galicia', 'Portugal', 'Andalusia', 'China PR', 'Australia',
           'Lithuania', 'Turkey', 'Central Spain', 'Mexico', 'Aruba', 'Egypt',
           'Haiti', 'Philippines', 'Bulgaria', 'Jamaica', 'Kenya', 'Bolivia',
           'Peru', 'Honduras', 'Guyana', 'Uganda', 'Belarus', 'El Salvador',
           'Barbados', 'Republic of Ireland', 'Trinidad and Tobago', 'Greece',
           'Curaçao', 'Dominica', 'Silesia', 'Guadeloupe', 'Israel',
           'Suriname', 'French Guiana', 'Cuba', 'Colombia', 'Ecuador',
           'Saint Kitts and Nevis', 'Panama', 'Slovakia', 'Manchukuo',
           'Croatia', 'Nicaragua', 'Afghanistan', 'India', 'Martinique',
           'Zimbabwe', 'Iceland', 'Albania', 'Madagascar', 'Zambia',
           'Mauritius', 'Tanzania', 'Iran', 'Djibouti', 'DR Congo', 'Vietnam',
           'Macau', 'Ethiopia', 'Puerto Rico', 'Réunion', 'Sierra Leone',
           'Zanzibar', 'South Korea', 'Ghana', 'South Africa',
           'New Caledonia', 'Fiji', 'Nigeria', 'Venezuela', 'Burma',
           'Sri Lanka', 'Tahiti', 'Gambia', 'Hong Kong', 'Singapore',
           'Malaysia', 'Indonesia', 'Guinea-Bissau', 'German DR', 'Vanuatu',
           'Kernow', 'Saarland', 'Cambodia', 'Lebanon', 'Pakistan',
           'Vietnam Republic', 'North Korea', 'Togo', 'Sudan', 'Malta',
           'Syria', 'Tunisia', 'Malawi', 'Morocco', 'Benin', 'Cameroon',
           'Central African Republic', 'Gabon', 'Ivory Coast', 'Congo',
           'Mali', 'North Vietnam', 'Mongolia', 'Chinese Taipei', 'Cyprus',
           'Iraq', 'Saint Lucia', 'Grenada', 'Thailand', 'Senegal', 'Libya',
           'Guinea', 'Algeria', 'Kuwait', 'Jordan', 'Solomon Islands',
           'Liberia', 'Laos', 'Saint Vincent and the Grenadines', 'Bermuda',
           'Niger', 'Bahrain', 'Montenegro', 'Palestine', 'Papua New Guinea',
           'Burkina Faso', 'Mauritania', 'Saudi Arabia', 'Eswatini',
           'Somalia', 'Lesotho', 'Cook Islands', 'Qatar',
           'Antigua and Barbuda', 'Faroe Islands', 'Bangladesh', 'Oman',
           'Yemen DPR', 'Burundi', 'Yemen', 'Mozambique', 'Guam', 'Chad',
           'Angola', 'Dominican Republic', 'Seychelles', 'Rwanda',
           'São Tomé and Príncipe', 'Botswana', 'Northern Cyprus',
           'Cape Verde', 'Kyrgyzstan', 'Georgia', 'Azerbaijan', 'Kiribati',
           'Tonga', 'Wallis Islands and Futuna', 'United Arab Emirates',
           'Brunei', 'Equatorial Guinea', 'Liechtenstein', 'Nepal',
           'Greenland', 'Niue', 'Samoa', 'American Samoa', 'Belize',
           'Anguilla', 'Cayman Islands', 'Palau', 'Sint Maarten', 'Namibia',
           'Åland Islands', 'Ynys Môn', 'Saint Martin', 'San Marino',
           'Slovenia', 'Jersey', 'Shetland', 'Isle of Wight', 'Moldova',
           'Ukraine', 'Kazakhstan', 'Tajikistan', 'Uzbekistan',
           'Turkmenistan', 'Armenia', 'Czech Republic', 'Guernsey',
           'Gibraltar', 'Isle of Man', 'North Macedonia', 'Montserrat',
           'Serbia', 'Canary Islands', 'Bosnia and Herzegovina', 'Maldives',
           'Andorra', 'British Virgin Islands', 'Frøya', 'Hitra',
           'United States Virgin Islands', 'Corsica', 'Eritrea', 'Bahamas',
           'Gotland', 'Saare County', 'Rhodes', 'Micronesia', 'Bhutan',
           'Orkney', 'Monaco', 'Tuvalu', 'Sark', 'Alderney', 'Mayotte',
           'Turks and Caicos Islands', 'East Timor', 'Western Isles',
           'Falkland Islands', 'Kosovo', 'Republic of St. Pauli', 'Găgăuzia',
           'Tibet', 'Crimea', 'Occitania', 'Sápmi',
           'Northern Mariana Islands', 'Menorca', 'Comoros', 'Provence',
           'Arameans Suryoye', 'Padania', 'Iraqi Kurdistan', 'Gozo',
           'Bonaire', 'Western Sahara', 'Raetia', 'Darfur', 'Tamil Eelam',
           'South Sudan', 'Abkhazia', 'St. Pierre & Miquelon', 'Artsakh',
           'Madrid', 'Vatican City', 'Ellan Vannin', 'South Ossetia',
           'County of Nice', 'Székely Land', 'Romani people', 'Felvidék',
           'Chagos Islands', 'United Koreans in Japan', 'Somaliland',
           'Western Armenia', 'Barawa', 'Kárpátalja', 'Yorkshire', 'Panjab',
           'Matabeleland', 'Cascadia', 'Kabylia', 'Timor-Leste', 'Curacao',
           'Myanmar', 'Parishes of Jersey', 'Chameria', 'Saint Helena',
           'St Vincent and the Grenadines'], dtype=object)




```python
allteam=results['home_team'].unique()
allteam

scored_home=[]
for team in allteam:
    goals = sum(results[results['home_team']==team].away_score)
    scored_home.append(goals)
    

scored_away=[]    
for team in allteam:
    goals2=sum(results[results['away_team']==team].home_score)
    scored_away.append(goals2)

total = pd.DataFrame({'team' : allteam, 'home_g' : scored_home, 'away_g':scored_away })
total['tgs'] = total['home_g']+total['away_g'] 
tot = total.sort_values(by=['tgs'], ascending=False)




```


```python
tot = total.sort_values(by=['tgs'], ascending=False)
tot.tail(60)
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
      <th>team</th>
      <th>home_g</th>
      <th>away_g</th>
      <th>tgs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>263</td>
      <td>Tibet</td>
      <td>13</td>
      <td>42</td>
      <td>55</td>
    </tr>
    <tr>
      <td>260</td>
      <td>Kosovo</td>
      <td>35</td>
      <td>18</td>
      <td>53</td>
    </tr>
    <tr>
      <td>141</td>
      <td>North Vietnam</td>
      <td>9</td>
      <td>43</td>
      <td>52</td>
    </tr>
    <tr>
      <td>251</td>
      <td>Monaco</td>
      <td>2</td>
      <td>50</td>
      <td>52</td>
    </tr>
    <tr>
      <td>270</td>
      <td>Provence</td>
      <td>29</td>
      <td>22</td>
      <td>51</td>
    </tr>
    <tr>
      <td>289</td>
      <td>Székely Land</td>
      <td>17</td>
      <td>34</td>
      <td>51</td>
    </tr>
    <tr>
      <td>279</td>
      <td>Tamil Eelam</td>
      <td>23</td>
      <td>27</td>
      <td>50</td>
    </tr>
    <tr>
      <td>266</td>
      <td>Sápmi</td>
      <td>27</td>
      <td>22</td>
      <td>49</td>
    </tr>
    <tr>
      <td>296</td>
      <td>Barawa</td>
      <td>41</td>
      <td>5</td>
      <td>46</td>
    </tr>
    <tr>
      <td>255</td>
      <td>Mayotte</td>
      <td>15</td>
      <td>22</td>
      <td>37</td>
    </tr>
    <tr>
      <td>277</td>
      <td>Raetia</td>
      <td>6</td>
      <td>30</td>
      <td>36</td>
    </tr>
    <tr>
      <td>203</td>
      <td>Niue</td>
      <td>33</td>
      <td>0</td>
      <td>33</td>
    </tr>
    <tr>
      <td>272</td>
      <td>Padania</td>
      <td>18</td>
      <td>15</td>
      <td>33</td>
    </tr>
    <tr>
      <td>189</td>
      <td>Northern Cyprus</td>
      <td>13</td>
      <td>18</td>
      <td>31</td>
    </tr>
    <tr>
      <td>287</td>
      <td>South Ossetia</td>
      <td>16</td>
      <td>14</td>
      <td>30</td>
    </tr>
    <tr>
      <td>273</td>
      <td>Iraqi Kurdistan</td>
      <td>11</td>
      <td>16</td>
      <td>27</td>
    </tr>
    <tr>
      <td>121</td>
      <td>Saarland</td>
      <td>18</td>
      <td>8</td>
      <td>26</td>
    </tr>
    <tr>
      <td>265</td>
      <td>Occitania</td>
      <td>19</td>
      <td>7</td>
      <td>26</td>
    </tr>
    <tr>
      <td>268</td>
      <td>Menorca</td>
      <td>7</td>
      <td>18</td>
      <td>25</td>
    </tr>
    <tr>
      <td>286</td>
      <td>Ellan Vannin</td>
      <td>11</td>
      <td>13</td>
      <td>24</td>
    </tr>
    <tr>
      <td>308</td>
      <td>Saint Helena</td>
      <td>6</td>
      <td>18</td>
      <td>24</td>
    </tr>
    <tr>
      <td>281</td>
      <td>Abkhazia</td>
      <td>11</td>
      <td>13</td>
      <td>24</td>
    </tr>
    <tr>
      <td>303</td>
      <td>Timor-Leste</td>
      <td>11</td>
      <td>12</td>
      <td>23</td>
    </tr>
    <tr>
      <td>305</td>
      <td>Myanmar</td>
      <td>6</td>
      <td>16</td>
      <td>22</td>
    </tr>
    <tr>
      <td>209</td>
      <td>Palau</td>
      <td>21</td>
      <td>0</td>
      <td>21</td>
    </tr>
    <tr>
      <td>295</td>
      <td>Western Armenia</td>
      <td>10</td>
      <td>10</td>
      <td>20</td>
    </tr>
    <tr>
      <td>274</td>
      <td>Gozo</td>
      <td>8</td>
      <td>12</td>
      <td>20</td>
    </tr>
    <tr>
      <td>247</td>
      <td>Rhodes</td>
      <td>12</td>
      <td>7</td>
      <td>19</td>
    </tr>
    <tr>
      <td>120</td>
      <td>Kernow</td>
      <td>16</td>
      <td>2</td>
      <td>18</td>
    </tr>
    <tr>
      <td>294</td>
      <td>Somaliland</td>
      <td>5</td>
      <td>13</td>
      <td>18</td>
    </tr>
    <tr>
      <td>68</td>
      <td>Silesia</td>
      <td>17</td>
      <td>0</td>
      <td>17</td>
    </tr>
    <tr>
      <td>276</td>
      <td>Western Sahara</td>
      <td>7</td>
      <td>9</td>
      <td>16</td>
    </tr>
    <tr>
      <td>79</td>
      <td>Manchukuo</td>
      <td>9</td>
      <td>7</td>
      <td>16</td>
    </tr>
    <tr>
      <td>299</td>
      <td>Panjab</td>
      <td>5</td>
      <td>11</td>
      <td>16</td>
    </tr>
    <tr>
      <td>33</td>
      <td>Brittany</td>
      <td>11</td>
      <td>4</td>
      <td>15</td>
    </tr>
    <tr>
      <td>302</td>
      <td>Kabylia</td>
      <td>2</td>
      <td>13</td>
      <td>15</td>
    </tr>
    <tr>
      <td>285</td>
      <td>Vatican City</td>
      <td>2</td>
      <td>13</td>
      <td>15</td>
    </tr>
    <tr>
      <td>291</td>
      <td>Felvidék</td>
      <td>4</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <td>301</td>
      <td>Cascadia</td>
      <td>3</td>
      <td>11</td>
      <td>14</td>
    </tr>
    <tr>
      <td>41</td>
      <td>Andalusia</td>
      <td>13</td>
      <td>1</td>
      <td>14</td>
    </tr>
    <tr>
      <td>283</td>
      <td>Artsakh</td>
      <td>9</td>
      <td>5</td>
      <td>14</td>
    </tr>
    <tr>
      <td>288</td>
      <td>County of Nice</td>
      <td>5</td>
      <td>8</td>
      <td>13</td>
    </tr>
    <tr>
      <td>39</td>
      <td>Galicia</td>
      <td>11</td>
      <td>2</td>
      <td>13</td>
    </tr>
    <tr>
      <td>300</td>
      <td>Matabeleland</td>
      <td>0</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <td>271</td>
      <td>Arameans Suryoye</td>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <td>297</td>
      <td>Kárpátalja</td>
      <td>9</td>
      <td>3</td>
      <td>12</td>
    </tr>
    <tr>
      <td>298</td>
      <td>Yorkshire</td>
      <td>7</td>
      <td>4</td>
      <td>11</td>
    </tr>
    <tr>
      <td>293</td>
      <td>United Koreans in Japan</td>
      <td>7</td>
      <td>4</td>
      <td>11</td>
    </tr>
    <tr>
      <td>264</td>
      <td>Crimea</td>
      <td>5</td>
      <td>6</td>
      <td>11</td>
    </tr>
    <tr>
      <td>262</td>
      <td>Găgăuzia</td>
      <td>9</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <td>290</td>
      <td>Romani people</td>
      <td>3</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <td>304</td>
      <td>Curacao</td>
      <td>1</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <td>261</td>
      <td>Republic of St. Pauli</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <td>242</td>
      <td>Corsica</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <td>46</td>
      <td>Central Spain</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <td>307</td>
      <td>Chameria</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <td>306</td>
      <td>Parishes of Jersey</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <td>234</td>
      <td>Canary Islands</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>284</td>
      <td>Madrid</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>309</td>
      <td>St Vincent and the Grenadines</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



By calculating total goals scored by each team in the database of international games and listing teams who scored the least, we dive into the world where international geopolitics is merging with a worlds sport number one. Not only we see teams that break international new headlines such as Iraqi Kurdistan national team, Crimea, Donetsk People Republic, but also an ethnic group of people such as Roma people, Galicia people to name a few. There also teams that existed in the short span of time such a team North Vietnam or Central Spain National team. There is, however, one tiny and a mighty unrecognised team Spain, a Catalonia. Not only it can rival Spain National Team itself, but Catalan parliament is actively nourishing the team infrastructure in the case of the break away from Spain.  


```python
countries=[]

for c1, c2 in zip(results['home_team'],results['away_team']):
    if c1 not in countries :
        countries.append(c1)
    if c2 not in countries :
        countries.append(c2)
        

```


```python
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=0 # chosen at random by flipping a coin; it was heads
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
    
#paste any value here 
show_wordcloud(countries)

```

<br/><img src='/images/output_6_0.png'>



Using the beautiful Word Cloud, which is installed via pip install, we can the snapshot of the countries. Given the time span of the dataset, there is Czechoslovakia. There is no more Czech Republic, let alone Czechoslovakia(the Czech Republic is finally renamed Czechia).


```python
matches=[]
for c in countries:
    nb=0
    for c1, c2 in zip(results['home_team'],results['away_team']):
            if c1==c or c2==c:
                nb=nb+1
    matches.append(nb)
        
victory=[]
defeat=[]
draw=[]
pourcent_vic=[]
for c in countries:
    nb_v=nb_d=nb_n=0
    for c1 ,(j ,c2) in zip(results['home_team'],enumerate(results['away_team'])):
        if c1 == c:
            if results['home_score'][j]>results['away_score'][j]:
                nb_v=nb_v+1
            elif results['home_score'][j]<results['away_score'][j]:
                nb_d=nb_d+1
            else:
                nb_n=nb_n+1
        elif c2 == c:
            if results['away_score'][j]>results['home_score'][j]:
                nb_v=nb_v+1
            elif results['away_score'][j]<results['home_score'][j]:
                nb_d=nb_d+1
            else:
                nb_n=nb_n+1
    victory.append(nb_v)
    defeat.append(nb_d)
    draw.append(nb_n)

statistics=pd.DataFrame(list(zip(countries,matches,victory,defeat,draw)),columns=['country','nb_match','nb_vict',
                                                                                     'nb_defts','nb_draws'])
statistics['%vic']=round((statistics['nb_vict']/statistics['nb_match'])*100)

statistics.head()
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
      <th>country</th>
      <th>nb_match</th>
      <th>nb_vict</th>
      <th>nb_defts</th>
      <th>nb_draws</th>
      <th>%vic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Scotland</td>
      <td>775</td>
      <td>367</td>
      <td>243</td>
      <td>165</td>
      <td>47.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>England</td>
      <td>998</td>
      <td>566</td>
      <td>191</td>
      <td>241</td>
      <td>57.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Wales</td>
      <td>652</td>
      <td>204</td>
      <td>307</td>
      <td>141</td>
      <td>31.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Northern Ireland</td>
      <td>640</td>
      <td>165</td>
      <td>329</td>
      <td>146</td>
      <td>26.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>United States</td>
      <td>684</td>
      <td>291</td>
      <td>249</td>
      <td>144</td>
      <td>43.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
statistics.sort_values(by=['nb_match'], ascending=False).head(50)
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
      <th>country</th>
      <th>nb_match</th>
      <th>nb_vict</th>
      <th>nb_defts</th>
      <th>nb_draws</th>
      <th>%vic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>16</td>
      <td>Sweden</td>
      <td>1014</td>
      <td>496</td>
      <td>293</td>
      <td>225</td>
      <td>49.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>England</td>
      <td>998</td>
      <td>566</td>
      <td>191</td>
      <td>241</td>
      <td>57.0</td>
    </tr>
    <tr>
      <td>25</td>
      <td>Brazil</td>
      <td>981</td>
      <td>625</td>
      <td>157</td>
      <td>199</td>
      <td>64.0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Argentina</td>
      <td>979</td>
      <td>526</td>
      <td>211</td>
      <td>242</td>
      <td>54.0</td>
    </tr>
    <tr>
      <td>15</td>
      <td>Germany</td>
      <td>946</td>
      <td>553</td>
      <td>199</td>
      <td>194</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Hungary</td>
      <td>926</td>
      <td>434</td>
      <td>290</td>
      <td>202</td>
      <td>47.0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Uruguay</td>
      <td>882</td>
      <td>381</td>
      <td>286</td>
      <td>215</td>
      <td>43.0</td>
    </tr>
    <tr>
      <td>46</td>
      <td>Mexico</td>
      <td>867</td>
      <td>437</td>
      <td>230</td>
      <td>200</td>
      <td>50.0</td>
    </tr>
    <tr>
      <td>110</td>
      <td>South Korea</td>
      <td>854</td>
      <td>449</td>
      <td>185</td>
      <td>220</td>
      <td>53.0</td>
    </tr>
    <tr>
      <td>12</td>
      <td>France</td>
      <td>838</td>
      <td>415</td>
      <td>245</td>
      <td>178</td>
      <td>50.0</td>
    </tr>
    <tr>
      <td>36</td>
      <td>Poland</td>
      <td>813</td>
      <td>349</td>
      <td>258</td>
      <td>206</td>
      <td>43.0</td>
    </tr>
    <tr>
      <td>17</td>
      <td>Norway</td>
      <td>802</td>
      <td>291</td>
      <td>328</td>
      <td>183</td>
      <td>36.0</td>
    </tr>
    <tr>
      <td>18</td>
      <td>Italy</td>
      <td>799</td>
      <td>422</td>
      <td>153</td>
      <td>224</td>
      <td>53.0</td>
    </tr>
    <tr>
      <td>13</td>
      <td>Switzerland</td>
      <td>795</td>
      <td>278</td>
      <td>344</td>
      <td>173</td>
      <td>35.0</td>
    </tr>
    <tr>
      <td>24</td>
      <td>Denmark</td>
      <td>793</td>
      <td>355</td>
      <td>268</td>
      <td>170</td>
      <td>45.0</td>
    </tr>
    <tr>
      <td>14</td>
      <td>Netherlands</td>
      <td>782</td>
      <td>397</td>
      <td>209</td>
      <td>176</td>
      <td>51.0</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Austria</td>
      <td>776</td>
      <td>322</td>
      <td>284</td>
      <td>170</td>
      <td>41.0</td>
    </tr>
    <tr>
      <td>0</td>
      <td>Scotland</td>
      <td>775</td>
      <td>367</td>
      <td>243</td>
      <td>165</td>
      <td>47.0</td>
    </tr>
    <tr>
      <td>11</td>
      <td>Belgium</td>
      <td>766</td>
      <td>331</td>
      <td>271</td>
      <td>164</td>
      <td>43.0</td>
    </tr>
    <tr>
      <td>19</td>
      <td>Chile</td>
      <td>760</td>
      <td>290</td>
      <td>310</td>
      <td>160</td>
      <td>38.0</td>
    </tr>
    <tr>
      <td>20</td>
      <td>Finland</td>
      <td>742</td>
      <td>189</td>
      <td>399</td>
      <td>154</td>
      <td>25.0</td>
    </tr>
    <tr>
      <td>99</td>
      <td>Zambia</td>
      <td>715</td>
      <td>327</td>
      <td>198</td>
      <td>190</td>
      <td>46.0</td>
    </tr>
    <tr>
      <td>29</td>
      <td>Paraguay</td>
      <td>714</td>
      <td>254</td>
      <td>273</td>
      <td>187</td>
      <td>36.0</td>
    </tr>
    <tr>
      <td>23</td>
      <td>Russia</td>
      <td>692</td>
      <td>355</td>
      <td>155</td>
      <td>182</td>
      <td>51.0</td>
    </tr>
    <tr>
      <td>37</td>
      <td>Spain</td>
      <td>692</td>
      <td>404</td>
      <td>130</td>
      <td>158</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>40</td>
      <td>Romania</td>
      <td>692</td>
      <td>305</td>
      <td>208</td>
      <td>179</td>
      <td>44.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>United States</td>
      <td>684</td>
      <td>291</td>
      <td>249</td>
      <td>144</td>
      <td>43.0</td>
    </tr>
    <tr>
      <td>54</td>
      <td>Bulgaria</td>
      <td>669</td>
      <td>249</td>
      <td>253</td>
      <td>167</td>
      <td>37.0</td>
    </tr>
    <tr>
      <td>66</td>
      <td>Trinidad and Tobago</td>
      <td>662</td>
      <td>289</td>
      <td>235</td>
      <td>138</td>
      <td>44.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Wales</td>
      <td>652</td>
      <td>204</td>
      <td>307</td>
      <td>141</td>
      <td>31.0</td>
    </tr>
    <tr>
      <td>60</td>
      <td>Kenya</td>
      <td>643</td>
      <td>243</td>
      <td>242</td>
      <td>158</td>
      <td>38.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Northern Ireland</td>
      <td>640</td>
      <td>165</td>
      <td>329</td>
      <td>146</td>
      <td>26.0</td>
    </tr>
    <tr>
      <td>138</td>
      <td>Thailand</td>
      <td>634</td>
      <td>235</td>
      <td>253</td>
      <td>146</td>
      <td>37.0</td>
    </tr>
    <tr>
      <td>55</td>
      <td>Egypt</td>
      <td>631</td>
      <td>309</td>
      <td>167</td>
      <td>155</td>
      <td>49.0</td>
    </tr>
    <tr>
      <td>63</td>
      <td>Peru</td>
      <td>622</td>
      <td>201</td>
      <td>269</td>
      <td>152</td>
      <td>32.0</td>
    </tr>
    <tr>
      <td>61</td>
      <td>Uganda</td>
      <td>622</td>
      <td>265</td>
      <td>190</td>
      <td>167</td>
      <td>43.0</td>
    </tr>
    <tr>
      <td>27</td>
      <td>Japan</td>
      <td>620</td>
      <td>296</td>
      <td>182</td>
      <td>142</td>
      <td>48.0</td>
    </tr>
    <tr>
      <td>38</td>
      <td>Portugal</td>
      <td>611</td>
      <td>294</td>
      <td>173</td>
      <td>144</td>
      <td>48.0</td>
    </tr>
    <tr>
      <td>172</td>
      <td>Saudi Arabia</td>
      <td>607</td>
      <td>288</td>
      <td>182</td>
      <td>137</td>
      <td>47.0</td>
    </tr>
    <tr>
      <td>32</td>
      <td>Costa Rica</td>
      <td>601</td>
      <td>256</td>
      <td>193</td>
      <td>152</td>
      <td>43.0</td>
    </tr>
    <tr>
      <td>120</td>
      <td>Ghana</td>
      <td>598</td>
      <td>284</td>
      <td>155</td>
      <td>159</td>
      <td>47.0</td>
    </tr>
    <tr>
      <td>132</td>
      <td>Malaysia</td>
      <td>594</td>
      <td>204</td>
      <td>240</td>
      <td>150</td>
      <td>34.0</td>
    </tr>
    <tr>
      <td>49</td>
      <td>China PR</td>
      <td>589</td>
      <td>284</td>
      <td>171</td>
      <td>134</td>
      <td>48.0</td>
    </tr>
    <tr>
      <td>147</td>
      <td>Tunisia</td>
      <td>579</td>
      <td>241</td>
      <td>170</td>
      <td>168</td>
      <td>42.0</td>
    </tr>
    <tr>
      <td>116</td>
      <td>Nigeria</td>
      <td>574</td>
      <td>265</td>
      <td>143</td>
      <td>166</td>
      <td>46.0</td>
    </tr>
    <tr>
      <td>51</td>
      <td>Turkey</td>
      <td>565</td>
      <td>219</td>
      <td>210</td>
      <td>136</td>
      <td>39.0</td>
    </tr>
    <tr>
      <td>59</td>
      <td>Republic of Ireland</td>
      <td>564</td>
      <td>219</td>
      <td>185</td>
      <td>160</td>
      <td>39.0</td>
    </tr>
    <tr>
      <td>158</td>
      <td>Ivory Coast</td>
      <td>563</td>
      <td>280</td>
      <td>140</td>
      <td>143</td>
      <td>50.0</td>
    </tr>
    <tr>
      <td>71</td>
      <td>Greece</td>
      <td>562</td>
      <td>206</td>
      <td>216</td>
      <td>140</td>
      <td>37.0</td>
    </tr>
    <tr>
      <td>58</td>
      <td>Jamaica</td>
      <td>562</td>
      <td>220</td>
      <td>214</td>
      <td>128</td>
      <td>39.0</td>
    </tr>
  </tbody>
</table>
</div>



Lets see how the countries of the former USSR block are doing in comparison the he major football powers. 


```python
statistics.loc[['Russia', 'Czechoslovakia', 'Brazil', 'Belarus', 'Ukraine', 'Slovakia', 'Kyrgyzstan', 'Moldova', 'Azerbaijan', 'Armenia', 'Kazakhstan', 'Georgia', 'Tajikistan', 'Turkmenistan', 'Uzbekistan',  'Czech Republic' ]]
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
      <th>nb_match</th>
      <th>nb_vict</th>
      <th>nb_defts</th>
      <th>nb_draws</th>
      <th>%vic</th>
    </tr>
    <tr>
      <th>country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Russia</td>
      <td>692</td>
      <td>355</td>
      <td>155</td>
      <td>182</td>
      <td>51.0</td>
    </tr>
    <tr>
      <td>Czechoslovakia</td>
      <td>496</td>
      <td>222</td>
      <td>161</td>
      <td>113</td>
      <td>45.0</td>
    </tr>
    <tr>
      <td>Brazil</td>
      <td>981</td>
      <td>625</td>
      <td>157</td>
      <td>199</td>
      <td>64.0</td>
    </tr>
    <tr>
      <td>Belarus</td>
      <td>248</td>
      <td>76</td>
      <td>109</td>
      <td>63</td>
      <td>31.0</td>
    </tr>
    <tr>
      <td>Ukraine</td>
      <td>265</td>
      <td>122</td>
      <td>69</td>
      <td>74</td>
      <td>46.0</td>
    </tr>
    <tr>
      <td>Slovakia</td>
      <td>289</td>
      <td>118</td>
      <td>103</td>
      <td>68</td>
      <td>41.0</td>
    </tr>
    <tr>
      <td>Kyrgyzstan</td>
      <td>140</td>
      <td>42</td>
      <td>76</td>
      <td>22</td>
      <td>30.0</td>
    </tr>
    <tr>
      <td>Moldova</td>
      <td>231</td>
      <td>45</td>
      <td>130</td>
      <td>56</td>
      <td>19.0</td>
    </tr>
    <tr>
      <td>Azerbaijan</td>
      <td>243</td>
      <td>47</td>
      <td>131</td>
      <td>65</td>
      <td>19.0</td>
    </tr>
    <tr>
      <td>Armenia</td>
      <td>208</td>
      <td>51</td>
      <td>114</td>
      <td>43</td>
      <td>25.0</td>
    </tr>
    <tr>
      <td>Kazakhstan</td>
      <td>196</td>
      <td>47</td>
      <td>102</td>
      <td>47</td>
      <td>24.0</td>
    </tr>
    <tr>
      <td>Georgia</td>
      <td>239</td>
      <td>73</td>
      <td>116</td>
      <td>50</td>
      <td>31.0</td>
    </tr>
    <tr>
      <td>Tajikistan</td>
      <td>138</td>
      <td>55</td>
      <td>58</td>
      <td>25</td>
      <td>40.0</td>
    </tr>
    <tr>
      <td>Turkmenistan</td>
      <td>133</td>
      <td>49</td>
      <td>61</td>
      <td>23</td>
      <td>37.0</td>
    </tr>
    <tr>
      <td>Uzbekistan</td>
      <td>269</td>
      <td>116</td>
      <td>95</td>
      <td>58</td>
      <td>43.0</td>
    </tr>
    <tr>
      <td>Czech Republic</td>
      <td>295</td>
      <td>159</td>
      <td>77</td>
      <td>59</td>
      <td>54.0</td>
    </tr>
  </tbody>
</table>
</div>



Czechoslovakia broke up due to labile ethnic and economic differences, which are proxy to the football results. Czechoslovakia had a 45% win ratio, while the Czech Republic alone has 54% win ratio, while Slovakia has only 41%. My home country of Kazakhstan has a mere 24% win ratio in UEFA, while the countries that remain in the Asian football federation such as Uzbekistan, Tajikistan and Turkmenistan are enjoying higher win percentages. Brazil is a standout example of 65% win ratio.

Now, I will recreate the data sets.


```python
#get the coloumn of results (wins, ties and losses)
win= np.where(results.home_score > results.away_score, 'win', None)
tie=np.where(results.home_score == results.away_score, 'tie', None)
loss= np.where(results.home_score < results.away_score, 'loss', None)

results2=pd.DataFrame([win, tie, loss]).T
results2

x=[value[value != None]  for value in results2.values]
results['result']= x
results['result']=np.squeeze(results.result.tolist())

#get the number of goals
results['goals']= results.home_score + results.away_score

#home
home_teams=results.groupby(['home_team','result']).count()['city'].sort_values(ascending=False).reset_index().rename(columns={'city': 'count'})

    
home_matches=[]
for team in home_teams.home_team:
    tot_matches= home_teams[home_teams.home_team== team]['count'].sum()
    home_matches.append(tot_matches)
   
home_teams['home_matches']=home_matches
home_teams['pct_home_victory']= home_teams['count']/ home_teams['home_matches']


#away
away_teams=results.groupby(['away_team','result']).count()['city'].sort_values(ascending=False).reset_index().rename(columns={'city': 'count'})
away_teams.replace({'loss': 'win', 'win':'loss'}, inplace=True) #loss means victory for the away team

away_tot_matches=[]
for team in away_teams.away_team:
    tot_matches= away_teams[away_teams.away_team == team]['count'].sum()
    away_tot_matches.append(tot_matches)

away_teams['away_matches']= away_tot_matches
away_teams['pct_victory_away'] = away_teams['count']/away_teams['away_matches']


#adjusting variable names and index
home_teams.rename(columns={'result': 'home_results', 'count': 'home_count'}, inplace=True)
home_teams.set_index('home_team', inplace=True)
away_teams.rename(columns={'result': 'away_results', 'count': 'away_count'}, inplace=True)
away_teams.set_index('away_team', inplace=True)



#defining winners and loosers
home_winners= home_teams[home_teams.home_results=='win']
away_winners= away_teams[away_teams.away_results=='win']
home_losers= home_teams[home_teams.home_results=='loss']
away_losers= away_teams[away_teams.away_results=='loss']


#merging datasets
winners=pd.merge(home_winners, away_winners, left_index=True, right_index=True, how='inner')
losers=pd.merge(home_losers, away_losers, left_index=True, right_index=True, how='inner')
losers.rename(columns={'pct_home_victory': 'pct_home_defeats', 'pct_victory_away': 'pct_away_defeats'}, inplace=True)

winners['tot_count']= winners.home_count + winners.away_count
winners['tot_matches']= winners.home_matches + winners.away_matches
winners['tot_pct_victory']= winners.tot_count/winners.tot_matches
winners= winners[winners.tot_matches >= 100] #getting only clubs who have played at least 100 matches
winners_pct= winners[['pct_home_victory', 'pct_victory_away', 'tot_pct_victory']]

losers['tot_count']= losers.home_count + losers.away_count
losers['tot_matches']= losers.home_matches + losers.away_matches
losers['tot_pct_defeats']= losers.tot_count/losers.tot_matches
losers= losers[losers.tot_matches >= 100] #getting only clubs who have played at least 100 matches
losers_pct= losers[['pct_home_defeats', 'pct_away_defeats', 'tot_pct_defeats']]

#total percentage
winners_pct.sort_values(by='tot_pct_victory', ascending=False)
winners_pct=np.round(winners_pct*100, 2)
winners_pct['tot_count']= winners.tot_count
winners_pct['tot_matches']= winners.tot_matches

losers_pct=np.round(losers_pct*100, 2)
losers_pct['tot_count']= losers.tot_count
losers_pct['tot_matches']= losers.tot_matches


winners_pct.sort_values(by='tot_pct_victory', ascending=False)
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
      <th>pct_home_victory</th>
      <th>pct_victory_away</th>
      <th>tot_pct_victory</th>
      <th>tot_count</th>
      <th>tot_matches</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Brazil</td>
      <td>71.08</td>
      <td>53.62</td>
      <td>63.71</td>
      <td>625</td>
      <td>981</td>
    </tr>
    <tr>
      <td>Germany</td>
      <td>62.62</td>
      <td>53.72</td>
      <td>58.46</td>
      <td>553</td>
      <td>946</td>
    </tr>
    <tr>
      <td>Spain</td>
      <td>68.07</td>
      <td>48.06</td>
      <td>58.38</td>
      <td>404</td>
      <td>692</td>
    </tr>
    <tr>
      <td>England</td>
      <td>61.91</td>
      <td>51.68</td>
      <td>56.71</td>
      <td>566</td>
      <td>998</td>
    </tr>
    <tr>
      <td>Iran</td>
      <td>63.08</td>
      <td>43.52</td>
      <td>54.55</td>
      <td>270</td>
      <td>495</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>Malta</td>
      <td>16.06</td>
      <td>6.78</td>
      <td>11.90</td>
      <td>47</td>
      <td>395</td>
    </tr>
    <tr>
      <td>Seychelles</td>
      <td>22.00</td>
      <td>3.28</td>
      <td>11.71</td>
      <td>13</td>
      <td>111</td>
    </tr>
    <tr>
      <td>Somalia</td>
      <td>11.63</td>
      <td>6.45</td>
      <td>8.57</td>
      <td>9</td>
      <td>105</td>
    </tr>
    <tr>
      <td>Luxembourg</td>
      <td>11.31</td>
      <td>5.03</td>
      <td>8.50</td>
      <td>34</td>
      <td>400</td>
    </tr>
    <tr>
      <td>Liechtenstein</td>
      <td>10.53</td>
      <td>4.95</td>
      <td>7.65</td>
      <td>15</td>
      <td>196</td>
    </tr>
  </tbody>
</table>
<p>192 rows × 5 columns</p>
</div>




```python
winners.sort_values(by=['pct_victory_away'], ascending=False).head(25)
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
      <th>home_results</th>
      <th>home_count</th>
      <th>home_matches</th>
      <th>pct_home_victory</th>
      <th>away_results</th>
      <th>away_count</th>
      <th>away_matches</th>
      <th>pct_victory_away</th>
      <th>tot_count</th>
      <th>tot_matches</th>
      <th>tot_pct_victory</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Germany</td>
      <td>win</td>
      <td>315</td>
      <td>503</td>
      <td>0.626243</td>
      <td>win</td>
      <td>238</td>
      <td>443</td>
      <td>0.537246</td>
      <td>553</td>
      <td>946</td>
      <td>0.584567</td>
    </tr>
    <tr>
      <td>Brazil</td>
      <td>win</td>
      <td>403</td>
      <td>567</td>
      <td>0.710758</td>
      <td>win</td>
      <td>222</td>
      <td>414</td>
      <td>0.536232</td>
      <td>625</td>
      <td>981</td>
      <td>0.637105</td>
    </tr>
    <tr>
      <td>England</td>
      <td>win</td>
      <td>304</td>
      <td>491</td>
      <td>0.619145</td>
      <td>win</td>
      <td>262</td>
      <td>507</td>
      <td>0.516765</td>
      <td>566</td>
      <td>998</td>
      <td>0.567134</td>
    </tr>
    <tr>
      <td>Tahiti</td>
      <td>win</td>
      <td>53</td>
      <td>94</td>
      <td>0.563830</td>
      <td>win</td>
      <td>56</td>
      <td>116</td>
      <td>0.482759</td>
      <td>109</td>
      <td>210</td>
      <td>0.519048</td>
    </tr>
    <tr>
      <td>Spain</td>
      <td>win</td>
      <td>243</td>
      <td>357</td>
      <td>0.680672</td>
      <td>win</td>
      <td>161</td>
      <td>335</td>
      <td>0.480597</td>
      <td>404</td>
      <td>692</td>
      <td>0.583815</td>
    </tr>
    <tr>
      <td>South Korea</td>
      <td>win</td>
      <td>273</td>
      <td>464</td>
      <td>0.588362</td>
      <td>win</td>
      <td>176</td>
      <td>390</td>
      <td>0.451282</td>
      <td>449</td>
      <td>854</td>
      <td>0.525761</td>
    </tr>
    <tr>
      <td>Croatia</td>
      <td>win</td>
      <td>94</td>
      <td>151</td>
      <td>0.622517</td>
      <td>win</td>
      <td>74</td>
      <td>165</td>
      <td>0.448485</td>
      <td>168</td>
      <td>316</td>
      <td>0.531646</td>
    </tr>
    <tr>
      <td>Netherlands</td>
      <td>win</td>
      <td>236</td>
      <td>415</td>
      <td>0.568675</td>
      <td>win</td>
      <td>161</td>
      <td>367</td>
      <td>0.438692</td>
      <td>397</td>
      <td>782</td>
      <td>0.507673</td>
    </tr>
    <tr>
      <td>Russia</td>
      <td>win</td>
      <td>183</td>
      <td>298</td>
      <td>0.614094</td>
      <td>win</td>
      <td>172</td>
      <td>394</td>
      <td>0.436548</td>
      <td>355</td>
      <td>692</td>
      <td>0.513006</td>
    </tr>
    <tr>
      <td>Iran</td>
      <td>win</td>
      <td>176</td>
      <td>279</td>
      <td>0.630824</td>
      <td>win</td>
      <td>94</td>
      <td>216</td>
      <td>0.435185</td>
      <td>270</td>
      <td>495</td>
      <td>0.545455</td>
    </tr>
    <tr>
      <td>Japan</td>
      <td>win</td>
      <td>189</td>
      <td>368</td>
      <td>0.513587</td>
      <td>win</td>
      <td>107</td>
      <td>252</td>
      <td>0.424603</td>
      <td>296</td>
      <td>620</td>
      <td>0.477419</td>
    </tr>
    <tr>
      <td>Czech Republic</td>
      <td>win</td>
      <td>95</td>
      <td>144</td>
      <td>0.659722</td>
      <td>win</td>
      <td>64</td>
      <td>151</td>
      <td>0.423841</td>
      <td>159</td>
      <td>295</td>
      <td>0.538983</td>
    </tr>
    <tr>
      <td>Italy</td>
      <td>win</td>
      <td>276</td>
      <td>442</td>
      <td>0.624434</td>
      <td>win</td>
      <td>146</td>
      <td>357</td>
      <td>0.408964</td>
      <td>422</td>
      <td>799</td>
      <td>0.528160</td>
    </tr>
    <tr>
      <td>Sweden</td>
      <td>win</td>
      <td>281</td>
      <td>485</td>
      <td>0.579381</td>
      <td>win</td>
      <td>215</td>
      <td>529</td>
      <td>0.406427</td>
      <td>496</td>
      <td>1014</td>
      <td>0.489152</td>
    </tr>
    <tr>
      <td>Australia</td>
      <td>win</td>
      <td>164</td>
      <td>292</td>
      <td>0.561644</td>
      <td>win</td>
      <td>89</td>
      <td>221</td>
      <td>0.402715</td>
      <td>253</td>
      <td>513</td>
      <td>0.493177</td>
    </tr>
    <tr>
      <td>Yugoslavia</td>
      <td>win</td>
      <td>108</td>
      <td>190</td>
      <td>0.568421</td>
      <td>win</td>
      <td>115</td>
      <td>290</td>
      <td>0.396552</td>
      <td>223</td>
      <td>480</td>
      <td>0.464583</td>
    </tr>
    <tr>
      <td>New Caledonia</td>
      <td>win</td>
      <td>83</td>
      <td>133</td>
      <td>0.624060</td>
      <td>win</td>
      <td>37</td>
      <td>94</td>
      <td>0.393617</td>
      <td>120</td>
      <td>227</td>
      <td>0.528634</td>
    </tr>
    <tr>
      <td>Serbia</td>
      <td>win</td>
      <td>57</td>
      <td>104</td>
      <td>0.548077</td>
      <td>win</td>
      <td>66</td>
      <td>169</td>
      <td>0.390533</td>
      <td>123</td>
      <td>273</td>
      <td>0.450549</td>
    </tr>
    <tr>
      <td>Scotland</td>
      <td>win</td>
      <td>212</td>
      <td>377</td>
      <td>0.562334</td>
      <td>win</td>
      <td>155</td>
      <td>398</td>
      <td>0.389447</td>
      <td>367</td>
      <td>775</td>
      <td>0.473548</td>
    </tr>
    <tr>
      <td>Zambia</td>
      <td>win</td>
      <td>165</td>
      <td>299</td>
      <td>0.551839</td>
      <td>win</td>
      <td>162</td>
      <td>416</td>
      <td>0.389423</td>
      <td>327</td>
      <td>715</td>
      <td>0.457343</td>
    </tr>
    <tr>
      <td>France</td>
      <td>win</td>
      <td>279</td>
      <td>483</td>
      <td>0.577640</td>
      <td>win</td>
      <td>136</td>
      <td>355</td>
      <td>0.383099</td>
      <td>415</td>
      <td>838</td>
      <td>0.495227</td>
    </tr>
    <tr>
      <td>Mexico</td>
      <td>win</td>
      <td>301</td>
      <td>512</td>
      <td>0.587891</td>
      <td>win</td>
      <td>136</td>
      <td>355</td>
      <td>0.383099</td>
      <td>437</td>
      <td>867</td>
      <td>0.504037</td>
    </tr>
    <tr>
      <td>Argentina</td>
      <td>win</td>
      <td>361</td>
      <td>547</td>
      <td>0.659963</td>
      <td>win</td>
      <td>165</td>
      <td>432</td>
      <td>0.381944</td>
      <td>526</td>
      <td>979</td>
      <td>0.537283</td>
    </tr>
    <tr>
      <td>Portugal</td>
      <td>win</td>
      <td>184</td>
      <td>323</td>
      <td>0.569659</td>
      <td>win</td>
      <td>110</td>
      <td>288</td>
      <td>0.381944</td>
      <td>294</td>
      <td>611</td>
      <td>0.481178</td>
    </tr>
    <tr>
      <td>German DR</td>
      <td>win</td>
      <td>77</td>
      <td>132</td>
      <td>0.583333</td>
      <td>win</td>
      <td>63</td>
      <td>166</td>
      <td>0.379518</td>
      <td>140</td>
      <td>298</td>
      <td>0.469799</td>
    </tr>
  </tbody>
</table>
</div>



Not surprisingly, Brazil is a home fortress. However, I am sure any Brazilian fan will gladly trade that statistic for a much lower value to forget the shameful game of 2014 Brazil lost to when eventual winner 
Germany 1-7 in home World Cup quarterfinals. 
 
Fascinatingly enough, to be in the top 25 of the team who wins away from home the team needs only 37pct,  while the average home win ratio is a mere 45%, which means approximately  20pct are draws. The results are drastically different in top-level competitions in different countries. For example in Russia, over 40% of all games are drawn.


```python
winners.mean()
```




    home_count          101.140625
    home_matches        205.723958
    pct_home_victory      0.456876
    away_count           58.890625
    away_matches        204.661458
    pct_victory_away      0.256587
    tot_count           160.031250
    tot_matches         410.385417
    tot_pct_victory       0.354718
    dtype: float64




```python
losers.sort_values(by=['pct_home_defeats'], ascending=False).head(25)
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
      <th>home_results</th>
      <th>home_count</th>
      <th>home_matches</th>
      <th>pct_home_defeats</th>
      <th>away_results</th>
      <th>away_count</th>
      <th>away_matches</th>
      <th>pct_away_defeats</th>
      <th>tot_count</th>
      <th>tot_matches</th>
      <th>tot_pct_defeats</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>San Marino</td>
      <td>loss</td>
      <td>78</td>
      <td>81</td>
      <td>0.962963</td>
      <td>loss</td>
      <td>78</td>
      <td>80</td>
      <td>0.975000</td>
      <td>156</td>
      <td>161</td>
      <td>0.968944</td>
    </tr>
    <tr>
      <td>Andorra</td>
      <td>loss</td>
      <td>65</td>
      <td>82</td>
      <td>0.792683</td>
      <td>loss</td>
      <td>73</td>
      <td>80</td>
      <td>0.912500</td>
      <td>138</td>
      <td>162</td>
      <td>0.851852</td>
    </tr>
    <tr>
      <td>Luxembourg</td>
      <td>loss</td>
      <td>162</td>
      <td>221</td>
      <td>0.733032</td>
      <td>loss</td>
      <td>153</td>
      <td>179</td>
      <td>0.854749</td>
      <td>315</td>
      <td>400</td>
      <td>0.787500</td>
    </tr>
    <tr>
      <td>Liechtenstein</td>
      <td>loss</td>
      <td>69</td>
      <td>95</td>
      <td>0.726316</td>
      <td>loss</td>
      <td>87</td>
      <td>101</td>
      <td>0.861386</td>
      <td>156</td>
      <td>196</td>
      <td>0.795918</td>
    </tr>
    <tr>
      <td>Somalia</td>
      <td>loss</td>
      <td>31</td>
      <td>43</td>
      <td>0.720930</td>
      <td>loss</td>
      <td>53</td>
      <td>62</td>
      <td>0.854839</td>
      <td>84</td>
      <td>105</td>
      <td>0.800000</td>
    </tr>
    <tr>
      <td>Malta</td>
      <td>loss</td>
      <td>138</td>
      <td>218</td>
      <td>0.633028</td>
      <td>loss</td>
      <td>143</td>
      <td>177</td>
      <td>0.807910</td>
      <td>281</td>
      <td>395</td>
      <td>0.711392</td>
    </tr>
    <tr>
      <td>Macau</td>
      <td>loss</td>
      <td>38</td>
      <td>61</td>
      <td>0.622951</td>
      <td>loss</td>
      <td>53</td>
      <td>65</td>
      <td>0.815385</td>
      <td>91</td>
      <td>126</td>
      <td>0.722222</td>
    </tr>
    <tr>
      <td>Faroe Islands</td>
      <td>loss</td>
      <td>71</td>
      <td>114</td>
      <td>0.622807</td>
      <td>loss</td>
      <td>88</td>
      <td>108</td>
      <td>0.814815</td>
      <td>159</td>
      <td>222</td>
      <td>0.716216</td>
    </tr>
    <tr>
      <td>Seychelles</td>
      <td>loss</td>
      <td>31</td>
      <td>50</td>
      <td>0.620000</td>
      <td>loss</td>
      <td>49</td>
      <td>61</td>
      <td>0.803279</td>
      <td>80</td>
      <td>111</td>
      <td>0.720721</td>
    </tr>
    <tr>
      <td>Nicaragua</td>
      <td>loss</td>
      <td>25</td>
      <td>46</td>
      <td>0.543478</td>
      <td>loss</td>
      <td>87</td>
      <td>105</td>
      <td>0.828571</td>
      <td>112</td>
      <td>151</td>
      <td>0.741722</td>
    </tr>
    <tr>
      <td>Puerto Rico</td>
      <td>loss</td>
      <td>28</td>
      <td>52</td>
      <td>0.538462</td>
      <td>loss</td>
      <td>46</td>
      <td>63</td>
      <td>0.730159</td>
      <td>74</td>
      <td>115</td>
      <td>0.643478</td>
    </tr>
    <tr>
      <td>Zanzibar</td>
      <td>loss</td>
      <td>31</td>
      <td>58</td>
      <td>0.534483</td>
      <td>loss</td>
      <td>92</td>
      <td>147</td>
      <td>0.625850</td>
      <td>123</td>
      <td>205</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <td>Pakistan</td>
      <td>loss</td>
      <td>38</td>
      <td>73</td>
      <td>0.520548</td>
      <td>loss</td>
      <td>78</td>
      <td>115</td>
      <td>0.678261</td>
      <td>116</td>
      <td>188</td>
      <td>0.617021</td>
    </tr>
    <tr>
      <td>Cyprus</td>
      <td>loss</td>
      <td>105</td>
      <td>202</td>
      <td>0.519802</td>
      <td>loss</td>
      <td>126</td>
      <td>158</td>
      <td>0.797468</td>
      <td>231</td>
      <td>360</td>
      <td>0.641667</td>
    </tr>
    <tr>
      <td>Cambodia</td>
      <td>loss</td>
      <td>61</td>
      <td>121</td>
      <td>0.504132</td>
      <td>loss</td>
      <td>81</td>
      <td>96</td>
      <td>0.843750</td>
      <td>142</td>
      <td>217</td>
      <td>0.654378</td>
    </tr>
    <tr>
      <td>Moldova</td>
      <td>loss</td>
      <td>48</td>
      <td>97</td>
      <td>0.494845</td>
      <td>loss</td>
      <td>82</td>
      <td>134</td>
      <td>0.611940</td>
      <td>130</td>
      <td>231</td>
      <td>0.562771</td>
    </tr>
    <tr>
      <td>Dominica</td>
      <td>loss</td>
      <td>46</td>
      <td>96</td>
      <td>0.479167</td>
      <td>loss</td>
      <td>54</td>
      <td>84</td>
      <td>0.642857</td>
      <td>100</td>
      <td>180</td>
      <td>0.555556</td>
    </tr>
    <tr>
      <td>Vanuatu</td>
      <td>loss</td>
      <td>30</td>
      <td>63</td>
      <td>0.476190</td>
      <td>loss</td>
      <td>66</td>
      <td>111</td>
      <td>0.594595</td>
      <td>96</td>
      <td>174</td>
      <td>0.551724</td>
    </tr>
    <tr>
      <td>Aruba</td>
      <td>loss</td>
      <td>28</td>
      <td>59</td>
      <td>0.474576</td>
      <td>loss</td>
      <td>42</td>
      <td>59</td>
      <td>0.711864</td>
      <td>70</td>
      <td>118</td>
      <td>0.593220</td>
    </tr>
    <tr>
      <td>Finland</td>
      <td>loss</td>
      <td>156</td>
      <td>329</td>
      <td>0.474164</td>
      <td>loss</td>
      <td>243</td>
      <td>413</td>
      <td>0.588378</td>
      <td>399</td>
      <td>742</td>
      <td>0.537736</td>
    </tr>
    <tr>
      <td>Sri Lanka</td>
      <td>loss</td>
      <td>41</td>
      <td>87</td>
      <td>0.471264</td>
      <td>loss</td>
      <td>79</td>
      <td>107</td>
      <td>0.738318</td>
      <td>120</td>
      <td>194</td>
      <td>0.618557</td>
    </tr>
    <tr>
      <td>Nepal</td>
      <td>loss</td>
      <td>40</td>
      <td>85</td>
      <td>0.470588</td>
      <td>loss</td>
      <td>67</td>
      <td>88</td>
      <td>0.761364</td>
      <td>107</td>
      <td>173</td>
      <td>0.618497</td>
    </tr>
    <tr>
      <td>Bangladesh</td>
      <td>loss</td>
      <td>53</td>
      <td>113</td>
      <td>0.469027</td>
      <td>loss</td>
      <td>58</td>
      <td>89</td>
      <td>0.651685</td>
      <td>111</td>
      <td>202</td>
      <td>0.549505</td>
    </tr>
    <tr>
      <td>Laos</td>
      <td>loss</td>
      <td>33</td>
      <td>72</td>
      <td>0.458333</td>
      <td>loss</td>
      <td>72</td>
      <td>89</td>
      <td>0.808989</td>
      <td>105</td>
      <td>161</td>
      <td>0.652174</td>
    </tr>
    <tr>
      <td>Lithuania</td>
      <td>loss</td>
      <td>67</td>
      <td>151</td>
      <td>0.443709</td>
      <td>loss</td>
      <td>130</td>
      <td>213</td>
      <td>0.610329</td>
      <td>197</td>
      <td>364</td>
      <td>0.541209</td>
    </tr>
  </tbody>
</table>
</div>



Funny story about the Italian coach, Giovanni Trapattoni. In a good spirit, he arrived at Rimini hotel, near San Marino, and started to greet the personnel of the hotel. Jokingly, he asked the Manager if they were excited about the game? The manager politely said yes and he was very much looking forward to the game since he was meticulously preparing for the game. Then the coached why was it so important for him? To which the manager replied: "I am the goalkeeper". 
