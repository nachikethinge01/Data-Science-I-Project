#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import itertools


# In[60]:


# organizing datasets
season1718 = pd.read_csv("./NBA_season1718_salary.csv")
season1718.head()


# In[61]:


players = pd.read_csv("./players.csv")
players.head()


# In[62]:


season_stats = pd.read_csv("./Seasons_Stats.csv")
season_stats.head()


# In[63]:


stats2017 = season_stats[season_stats['Year'] == 2017.0]
stat = stats2017.drop_duplicates(subset=['Player'], keep='first')
stats = stat[['Year','Player','Pos','Age','G',
                   'PTS','AST','BLK','STL']]
stats


# In[64]:


new_1718 = season1718.drop_duplicates(subset=['Player'], keep='first')
new_1718


# In[65]:


result = pd.merge(stats, new_1718, on="Player")
stats_salary = result.drop_duplicates(subset=['Player'], keep='first')
stats_salary


# In[66]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[67]:


print('histograms of variables to see distribution and spread')
stats_salary.hist()


# In[68]:


print('correlation matrix')
stats_corr = stats_salary.drop(['Year', 'Unnamed: 0'], axis = 1)
stats_corr.corr()


# In[69]:


# making training and testing with random selection
training = stats_salary.sample(frac=0.8, random_state=25)
testing = stats_salary.drop(training.index)
training


# In[70]:


testing


# In[71]:


plt.scatter(training['PTS'], training['season17_18'])
print('PTS and salary')


# In[72]:


x = training.drop(['season17_18', 'Player', 'Pos', 'Tm', 'Unnamed: 0'], axis = 1).values
y = training['season17_18'].values
reg = LinearRegression().fit(x, y)
print(f"Training score: {reg.score(x, y)}")


# In[73]:


x_test = testing.drop(['season17_18', 'Player', 'Pos', 'Tm', 'Unnamed: 0'], axis = 1).values
y_test = testing['season17_18'].values
print(f"Testing score: {reg.score(x_test, y_test)}")


# In[ ]:




