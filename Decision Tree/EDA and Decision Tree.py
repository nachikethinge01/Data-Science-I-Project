#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd    
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import tree

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
import random
from statistics import mean


# In[16]:


salary_table = pd.read_csv("salary_table.csv")
seasons = pd.read_csv("Seasons_Stats.csv")


# In[17]:


salary_table = salary_table[['Player','season17_18']]
salary_table.rename(columns={'season17_18':'salary17_18'},inplace = True) #variable rename
salary_table['salary17_18'] = salary_table['salary17_18']/1000000 #transform salary to 'million'

seasons = seasons[seasons['Year']>=2017] 
stats17 = seasons[['Year','Player','Pos','Age','G','PER',
                   'MP','PTS','AST','TRB','TOV','BLK','STL']]

stats17.drop_duplicates(subset=['Player'], keep='first',inplace=True) #drop duplicate data

c = ['MPG','PPG','APG','RPG','TOPG','BPG','SPG']
w = ['MP','PTS','AST','TRB','TOV','BLK','STL'] 

for i,s in zip(c,w):
    stats17[i] = stats17[s] / stats17['G']

stats17.drop(w,axis=1,inplace=True)
#stats17.drop(['G'],axis=1,inplace=True)
stats17.loc[stats17['Pos'] == 'PF-C','Pos'] = 'PF'
stats_salary = pd.merge(stats17, salary_table)


# In[18]:


stats_salary.describe()


# In[19]:


stats_salary.corr()


# In[20]:


stats_salary.to_csv("stats_salary.csv")


# In[21]:


stats_salary.sort_values(by='salary17_18',ascending = False,inplace = True)
stats_salary[['Player','salary17_18']].head(10)


# In[22]:


salary_table['salary17_18'].describe()


# In[23]:


stats_salary = stats_salary.dropna()
Y = stats_salary['salary17_18']
X = stats_salary.drop(['salary17_18','Year', 'Player', 'Pos'],axis=1)


# In[24]:


X.columns


# In[25]:


X.head()


# In[26]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2) 


# In[27]:


print(len(x_train))
print(len(y_test))


# In[28]:


clf = RandomForestRegressor(random_state=42)
dtree = clf.fit(x_train, y_train)

yp = clf.predict(x_test)
print(clf.score(x_test, y_test))


# In[29]:


print(clf.score(x_train, y_train))


# In[30]:


df = {'Age': 20, 'G':mean(X['G']), 'PER': mean(X['PER']), 'MPG': mean(X['MPG']), 'PPG': mean(X['G']), 'APG': mean(X['APG']), 'RPG': mean(X['RPG']), 'TOPG': mean(X['TOPG']), 'BPG': mean(X['BPG']), 'SPG': mean(X['SPG'])}


# In[31]:


df2 = pd.DataFrame([df])


# In[32]:


df2


# In[33]:


new_pred = clf.predict(df2)
new_pred


# In[141]:


"$" + str(6.55773715 * 1000000)


# In[ ]:




