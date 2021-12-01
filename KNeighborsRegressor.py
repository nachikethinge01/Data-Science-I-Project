#!/usr/bin/env python
# coding: utf-8

# In[1400]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from math import sqrt

get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets
from sklearn import metrics
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split


# In[1401]:


salary = pd.read_csv ('/Users/allanwaweru/Downloads/csci3360 finalproject/2017salary.csv')

stats = pd.read_csv ('/Users/allanwaweru/Downloads/csci3360 finalproject/2017seasonstats.csv')


# In[1402]:



stats.head()


# In[1403]:


salary.head()


# In[1404]:


salary = salary[['Player', 'season17_18']]
salary.rename(columns={'season17_18':'salary17_18'},inplace = True)
salary['salary17_18'] = salary['salary17_18'].astype(int)


# In[1405]:


stats = stats[['Year','Player','Pos','Age','G','PER', 'MP','PTS','AST','TRB','TOV','BLK','STL']]



# In[1406]:


salary.columns


# In[1407]:


stats.columns


# In[1408]:


stats_salary.drop_duplicates(subset=['Player'], keep='first',inplace=True)
stats_salary.head(10)


# In[1384]:


c = ['MPG', 'APG','RPG','TOPG','BPG','SPG','PPG']
w = ['MP','AST','TRB','TOV','BLK','STL', 'PTS'] 


for i,s in zip(c,w):
    stats[i] = stats[s] / stats['G']

stats.drop(w,axis=1,inplace=True)
stats_salary = pd.merge(stats, salary) 


# In[1385]:


stats_salary.count()


# In[1386]:


plt.hist(stats_salary['salary17_18'],density=True,bins=50)
plt.xlabel('2017-2018 Salary(million)')
plt.ylabel('Density')

plt.show()


# In[1387]:


corrmat = stats_salary.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
#plot heat map
g=sns.heatmap(stats_salary[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[1388]:



stats_salary.sort_values(by='PPG',ascending=False,inplace = True)
stats_salary[['Player','PPG', 'salary17_18']].head(15)


# In[1389]:


sns.lmplot(x="APG", y="salary17_18", data=stats_salary,lowess=True).set(xlabel='APG', ylabel='Salary 2017-2018(Millions)')


# In[1390]:


plt.hist(stats_salary['salary17_18'],density=True,bins=50)
plt.xlabel('2017-2018 Salary(million)')
plt.ylabel('Density')
plt.show()


# In[1392]:



X = stats_salary.iloc[:,3:13]  #independent columns
y = stats_salary.iloc[:,-1]    #target column i.e salary

model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) 
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[1393]:


KNeighborsRegressor


# In[1394]:


X = stats_salary.iloc[:,3:13]  #independent columns
y = stats_salary.iloc[:,-1] 
# Splitting data into train and test


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(X_train,y_train)
print(f"Original Testing score: {knn.score(X_test, y_test)}")


# In[1395]:



params = {'n_neighbors':list(range(1,50))}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=10)
model.fit(X_train,y_train)
model.best_params_


# In[1396]:


neighbors = np.arange(1, 50)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
 
# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
     
    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
 
# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
 
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[1397]:


# use the best hyperparameters
knn = KNeighborsRegressor(n_neighbors = 22)
knn.fit(X_train, y_train)


print(f"Training score after hyperparameter tuning: {knn.score(X_train, y_train)}")


# In[1398]:


# use the best hyperparameters
knn2 = KNeighborsRegressor(n_neighbors = 22)
knn2.fit(X_test, y_test)
print(f"Testing score after hyperparameter tuning: {knn.score(X_test, y_test)}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




