#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


bike = pd.read_csv('AdvWorksCusts22.csv')
tst = pd.read_csv('AW_test.csv')

print(bike.shape)
bike.head()
bike.columns


# In[3]:


bike_counts = bike[['Occupation','BikeBuyer']].groupby('Occupation').count()
print(bike_counts)


# In[ ]:





# In[4]:


dummies = pd.get_dummies(bike.Education)
dummies1 = pd.get_dummies(bike.Occupation)
dummies2 = pd.get_dummies(bike.Gender)
dummies3 = pd.get_dummies(bike.MaritalStatus)


# In[5]:


dummy = pd.get_dummies(tst.Education)
dummy1 = pd.get_dummies(tst.Occupation)
dummy2 = pd.get_dummies(tst.Gender)
dummy3 = pd.get_dummies(tst.MaritalStatus)


# In[6]:


merged = pd.concat([bike,dummies],axis ='columns')
merged1 = pd.concat([merged,dummies1],axis ='columns')
merged2 = pd.concat([merged1,dummies2],axis ='columns')
merged3 = pd.concat([merged2,dummies3],axis ='columns')
merged3.columns


# In[7]:


mrg = pd.concat([tst,dummy],axis ='columns')
mrg1 = pd.concat([mrg,dummy1],axis ='columns')
mrg2 = pd.concat([mrg1,dummy2],axis ='columns')
mrg3 = pd.concat([mrg2,dummy3],axis ='columns')


# In[8]:


final = merged3.drop(['Education','Occupation','Gender','MaritalStatus'
                      , 'Unnamed: 26'
                      ,'Partial High School','Manual','F','S'], axis='columns')

final.columns


# In[9]:


final2 = mrg3.drop(['Education','Occupation','Gender','MaritalStatus'
                      ,'Partial High School','Manual','F','S'], axis='columns')

final2.columns


# In[10]:


x = final.iloc[:,14:31].values
y = final.iloc[:,21].values


# In[11]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


# In[12]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[13]:


x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[14]:


x_train


# In[15]:


x_test


# In[16]:


from sklearn.linear_model import LinearRegression
linmod = linear_model.LinearRegression()
linmod.fit(x_train,y_train)


# In[17]:


y_pred = linmod.predict(x_test)


# In[18]:


y_pred


# In[19]:


from sklearn import metrics
def print_metrics(y_true, y_predicted, n_parameters):
    ## First compute R^2 and the adjusted R^2
    r2 = sklm.r2_score(y_true, y_predicted)
    r2_adj = r2 - (n_parameters - 1)/(y_true.shape[0] - n_parameters) * (1 - r2)
    
    ## Print the usual metrics and the R^2 values
    print('Mean Square Error      = ' + str(sklm.mean_squared_error(y_true, y_predicted)))
    print('Root Mean Square Error = ' + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))
    print('Mean Absolute Error    = ' + str(sklm.mean_absolute_error(y_true, y_predicted)))
    print('Median Absolute Error  = ' + str(sklm.median_absolute_error(y_true, y_predicted)))
    print('R^2                    = ' + str(r2))
    print('Adjusted R^2           = ' + str(r2_adj))
   
y_score = linmod.predict(x_test) 
print_metrics(y_test, y_score, 28)    


# In[20]:


def hist_resids(y_test, y_score):
    ## first compute vector of residuals. 
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    ## now make the residual plots
    sns.distplot(resids)
    plt.title('Histogram of residuals')
    plt.xlabel('Residual value')
    plt.ylabel('count')
    
hist_resids(y_test, y_score) 


# In[21]:


import scipy.stats as ss
def resid_qq(y_test, y_score):
    ## first compute vector of residuals. 
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    ## now make the residual plots
    ss.probplot(resids.flatten(), plot = plt)
    plt.title('Residuals vs. predicted values')
    plt.xlabel('Predicted values')
    plt.ylabel('Residual')
    
resid_qq(y_test, y_score)  


# In[22]:


def resid_plot(y_test, y_score):
    ## first compute vector of residuals. 
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    ## now make the residual plots
    sns.regplot(y_score, resids, fit_reg=False)
    plt.title('Residuals vs. predicted values')
    plt.xlabel('Predicted values')
    plt.ylabel('Residual')

resid_plot(y_test, y_score) 


# In[23]:


y_score_untransform = np.exp(y_score)
y_test_untransform = np.exp(y_test)
resid_plot(y_test_untransform, y_score_untransform) 


# In[24]:


final.shape

final


# In[25]:


final2


# In[26]:


import pickle
filename = 'finalized_model.sav'
pickle.dump(linmod,open(filename,'wb'))

final2


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[27]:


from sklearn import tree



desired_factors = ['HomeOwnerFlag', 'NumberCarsOwned', 'NumberChildrenAtHome',
       'TotalChildren', 'YearlyIncome', 'Age', 'Bachelors ', 'Graduate Degree',
       'High School', 'Partial College', 'Clerical', 'Management',
       'Professional', 'Skilled Manual', 'M', 'M','BikeBuyer']

train_data = final[desired_factors]
test_data = final2[desired_factors]
target = final.AveMonthSpend

linmod.fit(train_data, target)
z_test = linmod.predict(test_data)


# In[28]:


df = pd.DataFrame(z_test)
df.to_csv("foopaaty.csv")


# In[ ]:





# In[ ]:




