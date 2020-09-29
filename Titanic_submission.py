#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns             #Libraries needed for the Data pre-processing
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing


# In[51]:


train_df=pd.read_csv('F:/titanic/train.csv')
test_df=pd.read_csv('F:/titanic/test.csv')
train_df.head()


# In[57]:


train=train_df.drop(['Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Age'],axis=1)
test=test_df.drop(['Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Age'],axis=1)
train


# In[58]:


train.Sex[train.Sex == 'male'] = 1
train.Sex[train.Sex == 'female'] = 0
test.Sex[test.Sex == 'male'] = 1
test.Sex[test.Sex == 'female'] = 0
train


# In[59]:


test
#train.drop(['Age'],inplace=True)


# In[68]:


X_train=train[['PassengerId','Survived','Pclass','Sex']]
y_train=train['Survived']
X_train.drop(['Survived'],axis=1,inplace=True)
y_train


# In[71]:


from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(X_train,y_train)


# In[73]:


predictions=log.predict(test)


# In[75]:


predictions


# In[78]:


submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})
submission.head()
submission.to_csv('Titanic_predictions.csv',index=False)

