#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Some parts in lecltuure 6th for polynomial regresssioon


# # Titanic data set

# In[8]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# In[9]:


dg=pd.read_csv("titanic-training-data.csv")
dg.drop("PassengerId",axis=1,inplace=True)


# In[10]:


dg # here survived column is dependent variable , and it looks like a number, 0 means survive and 1 means not survive

# survival is classification not a regression


# In[12]:


dg.info()


# In[15]:


dg.isnull().sum()
# missing value in the data set


# In[16]:


dg.dtypes


# In[18]:


sns.countplot(x="Survived",data=dg)


# In[19]:


pd.crosstab(dg["Survived"],dg["Sex"])


# In[21]:


sns.countplot(x="Survived",hue="Sex",data=dg)


# In[23]:


sns.countplot(x="Survived",hue="Pclass",data=dg)


# In[24]:


sns.boxplot(x="Pclass",y="Age",data=dg)


# In[26]:


dg.isnull().sum()


# In[27]:


dg.drop("Cabin",axis=1,inplace=True)


# In[29]:


dg.head()


# In[30]:


dg.dropna(inplace=True)


# In[31]:


dg.isnull().sum()


# In[34]:


dg.shape


# In[35]:


dg=pd.get_dummies(dg,columns=["Sex","Pclass","Embarked"])


# In[37]:


dg.head()


# In[39]:


dg=dg.drop(["Name","Ticket","Fare"],axis=1)


# In[41]:


dg.info()


# In[42]:


x=dg.drop("Survived",axis=1)
y=dg["Survived"]


# In[44]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=1)


# In[45]:


model=LogisticRegression()


# In[46]:


model.fit(x_train,y_train)


# In[47]:


model.score(x_train,y_train)


# In[48]:


model.score(x_test,y_test)


# In[49]:


predictions=model.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test,predictions)

