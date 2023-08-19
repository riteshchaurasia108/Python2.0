#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Algorithm

# In[76]:


# 1. Linear Regression
# 2. Polynomial regression.
# 3. SVM
# 4.DT
# 5.Bagging
# 6.Boosting
# 7. Random Forest

# FOLLOWING are Not part  of ML algorithm
# KNN 
# NB


# ## Linear Regression

# In[77]:


#  In this Dependet variable is numerical in regression


# In[78]:


#  0. identify it is supervised( it has dependent variable ) or unsupervised learning( it has independent variable )
# Steps to proceed/solve ML problems for supervised learning 
 # 1. identify problem statement ( either dependent or independent varible)
# 1.2  If Dependent variable then that is superviseed learning.
# 2. Either it is numerical or categorical { If Numerical then Regression proble, and categorical then it is classification problems}
# 3. and dependent varible is divided in to train and test data

# 4. x is denoted as independent variable
    #  y is denoted as dependent varibale ( only one dependt variable)


# In[79]:


# for straight line 
# y= mx+c
# y is dependent varible
# m is slope   m=delY/delX,    y2-y1/x2-x1
# c is intercept cut where the y axis


# In[80]:


# Linear Regression means having one independent and one dependent variable
# Multiple Regression means having more than one independent varible with one dependent variable
# Polynomial regression


# In[81]:


# Problem statement:- we will construct a linear model that  explain car's milage with other attributes


# In[82]:


# import libraries:

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[83]:


car_df=pd.read_csv("auto-mpg.csv")
car_df
# here mpg is dependent variable as it is number regression problem


# In[84]:


car_df.drop("car name",axis=1,inplace=True)


# In[85]:


car_df


# In[86]:


# replacing categorical data into actual value
car_df["origin"]=car_df["origin"].replace({1:"america",2:"africa",3:"asia"})
car_df


# In[87]:


car_df=pd.get_dummies(car_df,columns=["origin"])  # to write column in rows
car_df


# In[88]:


car_df.isnull().sum()


# In[89]:


car_df.dtypes

# here horse power is oject


# In[90]:


# to change horse power in integer
hpIsDigit=pd.DataFrame(car_df.horsepower.str.isdigit())
hpIsDigit


# In[91]:


# And now print isDigit as False!
car_df[hpIsDigit['horsepower']==False]


# In[92]:


# replace ? to NaN valuue

car_df['horsepower']=car_df['horsepower'].replace('?',np.nan)
car_df['horsepower']=car_df['horsepower'].astype(float)


# In[93]:


# object can be channge by median  , here mid value is given to all the data

median1=car_df['horsepower'].median()
median1


# In[94]:


car_df['horsepower'].replace(np.nan,median1,inplace=True)


# In[95]:


car_df[hpIsDigit['horsepower']==False]


# In[96]:


car_df.dtypes

# finnaly horsepower is changed in to ffloat


# In[97]:


duplicate=car_df.duplicated()
duplicate.sum()


# ## BiVariate Plots

# In[98]:


## A bivariate analysis among the different variable can be  using scatter plot. The result acn be stored as a .png file


# In[99]:


sns.pairplot(car_df,diag_kind="kde") # it take time to run 


# In[100]:


# Split Data


# In[101]:


# lets build our linear model
# independant variables
X=car_df.drop(['mpg'],axis=1) # here except mpg all are independent variable , so deleted mpg 
# the dependent variable
y=car_df[['mpg']] # always inn 2D 


# In[102]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1)


# In[103]:


model_1=LinearRegression()
model_1.fit(X_train,y_train) # it also take time to run


# In[104]:


# Here are the coeeficients for each variable and the interrceopt 
# The score (R^2) for in sample and out of sample


# In[105]:


model_1.score(X_train,y_train)


# In[ ]:





# In[106]:


# complerte 6th lession at the end works on concrete data set due to unavailable of  dataset I dont do that


# In[ ]:





# In[ ]:





# In[107]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

#  degree may change to know how our model is performing(train and test)
poly=PolynomialFeatures(degree=3,interaction_only=True)
X_train2=poly.fit_transform(X_train)
X_test2=poly.fit_transform(X_test)

poly_clf=linear_model.LinearRegression()
poly_clf.fit(X_train2,y_train)

# y_pred=poly_clf.predict(X_test2)
# print(y_pred)
# In sample (training) R^2  wll always improve  with the number of variablle
print(poly_clf.score(X_train2,y_train))


# In[108]:


# out off sample(testing) R^2 is our measure of succes and does improve
print(poly_clf.score(X_test2,y_test))


# In[114]:





# In[115]:


dg


# In[ ]:




