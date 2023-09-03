#!/usr/bin/env python
# coding: utf-8

# # import librabies

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


# # load the data

# In[2]:


ad=pd.read_csv('advertising .csv')


# In[3]:


ad.head()


# 
# # data inspection
# 

# In[5]:


ad.shape


# In[6]:


ad.info()


# In[8]:


ad.describe()


# # Data Cleaning

# In[11]:


#checking null values in a dataset
ad.isnull().sum()*100/ad.shape[0]
#there is no null values in a dataset ,hence there is clean


# In[12]:


# Outlier Analysis
fig, axs = plt.subplots(3, figsize = (5,5))
plt1 = sns.boxplot(ad['TV'], ax = axs[0])
plt2 = sns.boxplot(ad['Newspaper'], ax = axs[1])
plt3 = sns.boxplot(ad['Radio'], ax = axs[2])
plt.tight_layout()


# In[15]:


# there is no outlier present in a dataset


# # explorating  data analysis
# 
# 

# # Univariate Analysis

# # sales(target variable)

# In[16]:


sns.boxplot(ad['Sales'])
plt.show()


# In[17]:


# Let's see how Sales are related with other variables using scatter plot.
sns.pairplot(ad, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()


# In[18]:


# Let's see the correlation between different variables.
sns.heatmap(ad.corr(), cmap="YlGnBu", annot = True)
plt.show()


# In[19]:


#As is visible from the pairplot and the heatmap, the variable TV seems to be most correlated with Sales. So let's go ahead and perform simple linear regression using TV as our feature variable.


# # Model Building
#  

# # Performing Simple Linear Regression

# In[22]:


#Equation of linear regression
#y=c+m1x1+m2x2+...+mnxn
 
#y
  #is the response
#c
 # is the intercept
#m1
 # is the coefficient for the first feature
#mn
 # is the coefficient for the nth feature
       
#In our case:

#y=c+m1×TV
 
#The  m
  #values are called the model coefficients or model parameters.


# In[23]:


X = ad['TV']
y = ad['Sales']


# In[25]:


#You now need to split our variable into training and testing sets. You'll perform this by importing train_test_split from the sklearn.model_selection library. It is usually a good practice to keep 70% of the data in your train dataset and the rest 30% in your test dataset


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[28]:


X_train.head()


# In[29]:


y_train.head()


# # Buliding a model

# In[30]:



#You first need to import the statsmodel.api library using which you'll perform the linear regression.


# In[31]:


import statsmodels.api as sm


# In[32]:


# Add a constant to get an intercept
X_train_sm = sm.add_constant(X_train)


# In[33]:


# Fit the resgression line using 'OLS'
lr = sm.OLS(y_train, X_train_sm).fit()


# In[34]:


# Print the parameters, i.e. the intercept and the slope of the regression line fitted
lr.params


# In[35]:


# Performing a summary operation lists out all the different parameters of the regression line fitted
print(lr.summary())


# # Looking at some key statistics from the summary

# In[36]:


#The values we are concerned with are -

#The coefficients and significance (p-values)
#R-squared
#F statistic and its significance
#1. The coefficient for TV is 0.054, with a very low p value
#The coefficient is statistically significant. So the association is not purely by chance.

#2. R - squared is 0.816
#Meaning that 81.6% of the variance in Sales is explained by TV

#This is a decent R-squared value.

#3. F statistic has a very low p value (practically low)
#Meaning that the model fit is statistically significant, and the explained variance isn't purely by chance.


# In[38]:


#The fit is significant. Let's visualize how well the model fit the data.

#From the parameters that we get, our linear regression equation becomes:

#Sales=6.948+0.054×TV


# In[39]:


plt.scatter(X_train, y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r')
plt.show()


# # Model Evaluation

# In[40]:


#Residual analysis
#To validate assumptions of the model, and hence the reliability for inference


# In[41]:


##Distribution of the error terms
##We need to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of the error terms and see what it looks like.


# In[42]:


y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)


# In[43]:


fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()


# In[44]:


#The residuals are following the normally distributed with a mean 0. All good!


# In[45]:


#Looking for patterns in the residuals


# In[46]:


plt.scatter(X_train,res)
plt.show()


# In[47]:


#We are confident that the model fit isn't by chance, and has decent predictive power. The normality of residual terms allows some inference on the coefficients.

#Although, the variance of residuals increasing with X indicates that there is significant variation that this model is unable to explain.

#As you can see, the regression line is a pretty good fit to the data


# # Predictions on the Test Set

# In[48]:


#Now that you have fitted a regression line on your train dataset, it's time to make some predictions on the test data. For this, you first need to add a constant to the X_test data like you did for X_train and then you can simply go on and predict the y values corresponding to X_test using the predict attribute of the fitted regression line.


# In[49]:


# Add a constant to X_test
X_test_sm = sm.add_constant(X_test)


# In[50]:


# Predict the y values corresponding to X_test_sm
y_pred = lr.predict(X_test_sm)


# In[51]:


y_pred.head()


# In[52]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[53]:


#Looking at the RMSE


# In[54]:


#Returns the mean squared error; we'll take a square root
np.sqrt(mean_squared_error(y_test, y_pred))


# # Checking the R-squared on the test set

# In[55]:


r_squared = r2_score(y_test, y_pred)
r_squared


# In[56]:


##Visualizing the fit on the testset


# In[57]:


plt.scatter(X_test, y_test)
plt.plot(X_test, 6.948 + 0.054 * X_test, 'r')
plt.show()


# In[ ]:




