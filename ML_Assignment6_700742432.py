#!/usr/bin/env python
# coding: utf-8

# In[59]:


#importing all libraries here for assignment
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")


# In[60]:


df = pd.read_csv('CC GENERAL.csv')
df.info()


# In[61]:


df.head()


# In[62]:


df.describe()


# In[63]:


df.isnull().sum()


# In[64]:


df1 = df.drop(['CUST_ID'], axis=1)
df1.head()


# In[65]:


df1.isnull().any()


# In[66]:


df1.fillna(df.mean(), inplace=True)
df1.isnull().any()


# In[67]:


df1.corr().style.background_gradient(cmap="Blues")


# In[68]:


x = df1.iloc[:,0:-1]
y = df1.iloc[:,-1]

scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled_df = pd.DataFrame(X_scaled_array, columns = x.columns)


# In[69]:


X_normalized = preprocessing.normalize(X_scaled_df)
X_normalized = pd.DataFrame(X_normalized)
X_normalized


# In[70]:


p = PCA(n_components=2)
principalComponents = p.fit_transform(X_normalized)

principalDf = pd.DataFrame(data = principalComponents, columns = ['PrincipalComponent1', 'PrincipalComponent2'])

finalDf = pd.concat([principalDf, df[['TENURE']]], axis = 1)
finalDf.head()


# In[77]:


plt.figure(figsize=(7,7))
plt.scatter(finalDf['PrincipalComponent1'],finalDf['PrincipalComponent2'],c=finalDf['TENURE'],cmap='prism', s =5)
plt.xlabel('PrincipalComponent1')
plt.ylabel('PrincipalComponent2')


# In[72]:


ac2 = AgglomerativeClustering(n_clusters = 2)
 
# Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(principalDf['PrincipalComponent1'], principalDf['PrincipalComponent2'],
           c = ac2.fit_predict(principalDf), cmap ='rainbow')
plt.show()


# In[73]:


ac3 = AgglomerativeClustering(n_clusters = 3)
 
# Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(principalDf['PrincipalComponent1'], principalDf['PrincipalComponent2'],
           c = ac3.fit_predict(principalDf), cmap ='rainbow')
plt.show()


# In[74]:


ac4 = AgglomerativeClustering(n_clusters = 4)
 
# Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(principalDf['PrincipalComponent1'], principalDf['PrincipalComponent2'],
           c = ac4.fit_predict(principalDf), cmap ='rainbow')
plt.show()


# In[75]:


ac5 = AgglomerativeClustering(n_clusters = 5)
 
# Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(principalDf['PrincipalComponent1'], principalDf['PrincipalComponent2'],
           c = ac5.fit_predict(principalDf), cmap ='rainbow')
plt.show()


# In[76]:


k = [2, 3, 4, 5]
 
# Appending the silhouette scores of the different models to the list
silhouette_scores = []
silhouette_scores.append(
        silhouette_score(principalDf, ac2.fit_predict(principalDf)))
silhouette_scores.append(
        silhouette_score(principalDf, ac3.fit_predict(principalDf)))
silhouette_scores.append(
        silhouette_score(principalDf, ac4.fit_predict(principalDf)))
silhouette_scores.append(
        silhouette_score(principalDf, ac5.fit_predict(principalDf)))

 
# Plotting a bar graph to compare the results
plt.bar(k, silhouette_scores)
plt.xlabel('Number of clusters', fontsize = 20)
plt.ylabel('S(i)', fontsize = 20)
plt.show()


# In[ ]:




