#!/usr/bin/env python
# coding: utf-8

# Question 1

# In[66]:


import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
warnings.filterwarnings("ignore")


# In[67]:


data = pd.read_csv("train.csv")
data.head()


# In[38]:


le = preprocessing.LabelEncoder()
data['Sex'] = le.fit_transform(data.Sex.values)
data['Survived'].corr(data['Sex'])


# In[78]:


matrix = data.corr()
print(matrix)


# In[91]:


data.corr().style.background_gradient(cmap="Blues")


# In[79]:


sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='icefire')
plt.show()


# In[94]:


train_raw = pd.read_csv("train.csv")
test_raw = pd.read_csv("test.csv")

train_raw["train"] = 1
test_raw["train"] = 0
data = train_raw.append(test_raw, sort=False)

features = ['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp']
target = 'Survived'

data = data[features + [target] + ["train"]]
data["Sex"] = data["Sex"].replace(["female", "male"], [0, 1])
data["Embarked"] = data['Embarked'].replace(['S', 'C', 'Q'], [1, 2, 3])
train = data.query("train == 1")
test = data.query("train == 0")


# In[69]:


train.dropna(axis=0, inplace=True)
labels = train[target].values


train.drop(["train", target, "Pclass"], axis=1, inplace=True)
test.drop(["train", target, "Pclass"], axis=1, inplace=True)


# In[52]:


from sklearn.model_selection import train_test_split, cross_validate

X_train, X_val, Y_train, Y_val = train_test_split(train, labels, test_size=0.2, random_state=1)


# In[53]:


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")


# In[48]:


classifier = GaussianNB()

classifier.fit(X_train, Y_train)


# In[54]:


y_pred = classifier.predict(X_val)
print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_val, y_pred))


# Question 2

# In[57]:


glass=pd.read_csv("glass.csv")
glass


# In[58]:


glass.head()


# In[90]:


glass.corr().style.background_gradient(cmap="Greens")


# In[61]:


matrix = glass.corr()
print(matrix)


# In[76]:


sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()

features = ['Rl', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
target = 'Type'
X_train, X_val, Y_train, Y_val = train_test_split(glass[::-1], glass['Type'],test_size=0.2, random_state=1)
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_val)
print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_val, y_pred))
# In[74]:


from sklearn.svm import SVC, LinearSVC
classifier = LinearSVC()
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_val)
print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))

from sklearn.metrics import accuracy_score
print("accuracy is",accuracy_score(Y_val, y_pred))


# In[ ]:




