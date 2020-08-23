#!/usr/bin/env python
# coding: utf-8

# # [Shanshan, Lin]¶
# # [2020-08-08]

# # Answer to Question [7], Part [2.a]

# In[1]:


import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns 
import tkinter as tk

import re
import string
import unicodedata


import pandas_profiling
import itertools
import scipy


import nltk
from nltk.corpus import stopwordsl
from nltk import word_tokenize
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

from bs4 import BeautifulSoup

import os

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# Read in Data:
# 

# In[2]:



df = pd.read_csv("OJ.csv")


list(df)
df.info()
df.shape
df.describe().transpose()
df.head()
df.tail()


# In[3]:


df.rename(columns={'Unnamed: 0':'Trans_ID'}, inplace=True)   #to give the 1st column with a title "Trans_ID"


# In[4]:


df.info()


# In[5]:


###Profile the Data

pandas_profiling.ProfileReport(df, check_correlation=False)


# EDA

# In[6]:


#check if any missing value:

df.isnull().sum()   
#this confirms there is no missing value from the dataset, which was told by pandas_profiling.ProfileReport earlier as well


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
color = sns.color_palette()


plt.figure(figsize=(15,8));
StoreID =df['StoreID'].value_counts();
sns.barplot(StoreID.index,StoreID.values);

plt.xticks(rotation='vertical');

plt.xlabel("StoreID");
plt.ylabel('Number of Purchases');
plt.title("Top Store in where more purchases occur");
plt.show();
#StoreID#7 is in the top store has the most sales


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
color = sns.color_palette()


plt.figure(figsize=(15,8));
Brand_Purchased =df['Purchase'].value_counts();
sns.barplot(Brand_Purchased.index,Brand_Purchased.values);
plt.xticks(rotation='vertical');

plt.xlabel("Brand_Purchased");
plt.ylabel('Number of Purchases');
plt.title("Frquency of purchases on each brand");
plt.show();
#CH has more sales than MM in our dataset

df['Purchase'].value_counts()  #we see CH has 653 observations (61.028%)) while MM has 417 (38.972%)


# In[9]:


#convert CH to 0 and MM to 1 from the target "Purchase" column for binary classification purpose later:

df['Purchase']=df.Purchase.replace(to_replace=['CH', 'MM'], value=[0, 1])

df.Purchase.head()

df.info()


# In[10]:


##feature selection by filter approach:
#####as per the correlations metric plots from pandas_profiling.ProfileReport, and by using business intuition, there are some features are highly correlatives to each other and we decide to remove 3 of them as per following:


#drop the feature "PctDiscMM" for modeling building use later, as it gives almost the same info as feature "DiscMM":
del df['PctDiscMM']


# In[11]:


#drop the feature "PctDiscCH" for modeling building use later, as it gives almost the same info as feature "DiscCH":
del df['PctDiscCH']


# In[12]:


#drop the features "STORE" & "Store7" for modeling building use later, as another feature "StoreID" already provide info where purchase made from:
del df['STORE']
del df['Store7']


# In[13]:


df.head(5)


# Encoding

# In[14]:


df.info()

#convert "StoreID", "SpecialCH" and "SepcialMM" from numeric to string, so they can be passed to one-hot encoding later:
df['StoreID'] = df['StoreID'].astype(str)
df['SpecialCH'] = df['SpecialCH'].astype(str)
df['SpecialMM'] = df['SpecialMM'].astype(str)

df.info()

#concert to dummy-variable format via one-hot encoding:
cols = ['StoreID', 'SpecialCH', 'SpecialMM']

df = pd.concat([df, pd.get_dummies(df[cols])],axis=1)
df = df.drop(cols, axis=1) 
df.head()

list(df)


# # Answer to Question [7], Part [2.b]¶

# Train Test Split

# In[15]:


#split df into train (80%) and test (20%) sets:  the test set these are unseen oberservations (protend these are future real life data) for the model in order to test our model's prediction power when comapring to the actual test set value:

#first we define the features under x as independent variables, and also set y=depedent target variable="Purchase":
y = df['Purchase']
X = df.drop(['Purchase',"Trans_ID"], axis=1)

np.bincount(y) #to show the break down of y target column by 2 classes (CH=653, MM=417)


feature_names = X.columns


#we use train_test_split function inside scikit-learn package to randomly split the dataset into random train and test subsets (80-20 split defined by test_size), random_state=30 for reproducible output
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)


X_train.head(5)


X_test.head(5)


# In[16]:


X_train.iloc[:,:10]


# In[17]:


#then we standardize x so that all features in the same scale:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train.iloc[:,:10] = scaler.fit_transform(X_train.iloc[:,:10])  #exclude those encoding features
X_test.iloc[:,:10] = scaler.transform(X_test.iloc[:,:10])  #exclude those encoding features


# In[18]:


X_train.head(5)
X_test.head(5)


# Handle Data Imbalancing using SMOTE

# In[19]:


from imblearn.over_sampling import SMOTE

X_resampled,y_resampled=SMOTE(random_state=20).fit_resample(X_train,y_train)

X_resampled.shape  #now our balanced_train x is called "X_resampled"
y_resampled.shape #now our balanced_train y is called "y_resampled"
np.bincount(y_resampled)  #now dataset is balanced, have both 2 classess equal to 528


# # Answer to Question [7], Part [2.c]

# Model Building

# In[20]:


def cv_results_to_df(cv_results):
    results = pd.DataFrame(list(cv_results['params']))
    results['mean_fit_time'] = cv_results['mean_fit_time']
    results['mean_score_time'] = cv_results['mean_score_time']
    results['mean_train_score'] = cv_results['mean_train_score']
    results['std_train_score'] = cv_results['std_train_score']
    results['mean_test_score'] = cv_results['mean_test_score']
    results['std_test_score'] = cv_results['std_test_score']
    results['rank_test_score'] = cv_results['rank_test_score']

    results = results.sort_values(['mean_test_score'], ascending=False)
    return results


# In[21]:


from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss, confusion_matrix

def evaluate_with_dt(X_train, X_test, y_train, y_test):
    clf_full = DecisionTreeClassifier(random_state=0)
    clf_full.fit(X_train, y_train)
    y_pred_dt = clf_full.predict(X_test)
    print ("\n Confusion Matrix : \n", confusion_matrix(y_test, y_pred_dt))
    print("Accuracy = {:.2f}".format(accuracy_score(y_test, y_pred_dt)))
    print("F1 Score = {:.2f}".format(f1_score(y_test, y_pred_dt)))
    print("Kappa = {:.2f}".format(cohen_kappa_score(y_test, y_pred_dt)))
    print("Log Loss = {:.2f}".format(log_loss(y_test, y_pred_dt)))


# # Decision Tree:

# In[22]:


#here we try "Grid Search" apporach to conduct Hyperparameter Tuning:

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



###first hyperparameter tuning using "Grid Search" apporach:
from sklearn.model_selection import GridSearchCV

#List Hyperparameters that we want to tune.
hyperparameters_dt= {'max_depth':[2,5,8,11,14],'max_features':['auto',None],'min_samples_split':[2,5,10,30,50],'criterion':['gini','entropy']}
#max_depth: The maximum depth of the tree
#max_features: The number of features to consider when looking for the best split
#min_Samples_Split: The minimum number of samples required to split an internal node
#criterion: The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.




#Create Decision Tree object
from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier(random_state=40)

#Use GridSearch
clf = GridSearchCV(clf_dt, hyperparameters_dt, cv=10)

#Fit the model
best_model = clf.fit(X_resampled,y_resampled)

#Print The value of best Hyperparameters
print('Best max_depth:', best_model.best_estimator_.get_params()['max_depth'])
print('Best max_features:', best_model.best_estimator_.get_params()['max_features'])
print('Best min_samples_split:', best_model.best_estimator_.get_params()['min_samples_split'])
print('Best criterion:', best_model.best_estimator_.get_params()['criterion'])

#this tell us the optimal combination is: max_depth=8, max_features='auto', min_samples_split=2, criterion=entropy


# In[23]:


#use the optimal hyperparameter combination suggested above to run decision tree model on balanced training set:
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=40, criterion="entropy", min_samples_split=2, max_depth=8, max_features="auto")

clf.fit(X_resampled, y_resampled)

y_pred_dt = clf.predict(X_test)


class_names = [str(x) for x in clf.classes_]


# In[24]:


clf.feature_importances_


# In[25]:


imp = clf.tree_.compute_feature_importances(normalize=False)
ind = sorted(range(len(imp)), key=lambda i: imp[i])[:] 
#sort in ascending order, we see that the most correlated feature on "Purchase" is "LoyalCH"

imp[ind]
feature_names[ind]


# In[26]:


from sklearn.metrics import confusion_matrix, accuracy_score

confusion_matrix(y_test, y_pred_dt) #display the confusion matrix


# In[27]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_dt, target_names=class_names)) #display the key measurement matrix 

accuracy_score(y_test,y_pred_dt)  #to display the accuracy rate


# In[28]:


#display the AUC value for our decision tree model:

from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_dt)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[29]:


#display the decision tree:

from sklearn.tree import plot_tree

plt.figure(figsize=[20,10]);
plot_tree(clf, filled=True, feature_names = feature_names, label='root', fontsize=5) #yet this is pretty hard to read....
plt.show();


# 
# # KNN:

# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

###first hyperparameter tuning using "Grid Search" apporach:
from sklearn.model_selection import GridSearchCV

#List Hyperparameters that we want to tune.
hyperparameters_knn= {'n_neighbors':[1,5,10,15,20],'p':[1,2]} 
#n_neighbors: number of neighbors to use by default for kneighbors queries.
#p: Power parameter for the Minkowski metric; p = 1 to be Manhattan distance and p = 2 to be Euclidean.

#Create KNN object
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier()

#Use GridSearch
clf = GridSearchCV(knn_classifier, hyperparameters_knn, cv=10)

#Fit the model
best_model = clf.fit(X_resampled,y_resampled)

#Print The value of best Hyperparameters
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

#this tell us the optimal combination is: p: 2, n_neighbors: 10


# In[31]:


##Now we will run our knn model using the optimal combination suggested above:
knn_clf = KNeighborsClassifier(n_neighbors=10, p=2)

knn_clf.fit(X_resampled, y_resampled)

y_pred_knn = knn_clf.predict(X_test)

print(accuracy_score(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_knn)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# # Gradient Boosting:

# In[32]:


from sklearn.ensemble import GradientBoostingClassifier

#List Hyperparameters that we want to tune, we use "grid search" approach here:

learning_rate = [0.15,0.01,0.001] # This determines the impact of each tree on the final outcome, note that lower values would require higher number of trees to model all the relations and will be computationally expensive.
n_estimators=[100,500,1000,1500]  #The number of sequential trees to be modeled
max_depth=[3,4,5,6] #maximum depth of the individual regression estimators

#Convert to dictionary
hyperparameters_gboost = dict(learning_rate=learning_rate, n_estimators=n_estimators,max_depth=max_depth)


gboost_clf = GradientBoostingClassifier()
clf = GridSearchCV(gboost_clf, hyperparameters_gboost, cv=10)     
     

best_model = clf.fit(X_resampled,y_resampled)

#Print The value of best Hyperparameters
print('Best learning_rate:', best_model.best_estimator_.get_params()['learning_rate'])
print('Best n_estimators:', best_model.best_estimator_.get_params()['n_estimators'])
print('Best max_depth:', best_model.best_estimator_.get_params()['max_depth'])

#this tell us the optimal combination is: learning_rate=0.01, n_estimators=1000, max_depth=3


# In[33]:


##Now we will run our gredient boosting model using the optimal combination suggested above:
gboost_clf = GradientBoostingClassifier(random_state=0, learning_rate=0.01, n_estimators=1000, max_depth=3)

gboost_clf.fit(X_resampled, y_resampled)

y_pred_gboost = gboost_clf.predict(X_test)

print(accuracy_score(y_test, y_pred_gboost))
print(confusion_matrix(y_test, y_pred_gboost))
print(classification_report(y_test, y_pred_gboost))


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_gboost)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc   

#as a result, comparing to decision tree and KNN, gredient boosting has the best performance among all 3 models#

