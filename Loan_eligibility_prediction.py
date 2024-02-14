#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().system('pip install xgboost')
import xgboost as xgb
from xgboost import XGBClassifier


# In[2]:


training_data= pd.read_csv('loan_train_csv.csv')


# In[3]:


print(training_data.shape)


# In[4]:


training_data.head()


# In[5]:


#treating null values


# In[6]:


#getting the sum of null values in each column


# In[7]:


nulls_sum= training_data.isnull().sum().sort_values(ascending=False)


# In[8]:


nulls_sum


# In[9]:


# Remove extra spaces in column names
training_data.columns = training_data.columns.str.strip()

# Handling missing values
training_data['Gender'] = training_data['Gender'].fillna(training_data['Gender'].dropna().mode().values[0])
training_data['Married'] = training_data['Married'].fillna(training_data['Married'].dropna().mode().values[0])
training_data['Dependents'] = training_data['Dependents'].fillna(training_data['Dependents'].dropna().mode().values[0])
training_data['Self_Employed'] = training_data['Self_Employed'].fillna(training_data['Self_Employed'].dropna().mode().values[0])
training_data['LoanAmount'] = training_data['LoanAmount'].fillna(training_data['LoanAmount'].dropna().mean())
training_data['Loan_Amount_Term'] = training_data['Loan_Amount_Term'].fillna(training_data['Loan_Amount_Term'].dropna().mode().values[0])
training_data['Credit_History'] = training_data['Credit_History'].fillna(training_data['Credit_History'].dropna().mode().values[0])


# In[10]:


training_data.describe()


# In[11]:


training_data.info()


# In[12]:


#handling categorical variables
#PRINTING ALL UNIQUE VALUES
print(set(training_data["Gender"]))
print(set(training_data['Dependents']))
print(set(training_data["Married"]))
print(set(training_data['Education']))
print(set(training_data['Self_Employed']))
print(set(training_data['Loan_Status']))
print(set(training_data['Property_Area']))


# In[13]:


training_data['Loan_Status'] = training_data['Loan_Status'].map({'N' :0, 'Y' :1}).astype(int)
training_data = pd.get_dummies(training_data, columns = ['Gender', 'Dependents', 'Married', 'Education','Self_Employed', 'Property_Area'])
#scaling numerical features
scaler = StandardScaler()
columns_to_scale = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term' ]
training_data[columns_to_scale] = scaler.fit_transform(training_data[columns_to_scale])


# In[14]:





# Check for infinite or missing values in the entire DataFrame
if training_data.isin([np.inf, -np.inf]).any().any() or training_data.isnull().any().any():
    print("There are infinite or missing values in the dataset.")
else:
    print("There are no infinite or missing values in the dataset.")


# In[15]:


#split dataframe into x and y and drop irrelevant columns of loan status and loan id

y = training_data[ 'Loan_Status']
x= training_data.drop(['Loan_Status', 'Loan_ID'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
# X_train. shape, X_test.shape: (491, 20), (123, 20)


# In[16]:


y


# In[17]:


x


# In[18]:


#hyperparam tuning
gbm_param_grid = {
    'n_estimators': range(1, 1000, 10),
    'max_depth': range(1, 20),
    'learning_rate': [0.1, 0.4, 0.45, 0.5, 0.55, 0.61],  
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],  
}

xgb_classifier = XGBClassifier()
xgb_random = RandomizedSearchCV(
    param_distributions=gbm_param_grid,
    estimator=xgb_classifier,
    scoring="accuracy",
    verbose=0,
    n_iter=100,
    cv=4
)

xgb_random.fit(X_train, y_train)
print(f'Best parameters: {xgb_random.best_params_}')
y_pred = xgb_random.predict(X_test)
print(f'Accuracy: {np.sum(y_pred == y_test) / len(y_test)}')


# In[ ]:




