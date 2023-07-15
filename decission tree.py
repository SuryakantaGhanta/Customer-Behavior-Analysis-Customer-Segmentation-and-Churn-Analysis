# -*- coding: utf-8 -*-
"""
Created on Thu May  4 09:59:11 2023

@author: SURJAKANTA
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the Telco Churn dataset from the UCI Machine Learning Repository
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn (3).csv")
df.sample(10)

#droping the extra column
df.drop('customerID',axis='columns',inplace=True)

#finding the type of the data(object or float or integer etc.)
df.dtypes
df_missing= df.isnull().sum(axis=0)
df_missing=df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()].shape
#values of a certain attribute
df.TotalCharges.values

# converting an attribute to numeric value and ignoring the errors comming due to spaces
pd.to_numeric(df.TotalCharges,errors='coerce').isnull()
df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()]

#shape of the missing data and the data set
df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()].shape
df.shape

#dropping the rows of the missing variables and forming a new data set
df1 = df[df.TotalCharges!=' ']
df1.shape
df1.dtypes

#forming the new data frame
df1.TotalCharges=pd.to_numeric(df1.TotalCharges)
df1.TotalCharges.dtypes
df1.sample(5)
df1.dtypes
#categorical data identification
def print_unique_col_values(df):    
    for col in df:
        if df[col].dtype=='object':
            print(f'{col}:{df[col].unique()}')
#replacing same variables in same name
print_unique_col_values(df1)
df1.replace('No internet service','No',inplace=True)
df1.replace('No phone service','No',inplace=True)
print_unique_col_values(df1)

#nominal data transformation
binary_col=['Partner', 'Dependents', 'PhoneService', 'MultipleLines','OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
            'StreamingMovies','PaperlessBilling', 'Churn']
for col in binary_col:
    df1[col].replace({'Yes': 1,'No': 0}, inplace = True)
#df1.replace({'Yes': 1,'No': 0}, inplace = True)
print_unique_col_values(df1)
for col in df1:
    print(f'{col}:{df1[col].unique()}')
    
#df1['gender'].replace({'Female': 1,'Male': 0}, inplace = True)
#for col in df1:
   # print(f'{col}:{df1[col].unique()}')
    
df2 = pd.get_dummies(data=df1,columns=['InternetService','Contract','PaymentMethod','gender'])
df2.columns    
df2.sample(4)
df2.dtypes
churners=sum(df2['Churn']==1)
non_churners= df2.shape[0]-churners
non_churners=(df2['Churn']==0).sum()
percentage_of_churners = (churners/df2.shape[0])*100

yearly_churn_rate = (df1["Churn"]==1).sum()/(df1.shape[0])*100
#monthly_churn_rate= (1-((1-yearly_churn_rate)**(1/12)))*100

total_revenue = df1['TotalCharges'].sum()
ARPC = total_revenue / df1.shape[0]
ACL = sum(df1['tenure']) / df1.shape[0]
CLV = (ARPC * ACL) /yearly_churn_rate

# Calculate the CLV
average_revenue_per_customer = df["MonthlyCharges"].mean()
average_yearly_churn_rate = df["Churn"].value_counts(normalize=True)["Yes"]
average_monthly_churn_rate= 1-((1-average_yearly_churn_rate)**(1/12))
average_customer_lifetime = 1 / average_monthly_churn_rate
clv = average_revenue_per_customer * average_customer_lifetime
total_money_loss=churners*clv
# Print the CLV
print("The average CLV is $", round(clv, 2))
# scaling
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])
df2.sample(3)
for col in df2:
    print(f'{col}:{df2[col].unique()}')
    
    
x = df2.drop('Churn',axis='columns')
y = df2['Churn']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

"""### Decission tree classifier ####"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
import matplotlib.pyplot as plt
dt = DecisionTreeClassifier(criterion= "gini")
scores = cross_val_score(dt,x_train,y_train,cv=10)
print(scores)
print(scores.mean())

DT= dt.fit(x_train,y_train)
y_pred_DT = DT.predict(x_test)

# plt.figure(figsize=(10, 6))
# tree.plot_tree(DT, feature_names=x.columns, class_names=[str(c) for c in DT.classes_], filled=True)
# plt.show()
format(DT.score(x_train,y_train))
format(DT.score(x_test,y_test))
from sklearn.metrics import confusion_matrix , classification_report
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
print(classification_report(y_test, y_pred_DT))
import tensorflow as tf
import seaborn as sn 
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred_DT)
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot= True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


"""### Decission tree classifier(hyper parametric tuning) ####"""
dthp = DecisionTreeClassifier()
parameters = {"splitter":["best","random"],
              "max_depth":[8,9,10,15,20,25,30],
              "min_weight_fraction_leaf":[0.1,0.4,0.5,0.6,0.8],
              "max_features":["auto","log2","sqrt",None],
              "max_leaf_nodes":[None,10,20,30,50,70,80,90,100]}

from sklearn.model_selection import GridSearchCV
grid_search= GridSearchCV(dthp, param_grid=parameters,n_jobs=-1,cv=50,verbose=3)
grid_search.fit(x_train,y_train)
y_pred_dthp=grid_search.predict(x_test)
format(grid_search.score(x_train,y_train))
format(grid_search.score(x_test,y_test))


from sklearn.metrics import confusion_matrix , classification_report
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
print(classification_report(y_test, y_pred_dthp))
import tensorflow as tf
import seaborn as sn 
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred_dthp)
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot= True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

# DOWNsampling SMOTEENN
from imblearn.combine import SMOTEENN
sm= SMOTEENN()
x_resampled,y_resampled=sm.fit_resample(x,y)


# Split the dataset into training and testing sets
xd_train, xd_test, yd_train, yd_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

"""### Decission tree classifier(hyper parametric tuning) ####"""
dthp = DecisionTreeClassifier()
parameters = {"splitter":["best","random"],
              "max_depth":[8,9,10,15,20,25,30],
              "min_weight_fraction_leaf":[0.1,0.4,0.5,0.6,0.8],
              "max_features":["auto","log2","sqrt",None],
              "max_leaf_nodes":[None,10,20,30,50,70,80,90,100]}

from sklearn.model_selection import GridSearchCV
grid_search= GridSearchCV(dthp, param_grid=parameters,n_jobs=-1,cv=50,verbose=3)
grid_search.fit(xd_train,yd_train)
yd_pred_dthp=grid_search.predict(xd_test)
format(grid_search.score(xd_train,yd_train))
format(grid_search.score(xd_test,yd_test))


from sklearn.metrics import confusion_matrix , classification_report
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
print(classification_report(yd_test, yd_pred_dthp))
import tensorflow as tf
import seaborn as sn 
cm = tf.math.confusion_matrix(labels=yd_test,predictions=yd_pred_dthp)
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot= True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
