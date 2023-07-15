#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin


# # ELM model

# In[3]:


class ELM(BaseEstimator, RegressorMixin):
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid',
                loss='mse'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.loss=loss
        self.W = np.random.normal(size=(input_size, hidden_size))
        self.b = np.random.normal(size=(1, hidden_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def elu(self, x):
        alpha = 1.0  # ELU hyperparameter (adjustable)
        return np.where(x >= 0, x, alpha * (np.exp(x) - 1))
    
    def activation_function(self, x):
        if self.activation == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation == 'relu':
            return self.relu(x)
        elif self.activation == 'tanh':
            return self.tanh(x)
        elif self.activation == 'elu':
            return self.elu(x)
        else:
            raise ValueError('Invalid activation function.')
    def mean_squared_error(y, pred):
        return 0.5 * np.mean((y - pred) ** 2)


    def mean_abs_error(y, pred):
        return np.mean(np.abs(y, pred))

    def fit(self, X, y, epochs=50):
        H = self.activation_function(np.dot(X, self.W) + self.b)
        if self.loss == 'mse':
            self.beta = np.dot(np.linalg.pinv(H), y)
        elif self.loss == 'mae':
            self.beta = np.dot(np.linalg.pinv(H), y)
        else:
            raise ValueError('Invalid loss function.')
        
    def predict(self, X):
        H = self.activation_function(np.dot(X, self.W) + self.b)
        y_pred = np.dot(H, self.beta)
        y_pred_binary = np.where(y_pred >= 0.5, 1, 0)
        return y_pred_binary
    
    def get_params(self, deep=True):
        # Return parameter names and their values
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "activation": self.activation
        }

    def set_params(self, **parameters):
        # Set the value of parameters
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    #def score(self, X, y):
        #y_pred = self.predict(X)
        #accuracy = np.sum(y_pred==y)/len(y)
        #return accuracy


# # The Telco Churn dataset data cleaning

# In[4]:


df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn (3).csv")
df.sample(10)

#droping the extra column
df.drop('customerID',axis='columns',inplace=True)

#finding the type of the data(object or float or integer etc.)
df.dtypes

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

# scaling
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])
df2.sample(3)
for col in df2:
    print(f'{col}:{df2[col].unique()}')
    
    


# # Assignning the dependent and inpendent variable and trai test split

# In[5]:


x = df2.drop('Churn',axis='columns')
y = df2['Churn']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



# # Training the model with class imbalance

# In[6]:


pip install tensorflow


# # Train the ELM model

# In[7]:


input_size = x_train.shape[1]
hidden_size = 500
output_size = 2

elm = ELM(input_size, hidden_size, output_size, activation= 'relu')
elm.fit(x_train.values, y_train, epochs=100)


y_pred = elm.predict(x_test.values)
# Evaluate the ELM model on the testing set
#y_pred = pd.Series(y_pred)

#accuracy = elm.score(y_pred, y_test)
#print('Accuracy: {:.2f}%'.format(accuracy * 100))

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# true binary labels and predicted probabilities from a classifier


# calculate fpr, tpr, and thresholds for different probability cutoffs
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# calculate AUC score
auc_score = roc_auc_score(y_test, y_pred)

# plot the ROC curve
plt.plot(fpr, tpr, label=f'AUC={auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

yp = elm.predict(x_test)
yp[:5]
y_pred=[]
for elements in yp:
    if elements>0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
        
        
y_test[:10]
y_pred[:10]

from sklearn.metrics import confusion_matrix , classification_report
# compute the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# compute the observed proportion of agreement
po = (tp + tn) / (tp + tn + fp + fn)

# compute the expected proportion of agreement
pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (tp + tn + fp + fn) ** 2

# compute Cohen's kappa
kappa = (po - pe) / (1 - pe)

print(f"Cohen's kappa: {kappa}")

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
print(classification_report(y_test, y_pred))
import tensorflow as tf
import seaborn as sn 
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot= True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[8]:


input_size = x_train.shape[1]
hidden_size = 500
output_size = 2

elmhp = ELM(input_size=x_train.shape[1], hidden_size=10, output_size=2, activation="relu")
parameters = {"hidden_size":[10,20,30,50,70,80,90,100,500],
              "activation":["relu","tanh","sigmoid",'elu'],
              "output_size":[2] ,
              "input_size": [x_train.shape[1]],
              #"epochs":[50,100,200,500]
             }

from sklearn.model_selection import GridSearchCV
grid_search= GridSearchCV(elmhp, param_grid=parameters,n_jobs=-1,cv=5,verbose=3,scoring="accuracy")
grid_search.fit(x_train,y_train)
yd_pred_elmhp=grid_search.predict(x_test)
yp = grid_search.predict(x_test)
yp[:5]
y_pred_elmhp=[]
for elements in yp:
    if elements>0.5:
        y_pred_elmhp.append(1)
    else:
        y_pred_elmhp.append(0)
format(grid_search.score(x_train,y_train))
format(grid_search.score(x_test,y_test))
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# true binary labels and predicted probabilities from a classifier


# calculate fpr, tpr, and thresholds for different probability cutoffs
fpr, tpr, thresholds = roc_curve(y_test, y_pred_elmhp)

# calculate AUC score
auc_score = roc_auc_score(y_test, y_pred_elmhp)

# plot the ROC curve
plt.plot(fpr, tpr, label=f'AUC={auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

      
y_test[:10]
y_pred_elmhp[:10]

from sklearn.metrics import confusion_matrix , classification_report
# compute the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_elmhp).ravel()

# compute the observed proportion of agreement
po = (tp + tn) / (tp + tn + fp + fn)

# compute the expected proportion of agreement
pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (tp + tn + fp + fn) ** 2

# compute Cohen's kappa
kappa = (po - pe) / (1 - pe)

print(f"Cohen's kappa: {kappa}")

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
print(classification_report(y_test, y_pred_elmhp))
import tensorflow as tf
import seaborn as sn 
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot= True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[9]:


#pip install openblas


# # Training the model by down sampling using SMOTEENN

# In[11]:


# DOWNsampling SMOTEENN
from imblearn.combine import SMOTEENN
sm= SMOTEENN()
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV



# Split the dataset into training and testing sets
xd_train, xd_test, yd_train, yd_test = train_test_split(x, y, test_size=0.2, random_state=42)
xd_train,yd_train=sm.fit_resample(xd_train,yd_train)
# Train the ELM model
input_size = xd_train.shape[1]

output_size = 2

elmhp = ELM(input_size=xd_train.shape[1], hidden_size=10, output_size=2, activation="relu")
parameters = {"hidden_size":[10,20,30,50,70,80,90,100,500],
              "activation":["relu","tanh","sigmoid",'elu'],
              "output_size":[2] ,
              "input_size": [xd_train.shape[1]],
              #"epochs":[50,100,200,500]
             }

from sklearn.model_selection import GridSearchCV
grid_search= GridSearchCV(elmhp, param_grid=parameters,n_jobs=-1,cv=5,verbose=3,scoring="accuracy")
grid_search.fit(xd_train,yd_train)
yd_pred_elmhp=grid_search.predict(xd_test)
yp = grid_search.predict(xd_test)
yp[:5]
yd_pred_elmhp=[]
for elements in yp:
    if elements>0.5:
        yd_pred_elmhp.append(1)
    else:
        yd_pred_elmhp.append(0)
format(grid_search.score(xd_train,yd_train))
format(grid_search.score(xd_test,yd_test))





from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# true binary labels and predicted probabilities from a classifier


# calculate fpr, tpr, and thresholds for different probability cutoffs
fpr, tpr, thresholds = roc_curve(yd_test, yd_pred_elmhp)

# calculate AUC score
auc_score = roc_auc_score(yd_test, yd_pred_elmhp)

# plot the ROC curve
plt.plot(fpr, tpr, label=f'AUC={auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()



# find the threshold that maximizes the sum of sensitivity and specificity
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# print the optimal threshold value
print("Optimal threshold value:", optimal_threshold)

        
        
yd_test[:10]
yd_pred_elmhp[:10]

from sklearn.metrics import confusion_matrix , classification_report
# compute the confusion matrix
tn, fp, fn, tp = confusion_matrix(yd_test, yd_pred_elmhp).ravel()

# compute the observed proportion of agreement
po_d = (tp + tn) / (tp + tn + fp + fn)

# compute the expected proportion of agreement
pe_d = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (tp + tn + fp + fn) ** 2

# compute Cohen's kappa
kappa_d = (po_d - pe_d) / (1 - pe_d)

print(f"Cohen's kappa: {kappa_d}")

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
print(classification_report(yd_test, yd_pred_elmhp))
import tensorflow as tf
import seaborn as sn 
cm = tf.math.confusion_matrix(labels=yd_test,predictions=yd_pred_elmhp)
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot= True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')



# # Training the model by up sampling using SMOTETomek

# In[12]:


#Up-sampling
from  imblearn.combine import SMOTETomek
st=SMOTETomek()
x_r,y_r =st.fit_resample(x,y)

# train test split over Tomek
from sklearn.model_selection import train_test_split
xr_train, xr_test, yr_train, yr_test = train_test_split(x,y,test_size=0.2,random_state=(5))

xr_train,yr_train =st.fit_resample(xr_train,yr_train)
xr_train.shape
xr_test.shape
xr_train[:10]

# Train the ELM model
input_size = xr_train.shape[1]

output_size = 2

elmhp = ELM(input_size=xd_train.shape[1], hidden_size=10, output_size=2, activation="relu")
parameters = {"hidden_size":[10,20,30,50,70,80,90,100],
              "activation":["relu","tanh","sigmoid",'elu'],
              "output_size":[2] ,
              "input_size": [xr_train.shape[1]],
              #"epochs":[50,100,200,500]
             }

from sklearn.model_selection import GridSearchCV
grid_search= GridSearchCV(elmhp, param_grid=parameters,n_jobs=-1,cv=5,verbose=3,scoring="accuracy")
grid_search.fit(xr_train,yr_train)
yr_pred_elmhp=grid_search.predict(xr_test)
yp = grid_search.predict(xr_test)
yp[:5]
yr_pred_elmhp=[]
for elements in yp:
    if elements>0.5:
        yr_pred_elmhp.append(1)
    else:
        yr_pred_elmhp.append(0)
format(grid_search.score(xr_train,yr_train))
format(grid_search.score(xr_test,yr_test))

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# true binary labels and predicted probabilities from a classifier


# calculate fpr, tpr, and thresholds for different probability cutoffs
fpr, tpr, thresholds = roc_curve(yr_test, yr_pred_elmhp)

# calculate AUC score
auc_score = roc_auc_score(yr_test, yr_pred_elmhp)

# plot the ROC curve
plt.plot(fpr, tpr, label=f'AUC={auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


        
        
yr_test[:10]
yr_pred_elmhp[:10]

from sklearn.metrics import confusion_matrix , classification_report
# compute the confusion matrix
tn, fp, fn, tp = confusion_matrix(yr_test, yr_pred_elmhp).ravel()

# compute the observed proportion of agreement
po_r = (tp + tn) / (tp + tn + fp + fn)

# compute the expected proportion of agreement
pe_r = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (tp + tn + fp + fn) ** 2

# compute Cohen's kappa
kappa_r = (po_r - pe_r) / (1 - pe_r)

print(f"Cohen's kappa: {kappa_r}")

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
print(classification_report(yr_test, yr_pred_elmhp))
import tensorflow as tf
import seaborn as sn 
cm = tf.math.confusion_matrix(labels=yr_test,predictions=yr_pred_elmhp)
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot= True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




