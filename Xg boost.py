# -*- coding: utf-8 -*-
"""
Created on Wed May  3 19:05:49 2023

@author: SURJAKANTA
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#from sklearn.metrics import f1_score

# Load the Telco Churn dataset from the UCI Machine Learning Repository
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
    
    
x = df2.drop('Churn',axis='columns')
y = df2['Churn']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)




""" XGB"""

from xgboost import XGBClassifier
model_xgb=XGBClassifier()
model_xgb.fit(x_train,y_train)
y_pred_xgb=model_xgb.predict(x_test)

format(model_xgb.score(x_train,y_train))
format(model_xgb.score(x_test,y_test))


from sklearn.metrics import classification_report
print(classification_report(y_pred_xgb,y_test))


#Labels= ["ckd" , "not ckd"]
import seaborn as sns
conf_matrix_xgb= confusion_matrix(y_test , y_pred_xgb)
plt.figure(figsize=(2,2))
sns.heatmap(conf_matrix_xgb,annot=True,fmt="d",cmap="cividis")
plt.title("Confusion matrix for XGBoost ",color="r",fontsize=15)
plt.ylabel("True class")
plt.xlabel("Predicted class")
plt.show()


acc_XGB= accuracy_score(y_test,y_pred_xgb)
prec_XGB= precision_score(y_test,y_pred_xgb,average='micro')
rec_XGB= recall_score(y_test,y_pred_xgb,average='micro')
#f1_score_XGB= f1_score(y_test,y_pred_xgb,average='micro')

print("acc_XGB: ", acc_XGB)
print("prec_XGB: ", prec_XGB)
print("rec_XGB: ", rec_XGB)
#print("f1_score_XGB: ", f1_score_XGB)

"""Hyper Parameter Tunning for XG Bosst"""

from sklearn.model_selection import RandomizedSearchCV

#Number of trees in a random forest 
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]
#Various learning rate parameters 
learning_rate=['0.05','0.1','0.2','0.3','0.5','0.6']
#Maximum number of lavels in tree
max_depth=[int(x) for x in np.linspace(5,30,num=6)]
#max_depth.append(none)
#Subsample parameter value
subsample=[0.7,0.6,0.8]
##Minimum child weight parameters 
min_child_weight=[3,4,5,6,7]


random_grid={'n_estimators':n_estimators,
           'learning_rate':learning_rate,
           'max_depth':max_depth,
           'subsample':subsample,
           'min_child_weight':min_child_weight}
print(random_grid)

model_xgb=XGBClassifier()
xgb_random=RandomizedSearchCV(estimator=model_xgb,param_distributions=random_grid,n_iter=100,cv=5,verbose=2,random_state=42,n_jobs=1)

xgb_random.fit(x_train,y_train)

xgb_random.best_params_

xgb_random.best_score_

 
format(xgb_random.score(x_train,y_train))
format(xgb_random.score(x_test,y_test))


prediction_xg_hpt=xgb_random.predict(x_test)


#Labels= ["ckd" , "not ckd"]
conf_matrix_xgb= confusion_matrix(y_test , prediction_xg_hpt)
plt.figure(figsize=(5,5))
sns.heatmap(conf_matrix_xgb,annot=True,fmt="d",cmap="cividis")
plt.title("Confusion matrix for XGBoost ",color="r",fontsize=15)
plt.ylabel("True class")
plt.xlabel("Predicted class")
plt.show()


acc_XGB_hyp= accuracy_score(y_test,prediction_xg_hpt)
prec_XGB_hyp= precision_score(y_test,prediction_xg_hpt,average='micro')
rec_XGB_hyp= recall_score(y_test,prediction_xg_hpt,average='micro')
# f1_score_XGB_hyp= f1_score(y_test,prediction_xg_hpt,average='micro')

print("acc_XGB_hyp: ", acc_XGB_hyp)
print("prec_XGB_hyp: ", prec_XGB_hyp)
print("rec_XGB_hyp: ", rec_XGB_hyp)
# print("f1_score_XGB_hyp: ", f1_score_XGB_hyp)


#sampling
from imblearn.combine import SMOTEENN
sm= SMOTEENN()
x_resampled,y_resampled=sm.fit_resample(x,y)


# Split the dataset into training and testing sets
xd_train, xd_test, yd_train, yd_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

from sklearn.model_selection import RandomizedSearchCV

#Number of trees in a random forest 
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]
#Various learning rate parameters 
learning_rate=['0.05','0.1','0.2','0.3','0.5','0.6']
#Maximum number of lavels in tree
max_depth=[int(x) for x in np.linspace(5,30,num=6)]
#max_depth.append(none)
#Subsample parameter value
subsample=[0.7,0.6,0.8]
##Minimum child weight parameters 
min_child_weight=[3,4,5,6,7]


random_grid={'n_estimators':n_estimators,
           'learning_rate':learning_rate,
           'max_depth':max_depth,
           'subsample':subsample,
           'min_child_weight':min_child_weight}
print(random_grid)

model_xgb=XGBClassifier()
xgb_random=RandomizedSearchCV(estimator=model_xgb,param_distributions=random_grid,n_iter=100,cv=5,verbose=2,random_state=42,n_jobs=1)

xgb_random.fit(xd_train,yd_train)

xgb_random.best_params_

xgb_random.best_score_

 
format(xgb_random.score(xd_train,yd_train))
format(xgb_random.score(xd_test,yd_test))


prediction_xgd_hpt=xgb_random.predict(xd_test)

#Labels= ["churn" , "not churn"]
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
print(classification_report(yd_test, prediction_xgd_hpt))
import tensorflow as tf
import seaborn as sn 
cm = tf.math.confusion_matrix(labels=yd_test,predictions=prediction_xgd_hpt)
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot= True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
#Labels= ["ckd" , "not ckd"]
conf_matrix_xgb= confusion_matrix(yd_test , prediction_xgd_hpt)
plt.figure(figsize=(5,5))
sns.heatmap(conf_matrix_xgb,annot=True,fmt="d",cmap="cividis")
plt.title("Confusion matrix for XGBoost ",color="r",fontsize=15)
plt.ylabel("True class")
plt.xlabel("Predicted class")
plt.show()


acc_XGB_hyp= accuracy_score(yd_test,prediction_xgd_hpt)
prec_XGB_hyp= precision_score(yd_test,prediction_xgd_hpt,average='micro')
rec_XGB_hyp= recall_score(yd_test,prediction_xgd_hpt,average='micro')
# f1_score_XGB_hyp= f1_score(y_test,prediction_xg_hpt,average='micro')

print("acc_XGB_hyp: ", acc_XGB_hyp)
print("prec_XGB_hyp: ", prec_XGB_hyp)
print("rec_XGB_hyp: ", rec_XGB_hyp)
# print("f1_score_XGB_hyp: ", f1_score_XGB_hyp)
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# true binary labels and predicted probabilities from a classifier


# calculate fpr, tpr, and thresholds for different probability cutoffs
fpd, tpd, thresholds_d = roc_curve(yd_test, prediction_xgd_hpt)

# calculate AUC score
auc_score = roc_auc_score(yd_test, prediction_xgd_hpt)

# plot the ROC curve
plt.plot(fpd, tpd, label=f'AUC={auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
from sklearn.metrics import confusion_matrix , classification_report
# compute the confusion matrix
tn_d, fp_d, fn_d, tp_d = confusion_matrix(yd_test,  prediction_xgd_hpt).ravel()

# compute the observed proportion of agreement
po_d = (tp_d + tn_d) / (tp_d + tn_d + fp_d + fn_d)

# compute the expected proportion of agreement
pe_d = ((tp_d + fp_d) * (tp_d + fn_d) + (tn_d + fp_d) * (tn_d + fn_d)) / (tp_d + tn_d + fp_d + fn_d) ** 2

# compute Cohen's kappa
kappa_d = (po_d - pe_d) / (1 - pe_d)

print(f"Cohen's kappa: {kappa_d}")


#Up-sampling
from  imblearn.combine import SMOTETomek
st=SMOTETomek()
x_r,y_r =st.fit_resample(x,y)

# train test split over Tomek
from sklearn.model_selection import train_test_split
xr_train, xr_test, yr_train, yr_test = train_test_split(x_r,y_r,test_size=0.2,random_state=(5))

from sklearn.model_selection import RandomizedSearchCV

#Number of trees in a random forest 
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]
#Various learning rate parameters 
learning_rate=['0.05','0.1','0.2','0.3','0.5','0.6']
#Maximum number of lavels in tree
max_depth=[int(x) for x in np.linspace(5,30,num=6)]
#max_depth.append(none)
#Subsample parameter value
subsample=[0.7,0.6,0.8]
##Minimum child weight parameters 
min_child_weight=[3,4,5,6,7]


random_grid={'n_estimators':n_estimators,
           'learning_rate':learning_rate,
           'max_depth':max_depth,
           'subsample':subsample,
           'min_child_weight':min_child_weight}
print(random_grid)

model_xgb=XGBClassifier()
xgb_random=RandomizedSearchCV(estimator=model_xgb,param_distributions=random_grid,n_iter=100,cv=5,verbose=2,random_state=42,n_jobs=1)

xgb_random.fit(xr_train,yr_train)

xgb_random.best_params_

xgb_random.best_score_

 
format(xgb_random.score(xr_train,yr_train))
format(xgb_random.score(xr_test,yr_test))


prediction_xgr_hpt=xgb_random.predict(xr_test)


#Labels= ["churn" , "not churn"]
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
print(classification_report(yr_test, prediction_xgr_hpt))
import tensorflow as tf
import seaborn as sn 
cm = tf.math.confusion_matrix(labels=yr_test,predictions=prediction_xgr_hpt)
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot= True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# acc_XGB_hyp= accuracy_score(yd_test,prediction_xgr_hpt)
# prec_XGB_hyp= precision_score(yd_test,prediction_xgr_hpt,average='micro')
# rec_XGB_hyp= recall_score(yd_test,prediction_xgr_hpt,average='micro')
# f1_score_XGB_hyp= f1_score(y_test,prediction_xg_hpt,average='micro')

# print("acc_XGB_hyp: ", acc_XGB_hyp)
# print("prec_XGB_hyp: ", prec_XGB_hyp)
# print("rec_XGB_hyp: ", rec_XGB_hyp)
# # print("f1_score_XGB_hyp: ", f1_score_XGB_hyp)
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# true binary labels and predicted probabilities from a classifier


# calculate fpr, tpr, and thresholds for different probability cutoffs
fpr, tpr, thresholds_r = roc_curve(yr_test, prediction_xgr_hpt)

# calculate AUC score
auc_score = roc_auc_score(yr_test, prediction_xgr_hpt)

# plot the ROC curve
plt.plot(fpr, tpr, label=f'AUC={auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
from sklearn.metrics import confusion_matrix , classification_report
# compute the confusion matrix
tn_r, fp_r, fn_r, tp_r = confusion_matrix(yr_test,  prediction_xgr_hpt).ravel()

# compute the observed proportion of agreement
po_r = (tp_r + tn_r) / (tp_r + tn_r + fp_r + fn_r)

# compute the expected proportion of agreement
pe_r = ((tp_r + fp_r) * (tp_r + fn_r) + (tn_r + fp_r) * (tn_r + fn_r)) / (tp_r + tn_r + fp_r + fn_r) ** 2

# compute Cohen's kappa
kappa_r = (po_r - pe_r) / (1 - pe_r)

print(f"Cohen's kappa: {kappa_r}")


