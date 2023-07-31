# Customer-Behavior-Analysis-Customer-Segmentation-and-Churn-Analysis: Project using Machine Learning and Survival Analysis
## **What is Churn?**
Churn is a very important as well as a very concerning issue of an institution where it affects the institution adversely.Churn analysis has become crucial as it helps provide good quality service.It also helps to detect the issues behind the churn and makes the institute aware of their shortcomings.
## **Why this analysis is important.**
Employees play a crucial role in delivering goods and services, while customers drive the growth and success of a company.Loyalty and retention of customers and employees are critical for business success, as they contribute to the company's reputation, revenue, and stability.Cultivating a community of loyal and active customers who engage with the company's products/services and share content within their social networks is crucial for leveraging word-of-mouth marketing and strengthening customer relationships.
## **Objective of this study.**
To observe segmented customers according to the attributes and observe their churning rate and surviving rate.To build models to predict the churn using various machine learning tools.The primary focus is to build a model with higher accuracy which is more than or equal to 94%.
## **Methodology.**
Data analysis and machine learning techniques are widely used to analyze customer behavior, involving data cleaning, preprocessing, exploratory data analysis (EDA), and classification models for predicting churn, enabling informed decision-making.The computation of customer lifetime value serves as an estimation of the monetary value that the institution stands to lose, given the available data.Customer segmentation is performed using survival analysis to understand the churning pattern among the customers.The data is resampled to overcome the issue of class imbalance of the dependent variable.Three distinct classification methodologies have been employed to classify the churn variable. These include feed-forward neural network, bagging, and boosting.
## **Data source and description.**
The data used in the study is the ‘WA_Fn-UseC_-Telco-Customer-Churn’ dataset that IBM publishes on GitHub; the link to the data is: https://raw.githubusercontent.com/treselle-systems/' 'customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv.The data considered is pre-processed secondary data, so we don’t disclose the customers’ information.In the data, there are 21 variables and 7043 customer data.After removing missing values, the dataset consists of 1869 instances of customer churn and 5163 active customers. The overall churn rate in the dataset is 26.5785%.
## **Exploratory data analysis**
### Correlation matrix
![corr matrix](https://github.com/SuryakantaGhanta/Customer-Behavior-Analysis-Customer-Segmentation-and-Churn-Analysis/assets/119864485/1061c777-c80e-4f33-a53a-29df97cb2abe)
The correlation matrix presented indicates that the features included within the dataset exhibit a considerable degree of independence from one another. It is noteworthy to mention that the majority of the correlation observed is derived from the categorical attributes, as the dummy variables interact with one another and consequently display a correlation. This finding under scores the importance of carefully examining the categorical attributes, as they play a pivotal role in the overall correlation structure of the dataset. In order to assess the association between the churn, which is the dependent variable in this particular study, and the other binary variables, the Cramer’s V correlation was utilized. The inclusion of these correlations provides valuable insight into the relationships between the variables being considered.
### The binary variables effecting churn
![n1](https://github.com/SuryakantaGhanta/Customer-Behavior-Analysis-Customer-Segmentation-and-Churn-Analysis/assets/119864485/6c74f667-5a2f-42dc-8f0e-d3733a0b36bb)

The analysis of binary variables in the dataset indicates an expected imbalance between churners and non-churners, reflecting lower churn rates in companies' operations.
Factors such as dependents, paperless billing, online security, partner, and tech support significantly impact churn rates, with customers without dependents, paperless billing, and these services exhibiting higher churn rates compared to their counterparts.
###  The multi-class variables effecting the churn
![n2](https://github.com/SuryakantaGhanta/Customer-Behavior-Analysis-Customer-Segmentation-and-Churn-Analysis/assets/119864485/766b1c74-3a23-4d6a-ac1a-623c39188377)
Graph displays multi-class variables related to internet service, contract, payment method, and their impact on churning rates.
Customers with fiber optics internet service are more prone to churning.
Monthly contract customers exhibit the highest churning rate.
Customers paying through electronic check are more likely to churn compared to other payment methods.
Identifying factors impacting churning rate can help the company take necessary steps to rectify or monitor the situation.
This can lead to improved service quality and ultimately reduce the churning rate.
## Survival Analysis of the customers according to the attributes
![survival1](https://github.com/SuryakantaGhanta/Customer-Behavior-Analysis-Customer-Segmentation-and-Churn-Analysis/assets/119864485/0c9ef5e5-be7f-4052-a004-eb5b159b1d70)

The survival curves denote the respective attributes while considering tenure as the time variable and churn as the event variable. These curves offer a visual representation of the influence of distinct attributes on customer churn, based on their respective categories. The survival curves, therefore, serve as a crucial tool for comprehending the factors that impact customer churn and devising effective strategies for its mitigation. The graphical representation of the survival curve depicts the likelihood of customer survival, which is observed to be approximately 0.6. The plot clearly indicates a sharp decline in the duration interval of 0 to 10 months, signifying that customers with a tenure of 0 to 10 months are more susceptible to churn. However, the curve becomes gentler as the tenure increases, indicating a positive correlation between customer survival and service tenure. Therefore, it can be deduced that the longer a customer stays with the service, the higher the probability of their survival.

![survival2](https://github.com/SuryakantaGhanta/Customer-Behavior-Analysis-Customer-Segmentation-and-Churn-Analysis/assets/119864485/22d73232-5910-4cbe-b20c-0bdcd7a9d75b)

The survival curve of male and female customers in the plot is indistinguishable. This observation suggests that the gender variable does not exert a significant impact on customer churn. It is reasonable to conclude that the likelihood of a customer churning is not influenced by their gender, as both male and female customers exhibit similar patterns of churn.

![survival3](https://github.com/SuryakantaGhanta/Customer-Behavior-Analysis-Customer-Segmentation-and-Churn-Analysis/assets/119864485/6edf84e0-ec8b-4467-a1d2-1dceff6f5a07)

In the figure, it is observable that senior citizens exhibit a higher probability of churning as opposed to those who are not categorized as senior citizens. Specifically, the survival probability for seniors is noted to be quite low, with a value as small as 0.421. This value is in stark contrast to non-senior citizens who possess a survival probability of 0.634 during the period of 70 months and beyond.

![survival4](https://github.com/SuryakantaGhanta/Customer-Behavior-Analysis-Customer-Segmentation-and-Churn-Analysis/assets/119864485/3ab6bf5d-dbcc-4bed-a176-6c1ed1d1a3f0)

The payment method used by customers can help predict their likelihood of churning, and it appears that those who pay through electronic cheque are the most likely to churn, with a survival probability of only 0.294. Customers who pay through mailed check have a higher survival probability of 0.726. Meanwhile, customers who pay through bank transfer (automatic) and credit card (automatic) have similar survival probabilities of 0.745 and 0.758 respectively. Given that the survival probability for customers who pay through electronic cheque is significantly lower than all other payment methods, service providers should focus on promoting more automatic payment methods to reduce churn.

![survival5](https://github.com/SuryakantaGhanta/Customer-Behavior-Analysis-Customer-Segmentation-and-Churn-Analysis/assets/119864485/f57f576a-a12a-46a4-8d34-018dfec8c050)

The illustration portrays the evident outcomes of the churning pattern with respect to the type of contract opted by the customers. The survival probability of the customers who have a month-to-month contract is only 0.129, which is the lowest amongst all the contract types. However, the probability increases to 0.568 and 0.936 for customers who have opted for the one year and two years contract, respectively. The low survival rate of the monthly customers is a matter of concern for the service provider, thereby making it imperative for them to offer more lucrative deals for the one year and two-year contracts to reduce churning.

![survival6](https://github.com/SuryakantaGhanta/Customer-Behavior-Analysis-Customer-Segmentation-and-Churn-Analysis/assets/119864485/5af618d1-0f7c-41fd-9900-48dcca9631b6)

The representation of customer survival in relation to internet service in the plot reveals a fascinating outcome. The customers who possess the greatest likelihood of survival are those who do not have access to internet service, with a probability of 0.902. On the other hand, the customers who are most prone to churn are those who have fiber optic, followed by those who possess DSL, with probabilities of 0.417 and 0.722, respectively.This highlights the importance of internet service in customer retention and underscores the need for providers to revisit their strategies for customer engagement. The survival curves of customer segmentation based on different attributes provided us with a comprehensive understanding of customer behavior, and also facilitated the prescriptive analysis of customer churn. Additionally, this data can be utilized to optimize marketing strategies and improve overall customer retention
## Data pre-processing.
Data preprocessing involves noise removal, handling missing values, variable transformation, and standardization to ensure data is in the correct format for analysis.
Imbalanced datasets can be addressed using data balancing techniques like SMOTE-ENN and SMOTE-Tomek, which enhance class balance in churn prediction scenarios.
- SMOTE-ENN combines synthetic minority oversampling and edited nearest neighbors to improve classification performance in imbalanced datasets by reducing noise and balancing classes.
- SMOTE-Tomek utilizes SMOTE for oversampling the minority class and Tomek Links for identifying and removing ambiguous or noisy instances, resulting in a balanced dataset for supervised machine learning tasks.
Data preprocessing and resampling techniques are crucial in accurately analyzing customer behavior, yielding reliable outcomes, and facilitating informed decision-making.

## Models' accuracy with imbalanced data
![image](https://github.com/SuryakantaGhanta/Customer-Behavior-Analysis-Customer-Segmentation-and-Churn-Analysis/assets/119864485/83e927ec-8aec-4974-9dea-742beb56f3a0)

The data is classified using various types of machine learning models. The feedforward neural network,  Extreme Learning Machine and the bagging technique decision tree performs comparatively well but not as good as XG Boost or Random forest. Boosting and bagging techniques both has the highest accuracy of 80% with the imbalanced data.

## Models and accuracy with up-sampled data
![image](https://github.com/SuryakantaGhanta/Customer-Behavior-Analysis-Customer-Segmentation-and-Churn-Analysis/assets/119864485/ef2aa74d-be2d-4633-a156-c57a0546ef61)

The data classification models  performs better with the balanced class of churners and non-churners, where there is an increase in overall performance of all the models used.Extreme Learning Machine and Decision tree shows a similar type of result as in the case of imbalanced data.Boosting technique has the highest accuracy of 87% with the up-sampled data.
## Models and accuracy with down-sampled data

![image](https://github.com/SuryakantaGhanta/Customer-Behavior-Analysis-Customer-Segmentation-and-Churn-Analysis/assets/119864485/3c5ce3ff-a43a-4af2-b987-67cfc0cd438b)

The optimal data classification of churners and non-churners occurs through down-sampling of the data, leading to near 90% accuracy across almost all models.Both the Extreme Learning Machine and Decision Tree models have exhibited excellent performance when trained on this balanced dataset, achieving an impressive accuracy score of 90% and 88%, respectively.The Random forest model has displayed superior performance when compared to all other models, as it has achieved the highest level of accuracy at a remarkable 98%. Furthermore, it has successfully fulfilled the primary objective of this study.
## Feature importance
![image](https://github.com/SuryakantaGhanta/Customer-Behavior-Analysis-Customer-Segmentation-and-Churn-Analysis/assets/119864485/64de641d-060b-4b0b-83e6-b99ac5e41da9)

## Results
Yearly stake: $4,770,672.45, with an annual churn rate of 26.5785%. Primary drivers of churn: Total charges, Monthly charges, and tenure. Services impacting churn: Customers not availing most services are more likely to churn, while those with streaming TV and paperless billing have a higher tendency to churn. Factors influencing churn: Fiber optic internet service, monthly tenure, and electronic check payments increase the likelihood of churn.
## Conclusion 
From the available statistics, it can be inferred that the organization holds a yearly stake of $4,770,672.45 and an annual churn rate of 26.5785%. The primary driving factors responsible for the churn rate are total charges, monthly charges, and tenure. The data under investigation involves binary variables that represent the services offered by a company, which have a noteworthy impact on the company’s churn. Based on the findings of the exploratory data analysis, it is evident that customers who do not avail most of the services offered by the company are more likely to churn. However, those who have subscribed to the services of streaming TV and paperless billing exhibit a greater tendency towards churn. Therefore, the company ought to concentrate its efforts on these two services in order to minimize churn. Additionally, the data features multi-class variables that correspond to various types of services offered by the company. The analysis reveals that customers who utilize fiber optic internet service, have a monthly tenure, and make electronic check payments are more likely to churn. Hence, the company must remain updated on these types of services and strive to address the issues related to them in order to mitigate churn. Overall, the results of this study suggest that a company can reduce churn by focusing on specific services and monitoring various types of services offered. The senior citizens, who demonstrate a survival probability of 0.421, individuals who pay through electronic check with a survival probability of 0.294, and customers with a monthly tenure and a survival probability of 0.129 are the primary contributors to the churn rate. The differentiation in the number of male and female churners is trivial, and gender has no substantial effect on the rate of churn. The factors that exert the greatest influence on churn are the overall charges, monthly charges, and tenure. In other words, an increase in total charges, monthly charges, or tenure would result in an increase in the churn rate. Therefore, to minimize the churn rate, it is necessary to maintain reasonable charges and ensure customer satisfaction throughout their tenure. The data has been trained using various models, including ELM, Decision Tree, Random Forest, and XGBoost. Balancing the data by down sampling the dataset and developing balanced classes for churners and non-churners has resulted in an accuracy of 98% for the Random Forest classifier, making it the most efficient model. The Random Forest classifier demonstrates the capacity to deliver the most reliable outcomes when it comes to predicting churn rates.














