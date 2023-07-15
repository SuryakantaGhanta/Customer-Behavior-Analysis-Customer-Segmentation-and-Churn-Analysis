# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:58:23 2023

@author: SURJAKANTA
"""

import pandas as pd
from lifetimes import BetaGeoFitter
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn (3).csv")

# Calculate the CLV
average_revenue_per_customer = data["MonthlyCharges"].mean()
average_monthly_churn_rate = data["Churn"].value_counts(normalize=True)["Yes"]
average_customer_lifetime = 1 / average_monthly_churn_rate
clv = average_revenue_per_customer * average_customer_lifetime

# Print the CLV
print("The average CLV is $", round(clv, 2))





# # Load the Telco-Customer-Churn dataset
# df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn (3).csv")

# # Convert the "TotalCharges" column to a numeric data type
# df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# # Drop rows with missing values
# df.dropna(inplace=True)

# # Create a new DataFrame with the necessary columns for the BG/NBD model
# bgf_data = df[["customerID", "tenure", "TotalCharges"]]

# # Fit the BG/NBD model to the data
# bgf = BetaGeoFitter(penalizer_coef=0.0)
# bgf.fit(bgf_data["tenure"], bgf_data["TotalCharges"])

# # Calculate the expected number of transactions and the expected revenue for each customer over the next year
# t = 12
# bgf_data["expected_number_of_transactions"] = bgf.conditional_expected_number_of_purchases_up_to_time(t, bgf_data["tenure"], bgf_data["TotalCharges"])
# bgf_data["expected_revenue"] = bgf_data["expected_number_of_transactions"] * bgf.conditional_expected_average_profit(bgf_data["tenure"], bgf_data["TotalCharges"])

# # Calculate the CLV for each customer
# bgf_data["CLV"] = bgf.customer_lifetime_value(bgf, bgf_data["tenure"], bgf_data["TotalCharges"], bgf_data["expected_number_of_transactions"], bgf_data["expected_revenue"], time=t)

# # Print the top 10 customers with the highest CLV
# print(bgf_data.sort_values(by="CLV", ascending=False).head(10))





# import pandas as pd
# from lifetimes import ParetoNBDFitter

# # Load the Telco-Customer-Churn dataset
# df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# # Convert the "TotalCharges" column to a numeric data type
# df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# # Drop rows with missing values
# df.dropna(inplace=True)

# # Create a new DataFrame with the necessary columns for the Pareto/NBD model
# pnbd_data = df[["customerID", "tenure", "TotalCharges"]]

# # Fit the Pareto/NBD model to the data
# pnbd = ParetoNBDFitter(penalizer_coef=0.0)
# pnbd.fit(pnbd_data["tenure"], pnbd_data["TotalCharges"], pnbd_data["customerID"])

# # Calculate the expected number of transactions and the expected revenue for each customer over the next year
# t = 12
# pnbd_data["expected_number_of_transactions"] = pnbd.conditional_expected_number_of_purchases_up_to_time(t, pnbd_data["tenure"], pnbd_data["TotalCharges"])
# pnbd_data["expected_revenue"] = pnbd_data["expected_number_of_transactions"] * pnbd.conditional_expected_average_profit(pnbd_data["tenure"], pnbd_data["TotalCharges"])

# # Calculate the CLV for each customer
# pnbd_data["CLV"] = pnbd.customer_lifetime_value(pnbd, pnbd_data["tenure"], pnbd_data["TotalCharges"], pnbd_data["expected_number_of_transactions"], pnbd_data["expected_revenue"], time=t)

# # Print the top 10 customers with the highest CLV
# print(pnbd_data.sort_values(by="CLV", ascending=False).head(10))




# import pandas as pd
# from lifetimes import BetaGeoFitter
# from lifetimes.utils import summary_data_from_transaction_data

# # Step 1: Load and preprocess the data
# df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
# # Preprocess the data as required (e.g., drop irrelevant columns, convert data types, handle missing values)

# # Step 2: Calculate the frequency and recency values
# summary = summary_data_from_transaction_data(df, 'customerID', 'tenure', 'TotalCharges')
# summary = summary[summary['frequency'] > 0]

# # Step 3: Fit the BG/NBD model
# bgf = BetaGeoFitter()
# bgf.fit(summary['frequency'], summary['recency'], summary['T'])

# # Step 4: Predict future purchases and customer value
# summary['predicted_purchases'] = bgf.predict(summary['frequency'], summary['recency'], summary['T'])
# summary['customer_value'] = bgf.customer_lifetime_value(
#     bgf, summary['frequency'], summary['recency'], summary['T'],
#     summary['monetary_value'], time=12, discount_rate=0.01
# )

# # Step 5: Calculate the customer lifetime value
# avg_profit_per_purchase = df['MonthlyCharges'].mean() - df['TotalCharges'].mean()
# summary['clv'] = summary['predicted_purchases'] * summary['customer_value'] * avg_profit_per_purchase
