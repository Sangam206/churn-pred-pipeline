import pandas as pd  
import numpy as np  
import os 
import logging

from sklearn.preprocessing import OneHotEncoder
from data_injection import data_cleaning,load_data,data_splitting
from preprocessing_df import encoding_cat

df=load_data('E:\customer churn\churn-pred-pipeline\dataset\ecommerce_customer_churn_dataset.csv')
clean = data_cleaning(df)

x_train, y_train, x_test, y_test = data_splitting(clean)

encode = encoding_cat(x_train, x_test)
