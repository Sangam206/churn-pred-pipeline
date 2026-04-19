import pandas as pd  
import numpy as np  
import os 
import logging
from data_injection import data_cleaning,load_data,data_splitting
from preprocessing_df import encoding_cat
from model_building import hyperparameter_tuning, model_training,save_model

df=load_data('E:\customer churn\churn-pred-pipeline\dataset\ecommerce_customer_churn_dataset.csv')
clean = data_cleaning(df)

x_train,x_test,y_train,y_test = data_splitting(clean)

x_train_en,x_test_en= encoding_cat(x_train, x_test)

hyper_param=hyperparameter_tuning(x_train_en,y_train)

model_train=model_training(x_train_en,y_train,hyper_param)




