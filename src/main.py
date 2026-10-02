import pandas as pd  
import numpy as np  
import os 
import logging
from data_injection import data_cleaning,load_data,data_splitting
from preprocessing_df import encoding_cat
from model_building import hyperparameter_tuning, model_training,save_model
from model_test import test_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "dataset", "ecommerce_customer_churn_dataset.csv")
df = load_data(data_path)
clean = data_cleaning(df)

x_train,x_test,y_train,y_test = data_splitting(clean)

x_train_en,x_test_en= encoding_cat(x_train, x_test)

hyper_param=hyperparameter_tuning(x_train_en,y_train)

model_train=model_training(x_train_en,y_train,hyper_param)

path = os.path.join(BASE_DIR, "model", "model.pkl")
pkl_model=save_model(model_train,path)

test=test_model(model_train,x_test_en,y_test)





