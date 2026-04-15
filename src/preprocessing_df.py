import pandas as pd  
import numpy as np  
import os 
import logging

from sklearn.preprocessing import OneHotEncoder

mk_dir = "log"
os.makedirs(mk_dir, exist_ok=True)

logger = logging.getLogger("log")
logger.setLevel(logging.DEBUG)

# handler
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)

file_dir = os.path.join(mk_dir, "logs.log")
file_han = logging.FileHandler(file_dir)
file_han.setLevel(logging.DEBUG)

# formatter
format = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# attach formatter to handlers
console.setFormatter(format)
file_han.setFormatter(format)

# add handlers to logger
logger.addHandler(console)
logger.addHandler(file_han)


def encoding_cat (x_train,x_test):
    
    try:
        x_train= x_train.drop(columns=['City'])
        x_test=x_test.drop(columns=['City'])
        logger.debug("encoding started")

        ohe=OneHotEncoder(sparse_output=False)
        train_ohe=ohe.fit_transform(x_train[['Country']])
        x_train=pd.concat([x_train,pd.DataFrame(train_ohe,columns=ohe.get_feature_names_out())],axis=1)
        test_ohe=ohe.transform(x_test[['Country']])
        x_test=pd.concat([x_test,pd.DataFrame(test_ohe,columns=ohe.get_feature_names_out())],axis=1)
        x_train_gen=ohe.fit_transform(x_train[['Gender']])
        x_test_gen=ohe.transform(x_test[['Gender']])
        x_train=pd.concat([x_train,pd.DataFrame(x_train_gen,columns=ohe.get_feature_names_out())],axis=1)
        x_test=pd.concat([x_test,pd.DataFrame(x_test_gen,columns=ohe.get_feature_names_out())],axis=1)



        x_train=x_train.drop(columns=["Country",'Gender'])
        x_test=x_test.drop(columns=["Country",'Gender'])

        mk_dir='after encoding'
        os.makedirs(mk_dir,exist_ok=True)
        x_train.to_csv(os.path.join(mk_dir,"x_train.csv"),index=False)
        x_test.to_csv(os.path.join(mk_dir,"x_test.csv"),index=False)
        logger.debug("encoding done")
        return x_train,x_test
    except Exception as e:
        logger.error(e)
        raise





