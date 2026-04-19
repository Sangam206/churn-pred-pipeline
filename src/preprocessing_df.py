import pandas as pd  
import numpy as np  
import os 
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

mk_dir = "log"
os.makedirs(mk_dir, exist_ok=True)

logger = logging.getLogger("preprocess")
logger.setLevel(logging.DEBUG)

# handler
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)

file_dir = os.path.join(mk_dir, "preprocess.log")
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
        x_train.drop(columns="City",inplace=True)
        x_test.drop(columns="City",inplace=True)
        cat=x_train.select_dtypes(include="object").columns
        num=x_train.select_dtypes(include=["int64",'float64']).columns
        
        ct=ColumnTransformer(transformers=[
            ("cat_cols",OneHotEncoder(sparse_output=False),cat),
            ("num_cols",StandardScaler(),num)])
        x_train_array=ct.fit_transform(x_train)
        x_train=pd.DataFrame(x_train_array,columns=ct.get_feature_names_out())

        x_test_array=ct.transform(x_test)
        x_test=pd.DataFrame(x_test_array,columns=ct.get_feature_names_out())
        
        mk_dir="after encoding"
        os.makedirs(mk_dir,exist_ok=True)

        x_train.to_csv(os.path.join(mk_dir,"x_train.csv"))
        x_test.to_csv(os.path.join(mk_dir,"x_test.csv"))
        logger.debug("successfully encoding")

        return x_train,x_test
    except Exception as e:
        logger.error("error occured: {e}")
        raise


    


