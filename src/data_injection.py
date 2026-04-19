import pandas as pd 
import numpy as np 
import os 
import logging
from sklearn.model_selection import train_test_split


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

def load_data(path):
    try:
        df = pd.read_csv(path)
        logger.debug("data loded successfully")
        return df
    except Exception as e:
        logger.error(f"error occured: {e}")
        raise


def data_cleaning (df):
    # remove duplicate
    # data filling
    # handeling outliars
    try:
        df=df.drop_duplicates()
        logger.debug("duplicates value is removed from dataset")
        
        # handeling outliars
    
        df.loc[(df['Age']>100) | (df['Age']<10),'Age']=np.nan
        
        df.loc[(df['Cart_Abandonment_Rate']>100) | (df['Cart_Abandonment_Rate']<0),'Cart_Abandonment_Rate']=np.nan
        
        
        df.loc[(df['Total_Purchases']<0),'Total_Purchases']=np.nan
        
        
        df.loc[df['Discount_Usage_Rate'] > 100, 'Discount_Usage_Rate'] = np.nan
        logger.debug("outliars is handeled")

        # data filling 
        num_cols=df.select_dtypes(include=['float64','int64'])  
        for col in num_cols:
            df[col].fillna(df[col].median(), inplace=True)
        logger.debug("numerical columns is filled by meadian filling")
        

        cat_cols=df.select_dtypes(include=['object'])
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0])
        logger.debug("catogorical columns is filled by mode filling")
        
        return df
    
    except Exception as e:
        logger.error("error occured:{e}")


def data_splitting(df):
    logger.debug("data splitting start")
    mk_dir='splitting data'
    os.makedirs(mk_dir,exist_ok=True)
    x=df.drop(columns='Churned')
    y=df['Churned']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
    x_train.to_csv(os.path.join(mk_dir,"x_train.csv"),index=False)
    y_train.to_csv(os.path.join(mk_dir,"y_train.csv"),index=False)
    x_test.to_csv(os.path.join(mk_dir,"x_test.csv"),index=False)
    y_test.to_csv(os.path.join(mk_dir,"y_test.csv"),index=False)
    logger.debug("data splitting done")
    return x_train,x_test,y_train,y_test 

    





