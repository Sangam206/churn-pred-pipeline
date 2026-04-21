from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import os 
import logging
import optuna
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.metrics import classification_report

# handaler 
# formatter
# attach formater
# add handeler

mk_dir = "log"
os.makedirs(mk_dir, exist_ok=True)

logger = logging.getLogger("test model")
logger.setLevel(logging.DEBUG)

# handler
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)

file_dir = os.path.join(mk_dir, "test.log")
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


def test_model(model,x_test,y_test):
    try:
        mk_dir="test_metrics"
        os.makedirs(mk_dir,exist_ok=True)
        pred=model.predict(x_test)
        test_report=classification_report(y_test,pred)

        
        file_path= os.path.join(mk_dir,'test.txt')

        with open(file_path,'a') as file:
            file.write(test_report) 

    except Exception as e:
        logger.error(f"error occured{e}")
        raise
    

# import pickle
# import pandas as pd
# from sklearn.metrics import classification_report

# # load model
# with open("model/model.pkl", "rb") as f:
#     model = pickle.load(f)

# # load data
# x_test = pd.read_csv(r"E:\customer churn\churn-pred-pipeline\after encoding\x_test.csv")
# y_test = pd.read_csv(r"E:\customer churn\churn-pred-pipeline\splitting data\y_test.csv")

# # predict
# y_pred = model.predict(x_test)

# # evaluate
# print(classification_report(y_test, y_pred))