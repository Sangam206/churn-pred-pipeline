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

logger = logging.getLogger("train model")
logger.setLevel(logging.DEBUG)

# handler
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)

file_dir = os.path.join(mk_dir, "train.log")
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





# def hyperparameter_tuning(x_train,y_train):
#     #randomforest train
#     try:
#         def objective(trial):
#             model = RandomForestClassifier(
#                 n_estimators=trial.suggest_int("n_estimators", 100, 1000),
#                 max_depth=trial.suggest_int("max_depth", 2, 20),
#                 min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
#                 min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
#                 max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
#                 class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
#                 random_state=42,
#                 n_jobs=-1
#             )

#             scores = cross_val_score(
#                 model,
#                 x_train,y_train,
#                 cv=5,
#                 scoring="f1_macro"
#             )

#             return scores.mean()
        
#         study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())
#         study.optimize(objective,n_trials=50)
#         best_par =study.best_params
#         logger.debug("hyperparamater tuning is done")
#         return best_par
    
#     except Exception as e:
#         logger.error(f"error occured: {e}")
#         raise


# def model_training(x_train,y_train,best_par):

#     rf = RandomForestClassifier(
#     n_estimators=best_par["n_estimators"],
#     max_depth=best_par["max_depth"],
#     min_samples_split=best_par["min_samples_split"],
#     min_samples_leaf=best_par["min_samples_leaf"],
#     max_features=best_par["max_features"],
#     class_weight=best_par["class_weight"],
#     random_state=42,
#     n_jobs=-1
# )
#     rf.fit(x_train,y_train)
#     mk_dir="train metrics"
#     file_path=os.makedirs(mk_dir,exist_ok=True)
#     report=classification_report(x_train,y_train)
#     file="train_model_metrics"
#     with open (file_path,'ab') as file:

#         file.write(report)
#         file.write("\n\n")

#     return rf 
    
import optuna
import os
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


def hyperparameter_tuning(x_train, y_train):

    try:
        def objective(trial):

            model = XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 12),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                gamma=trial.suggest_float("gamma", 0, 5),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                reg_alpha=trial.suggest_float("reg_alpha", 0, 5),
                reg_lambda=trial.suggest_float("reg_lambda", 0, 5),
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss"
            )

            scores = cross_val_score(
                model,
                x_train,
                y_train,
                cv=5,
                scoring="f1_macro"
            )

            return scores.mean()

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=50)

        best_par = study.best_params
        logger.debug("Hyperparameter tuning for XGBoost is done")

        return best_par

    except Exception as e:
        logger.error(f"error occurred: {e}")
        raise


def model_training(x_train, y_train, best_par):

    try:
        xgb = XGBClassifier(
            n_estimators=best_par["n_estimators"],
            max_depth=best_par["max_depth"],
            learning_rate=best_par["learning_rate"],
            subsample=best_par["subsample"],
            colsample_bytree=best_par["colsample_bytree"],
            gamma=best_par["gamma"],
            min_child_weight=best_par["min_child_weight"],
            reg_alpha=best_par["reg_alpha"],
            reg_lambda=best_par["reg_lambda"],
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )

        xgb.fit(x_train, y_train)

        # metrics folder
        mk_dir = "train_metrics"
        os.makedirs(mk_dir, exist_ok=True)

        report = classification_report(y_train, xgb.predict(x_train))

        file_path = os.path.join(mk_dir, "train_model_metrics.txt")

        with open(file_path, "a") as file:
            file.write(report)
            file.write("\n\n")

        return xgb

    except Exception as e:
        logger.error(f"training error: {e}")
        raise

def save_model(model,file_path):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        
        with open (file_path,'wb') as file:
            pickle.dump(model,file)

    except Exception as e:
        logger.error(f"error occured{e}")

  