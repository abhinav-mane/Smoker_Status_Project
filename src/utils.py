import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

# using this function to save objects that we need all the time
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
# calculating the model sensitivity
def model_sensitivity(TP,FN):
    return TP / float(TP + FN) # formula to calculate sensitivity
    
# calculating the model specificity
def model_specificity(TN,FP):
    return TN / float(TN + FP) # formula to calculate specificity

# calculating the model False Positive Rate
def model_FPR(TN,FP):
    return FP/ float(TN + FP)

# calculating the model negative prediction rate
def model_negative_pred_val(TN,FN):
    return TN / float(TN + FN)


def evaluate_models(X_train,y_train,X_test,y_test,models):

    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)   

            y_test_pred = model.predict(X_test)

            conf_mat = confusion_matrix(y_true=y_test,y_pred=y_test_pred)

            TP = conf_mat[1,1] # true positive 
            TN = conf_mat[0,0] # true negatives
            FP = conf_mat[0,1] # false positives
            FN = conf_mat[1,0] # false negatives

            model_sensi = model_sensitivity(TP,FN)
            model_speci = model_specificity(TN,FP)
            model_false_pos_rate = model_FPR(TN,FP)
            model_neg_pred_val = model_negative_pred_val(TN,FN)

            report[list(models.keys())[i]] = {"Model_Sensitivity/Recall":model_sensi,"Model_Specificity":model_speci,"False_Positive_Rate":model_false_pos_rate,"Model_Negative_Prediction_Rate":model_neg_pred_val}

            return report
    
    except Exception as e:
        raise CustomException(e, sys)



def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)