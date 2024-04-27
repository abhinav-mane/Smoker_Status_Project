import os
import sys
from dataclasses import dataclass

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    #     # calculating the model sensitivity
    # def model_sensitivity(TP,FN):
    #     return TP / float(TP + FN) # formula to calculate sensitivity
        
    # # calculating the model specificity
    # def model_specificity(TN,FP):
    #     return TN / float(TN + FP) # formula to calculate specificity

    # # calculating the model False Positive Rate
    # def model_FPR(TN,FP):
    #     return FP/ float(TN + FP)

    # # calculating the model negative prediction rate
    # def model_negative_pred_val(TN,FN):
    #     return TN / float(TN + FN)

    def initiate_model_trainer(self,train_array,test_array):

        try:
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Logistic Regression": LogisticRegression()
            }

            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            # predicted = best_model.predict(X_test)

            return model_report

            # conf_mat = confusion_matrix(y_true=y_test,y_pred=predicted)

            # TP = conf_mat[1,1] # true positive 
            # TN = conf_mat[0,0] # true negatives
            # FP = conf_mat[0,1] # false positives
            # FN = conf_mat[1,0] # false negatives

            # model_sensi = self.model_sensitivity(TP=TP,FN=FN)
            # model_speci = self.model_specificity(TN,FP)
            # model_false_pos_rate = self.model_FPR(TN,FP)
            # model_neg_pred_val = self.model_negative_pred_val(TN,FN)

            # print(conf_mat)

            # return {"Model_Sensitivity":model_sensi,"Model_Specificity":model_speci,"False_Positive_Rate":model_false_pos_rate,"Model_Negative_Prediction_Rate":model_neg_pred_val}

        except Exception as e:
            raise CustomException(e,sys)