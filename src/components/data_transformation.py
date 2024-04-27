import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function will help use to transform the data and give us a preprocessor file.
        
        '''
        try:
            # Defining the numerical columns
            numerical_columns = ["AST", "eyesight(right)","triglyceride","Gtp"]

            # Setting up the pipeline to perform transformations
            num_pipeline= Pipeline(
                steps=[
                ("scaler",MinMaxScaler())
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")

            # Activating the pipeline
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ]
            )

            # Returning the preprocessor
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            # Reading train and test datasets
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            # Loading the preprocessor object
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="smoking"
            numerical_columns = ["AST", "eyesight(right)","triglyceride","Gtp"]

            # Setting the target var and features in train and test set
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Using the preprocessing function to fit and transform training and testing sets
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # np.c_: This is a special object provided by NumPy that allows for the concatenation of arrays along the second axis, which is essentially column-wise concatenation.
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # Using the save_object from the utils.py file to store the object
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)