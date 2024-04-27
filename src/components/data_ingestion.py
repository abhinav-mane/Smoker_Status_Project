import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path: str= os.path.join('artifacts',"train.csv")
    test_data_path: str= os.path.join('artifacts',"split_test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered the Data ingestion method or component")

        try:
            # train_data = pd.read_csv(os.path.join(self.ingestion_config.train_data_path))

            # test_data = train_test_split(train_data)

            # test_data = pd.read_csv(r"D:\Kaggle_Work\Smoker_Status_Project\Smoker_Status_Project\Dataset\test.csv")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))