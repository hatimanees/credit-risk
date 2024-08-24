import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from src.utlis import CustomTransformer
from src.exception import CustomException
from src.logger import logging
import os

from src.utlis import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
                # Create the final pipeline
            pipeline = Pipeline([
                ("custom_transformer", CustomTransformer()),
            ])

            return pipeline

        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,raw_path):

        try:
            raw_df=pd.read_csv(raw_path)

            logging.info("Read train and test data completed")

            logging.info("obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr, target_feature_train_df_arr, input_feature_test_arr, target_feature_test_df_arr = preprocessing_obj.fit_transform(
                raw_df, raw_df['loan_status'])

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df_arr)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df_arr)
            ]

            logging.info(f"saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr
            )
        except Exception as e:
            raise CustomException(e,sys)