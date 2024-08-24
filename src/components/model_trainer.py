import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from src.exception import CustomException
from src.logger import logging
from src.utlis import save_object

@dataclass
class ModelTrainerConfig:
     trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
     def __init__(self):
          self.model_trainer_config=ModelTrainerConfig()

     def initiate_model_trainer(self,train_array,test_array):
          try:
               logging.info("Split training and test input data")
               X_train,y_train,X_test,y_test=(
                    train_array[:,:-1],
                    train_array[:,-1],
                    test_array[:,:-1],
                    test_array[:,-1]
               )
               # models={
               #      "Random Forest": RandomForestRegressor()
               # }
               rf= RandomForestClassifier()
               # Fitting the model
               rf.fit(X_train, y_train)
               # Predicting the model
               pred_rf = rf.predict(X_test)

               logging.info(f"Best found model on both training and testing dataset")

               save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=rf
               )

               print("The accuracy of Regression model is:", accuracy_score(y_test, pred_rf))
               print(classification_report(y_test, pred_rf))



               return accuracy_score(y_test, pred_rf)





          except Exception as e:
               raise CustomException(e, sys)