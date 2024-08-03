import sys
import pandas as pd
from src.exception import CustomException
from src.utlis import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:

    def __init__(self,
                 Year_Birth,
                 Dt_Customer,
                 Education,
                 Income,
                 Kidhome,
                 Teenhome,
                 Recency,
                 MntWines,
                 MntFruits,
                 MntMeatProducts,
                 MntFishProducts,
                 MntSweetProducts,
                 MntGoldProds,
                 NumDealsPurchases,
                 NumWebPurchases,
                 NumCatalogPurchases,
                 NumStorePurchases,
                 NumWebVisitsMonth,
                 Complain,
                 Marital_Status):
        self.Year_Birth = Year_Birth
        self.Dt_Customer = Dt_Customer
        self.Education = Education
        self.Income = Income
        self.Kidhome = Kidhome
        self.Teenhome = Teenhome
        self.Recency = Recency
        self.MntWines = MntWines
        self.MntFruits = MntFruits
        self.MntMeatProducts = MntMeatProducts
        self.MntFishProducts = MntFishProducts
        self.MntSweetProducts = MntSweetProducts
        self.MntGoldProds = MntGoldProds
        self.NumDealsPurchases = NumDealsPurchases
        self.NumWebPurchases = NumWebPurchases
        self.NumCatalogPurchases = NumCatalogPurchases
        self.NumStorePurchases = NumStorePurchases
        self.NumWebVisitsMonth = NumWebVisitsMonth
        self.Complain = Complain
        self.Marital_Status = Marital_Status

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Year_Birth": [self.Year_Birth],
                "Dt_Customer": [self.Dt_Customer],
                "Education": [self.Education],
                "Income": [self.Income],
                "Kidhome": [self.Kidhome],
                "Teenhome": [self.Teenhome],
                "Recency": [self.Recency],
                "MntWines": [self.MntWines],
                "MntFruits": [self.MntFruits],
                "MntMeatProducts": [self.MntMeatProducts],
                "MntFishProducts": [self.MntFishProducts],
                "MntSweetProducts": [self.MntSweetProducts],
                "MntGoldProds": [self.MntGoldProds],
                "NumDealsPurchases": [self.NumDealsPurchases],
                "NumWebPurchases": [self.NumWebPurchases],
                "NumCatalogPurchases": [self.NumCatalogPurchases],
                "NumStorePurchases": [self.NumStorePurchases],
                "NumWebVisitsMonth": [self.NumWebVisitsMonth],
                "Complain": [self.Complain],
                "Marital_Status": [self.Marital_Status]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)