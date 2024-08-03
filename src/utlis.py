import os
import sys
import pickle
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)





class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        print(" ")

    def fit(self, X, y=None):
        # No fitting needed for this transformer
        return self

    def transform(self, X, y=None):
        df = X.copy()

        # Apply custom transformations
        df['month'] = df['Dt_Customer'].str.split('/').str[0].astype(int)
        df['date'] = df['Dt_Customer'].str.split('/').str[1].astype(int)
        df['year'] = df['Dt_Customer'].str.split('/').str[2].astype(int)

        education_mapping = {'Basic': 0, 'Graduation': 1, '2n Cycle': 3, 'Master': 4, 'PhD': 5}
        df['Education'] = df['Education'].map(education_mapping)

        df = df.dropna(subset=['Income'])

        # Ensure the column names are preserved
        return df


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        # No fitting needed for this transformer
        return self

    def transform(self, X):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("LogTransformer requires X to be a pandas DataFrame")

        df = X.copy()

        # Apply log transformation to the specified columns
        for column in self.columns:
            if column in df.columns:
                # Apply log transformation with a small constant to avoid log(0)
                df[column] = np.log1p(df[column])
            else:
                raise ValueError(f"Column {column} not found in the DataFrame")

        return df