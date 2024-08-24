import os
import sys
import pickle
from src.exception import CustomException
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

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
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.columns_to_scale = None  # To store the columns for scaling
        self.ohe_columns = None  # To store the one-hot encoded columns

    def fit(self, X, y=None):
        df = X.copy()

        # Impute loan_int_rate with the median
        df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

        # Drop rows with missing person_emp_length
        df.dropna(subset=['person_emp_length'], inplace=True)

        # One-hot encoding for 'person_home_ownership' and 'loan_intent'
        df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent'], drop_first=True)
        self.ohe_columns = df.columns  # Save the columns after encoding

        # Fit label encoder on 'cb_person_default_on_file'
        df['cb_person_default_on_file'] = self.label_encoder.fit_transform(df['cb_person_default_on_file'])

        # Separate features and target
        self.columns_to_scale = df.drop(['loan_status'], axis=1).columns

        # Fit the scaler
        self.scaler.fit(df[self.columns_to_scale])

        return self

    def transform(self, X, y=None):
        df = X.copy()

        # Impute loan_int_rate with the median
        df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

        # Drop rows where person_emp_length is NaN
        df = df.dropna(subset=['person_emp_length']).reset_index(drop=True)

        # One-hot encoding for 'person_home_ownership' and 'loan_intent'
        df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent'], drop_first=True)

        # Align the columns of one-hot encoded data with the training data
        df = df.reindex(columns=self.ohe_columns, fill_value=0)

        # Label encoding for 'cb_person_default_on_file'
        df['cb_person_default_on_file'] = self.label_encoder.transform(df['cb_person_default_on_file'])

        # Extract independent features (X)
        X = df.drop(['loan_status'], axis=1) if 'loan_status' in df.columns else df

        # Standard scaling
        X = self.scaler.transform(X)

        # Only perform train_test_split and SMOTE during training (when y is provided)
        if y is not None and len(df) > 1:  # Ensure there's more than one sample
            # Align indices between df and y
            y = y.loc[df.index].reset_index(drop=True)
            y = df['loan_status'].reset_index(drop=True) if 'loan_status' in df.columns else y

            # Split into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

            # Apply SMOTE for oversampling the minority class
            smote = SMOTE()
            X_train, y_train = smote.fit_resample(X_train, y_train)

            return X_train, y_train, X_test, y_test
        else:
            return X  # For prediction, return only the transformed features

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)

