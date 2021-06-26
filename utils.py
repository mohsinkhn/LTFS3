from functools import partial

import numpy as np
import pandas as pd
from pathlib import Path
import scipy as sp
from sklearn.metrics import f1_score

from constants import DATA_FOLDER, Files


def read_files(data_path):
    train = pd.read_csv(Path(DATA_FOLDER) / Files.train_csv, thousands=",")
    test = pd.read_csv(Path(DATA_FOLDER) / Files.test_csv, thousands=",")
    
    train["DisbursalDate"] = pd.to_datetime(train["DisbursalDate"])
    test["DisbursalDate"] = pd.to_datetime(test["DisbursalDate"])
    
    train_bureau = pd.read_csv(Path(DATA_FOLDER) / Files.train_bureau_csv, thousands=",")
    test_bureau = pd.read_csv(Path(DATA_FOLDER) / Files.test_bureau_csv, thousands=",")
    
    data = pd.concat([train, test])
    
    bureau = pd.concat([train_bureau, test_bureau]).reset_index(drop=True)
    bureau["DISBURSED-DT"] = pd.to_datetime(bureau["DISBURSED-DT"])
    bureau = bureau.sort_values(by=["ID", "DISBURSED-DT"])
    
    bureau = pd.merge(bureau, data[["ID", "DisbursalDate", "Tenure", "DisbursalAmount", "Frequency", "EMI"]], on="ID", how="left")
    
    return train, test, bureau


def lgb_f1_score(y_true, y_hat):
    y_hat = np.round(y_hat).clip(0, 6)
    return 'f1', f1_score(y_true, y_hat, average='macro'), True


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            elif pred >= coef[4] and pred < coef[5]:
                X_p[i] = 5
            elif pred >= coef[5]:
                X_p[i] = 6

        ll = f1_score(y, X_p, average="macro")
        return -1*ll

    def fit(self, X, y):
        loss_partial = partial(self._loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.2, 4.2, 5.0]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            elif pred >= coef[4] and pred < coef[5]:
                X_p[i] = 5
            elif pred >= coef[5]:
                X_p[i] = 6
        return X_p

    def coefficients(self):
        return self.coef_['x']
