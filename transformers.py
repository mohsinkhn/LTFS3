import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SelectCols(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.cols]


class BureauFeatureMapper(BaseEstimator, TransformerMixin):
    def __init__(self, bureau_df, id_col, agg_col, filter_cond=None, agg_func=len):
        self.bureau_df = bureau_df
        self.id_col = id_col
        self.agg_col = agg_col
        self.filter_cond = filter_cond
        self.agg_func = agg_func

        if self.filter_cond is None:
            self.filter_cond = []
        self.feature_dict = None

    def fit(self, X, y=None):
        cond = 1
        for col, value in self.filter_cond:
            cond &= self.bureau_df[col] == value
        if isinstance(self.agg_func, str):
            self.feature_dict = getattr(self.bureau_df.loc[cond].groupby(self.id_col)[self.agg_col], self.agg_func)().to_dict()
        else:
            self.feature_dict = self.bureau_df.loc[cond].groupby(self.id_col)[self.agg_col].apply(self.agg_func).to_dict()
        return self

    def transform(self, X, y=None):
        return X[self.id_col].map(self.feature_dict)
