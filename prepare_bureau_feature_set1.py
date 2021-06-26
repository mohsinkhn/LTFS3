import pandas as pd

from constants import DATA_FOLDER, Files
from utils import read_files


def get_base_features(df, bureau_df):
    bureau_df.groupby("ID").agg({"ID": "count"})
