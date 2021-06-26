"""All constants for project go here."""


DATA_FOLDER = "data"


class Files:
    """Easy retrieval of file paths."""
    train = "Train/train_Data.xlsx"
    test = "test_Data.xlsx"
    data_dict = "Train/data_dict.xlsx"
    train_csv = "train_Data.csv"
    test_csv = "test_Data.csv"
    train_bureau = "Train/train_bureau.xlsx"
    test_bureau = "test_bureau.xlsx"
    train_bureau_csv = "train_bureau.csv"
    test_bureau_csv = "test_bureau.csv"


TargetMap = {
    "No Top-up Service": 6,
    " > 48 Months": 5,
    "36-48 Months": 4,
    "30-36 Months": 3,
    "24-30 Months": 2,
    "18-24 Months": 1,
    "12-18 Months": 0
}

TargetRevMap = {v: k for k, v in TargetMap.items()}