"""
    @Author: rmyazdi
    Module to provide data cleaning services for the
    housing price project.
"""

import pandas as pd
import numpy as np
import seaborn as sns
from typing import TypeVar, List
from pandas.api.types import CategoricalDtype
from pathlib import Path

"""
    Lists of the various columns:
        1. Nominal Categories: these are categorical but with out
        any particular order in their values.
        2. Ordinal categories: categories whose values are
"""

nominals = [
    "MSSubClass",
    "MSZoning",
    "Street",
    "Alley",
    "Utilities",
    "LotConfig",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "Foundation",
    "Heating",
    "Electrical",
    "GarageType",
    "MiscFeature",
    "SaleType",
    "SaleCondition",
    "CentralAir",
]

## ordered categories:
ten_levels = list(range(10))
five_levels = ["Po", "Fa", "TA", "Gd", "Ex"]

ordered_categories = {
    "OverallQual": ten_levels,
    "OverallCond": ten_levels,
    "ExterCond": five_levels,
    "ExterQual": five_levels,
    "BsmtQual": five_levels,
    "BsmtCond": five_levels,
    "BsmtExposure": five_levels,
    "HeatingQC": five_levels,
    "KitchenQual": five_levels,
    "FireplaceQu": five_levels,
    "GarageQual": five_levels,
    "GarageCond": five_levels,
    "PoolQC": five_levels,
    "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "GarageFinish": ["Unf", "RFn", "Fin"],
    "PavedDrive": ["N", "P", "Y"],
    "LandSlope": ["Sec", "Mod", "Gtl"],
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "LandContour": ["Lvl", "Bnk", "HLS", "Low"],
    "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"],
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
}
## Add a _none_ level for each of the above, to catch missing values
ordered_categories = {
    key: ["None"] + value for key, value in ordered_categories.items()
}

## real-valued columns:
real_val_features = [
    "LotFrontage",
    "LotArea",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "FirstFlrSF",
    "SecondFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "ThreeSeasonPorch",
    "ScreenPorch",
    "PoolArea",
]

""" ****************** Functions **********************"""


def encode(dataframe: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    # Nominal categories
    for col in nominals:
        dataframe[col] = dataframe[col].astype("category")
        if "None" not in dataframe[col].cat.categories:
            if col != "MSSubClass":
                dataframe[col] = dataframe[col].cat.add_categories("None")
            else:
                dataframe[col] = dataframe[col].cat.add_categories(0)
    # Ordinal:
    for col, levels in ordered_categories.items():
        dataframe[col] = dataframe[col].astype(CategoricalDtype(levels, ordered=True))
    return dataframe


def impute(dataframe: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    df = dataframe.copy()
    for name in real_val_features:  # df.select_dtypes(np.float64):
        df[name].fillna(0.0, inplace=True)
    for name in df.select_dtypes("category"):
        df[name].fillna("None", inplace=True)
    return df


def clean(dataframe: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    df = dataframe.copy()
    df["Exterior2nd"] = df["Exterior2nd"].replace({"Brk Cmn": "BrkCmn"})
    df["GarageYrBlt"] = df["GarageYrBlt"].where(
        df["GarageYrBlt"].notna(), other=df["YearBuilt"]
    )
    df.rename(
        columns={
            "1stFlrSF": "FirstFlrSF",
            "2ndFlrSF": "SecondFlrSF",
            "3SsnPorch": "ThreeSeasonPorch",
        },
        inplace=True,
    )
    df["LotArea"].astype(np.float64)
    df["LotFrontage"] = df["LotFrontage"].fillna(0)  # df["LotFrontage"].mean())
    for indx, row in df.iterrows():
        if row.MasVnrType == "None" and row.MasVnrArea != 0:
            row.MasVnrArea = 0.0
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0.0)  # df["MasVnrArea"].mean())
    df["MSSubClass"] = df["MSSubClass"].fillna(0)
    return df


def load_data(path: Path, cat: str) -> pd.core.frame.DataFrame:
    """
    load_data():
        returns the test and train data set.
    """

    df = pd.read_csv(path / f"{cat}.csv")
    df = clean(df)
    df = encode(df)
    df = impute(df)
    return df


def add_features(df):
    ## total number of bathrooms
    df["n_baths"] = (
        df["FullBath"]
        + df["BsmtFullBath"]
        + 0.5 * (df["HalfBath"] + df["BsmtHalfBath"])
    )
    ## total liveable area?
    ## add total basement area to the above ground living area, both in sqft
    df["area_with_bsmt"] = df["TotalBsmtSF"] + df["GrLivArea"]
    df["area_with_bsmt"].astype(np.float64)
    df["Age"] = pd.to_numeric(df["YrSold"]) - pd.to_numeric(df["YearBuilt"])
    df["Renovate"] = pd.to_numeric(df["YearRemodAdd"]) - pd.to_numeric(df["YearBuilt"])
    return df


if __name__ == "__main__":
    path = Path("/Users/rmyazdi/Documents/kaggle/house_prices/data")
    train = load_data(path, "train")
    test = load_data(path, "test")

    train.to_csv(path / "train_clean.csv", index=False)
    test.to_csv(path / "test_clean.csv", index=False)

    print(
        len(nominals),
        len(ordered_categories),
        len([name for name in train.select_dtypes(np.float64)]),
    )
