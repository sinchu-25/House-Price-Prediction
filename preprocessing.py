"""
Shared preprocessing for House Price Prediction.
Both train_house_price.py and app.py import from this file so the
exact same transformations are applied at training time and at
prediction time.
"""
import numpy as np
import pandas as pd

QUAL_MAP = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
BSMT_EXPOSURE_MAP = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
BSMT_FIN_MAP = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
GARAGE_FINISH_MAP = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
LOT_SHAPE_MAP = {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}
LAND_SLOPE_MAP = {'Sev': 0, 'Mod': 1, 'Gtl': 2}
PAVED_DRIVE_MAP = {'N': 0, 'P': 1, 'Y': 2}
FENCE_MAP = {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
FUNCTIONAL_MAP = {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7}
CENTRAL_AIR_MAP = {'N': 0, 'Y': 1}

NONE_FILL = [
    'Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType',
    'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence',
]
ZERO_FILL = [
    'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'GarageCars', 'GarageArea',
]
ZERO_VAR_FEATURES = ['Street', 'Utilities', 'RoofMatl']
USELESS_FEATURES = ['MiscFeature', 'MiscVal', 'LowQualFinSF', 'Condition2', 'Heating']
ORDINAL_QUAL_FEATURES = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu']
OHE_FEATURES = [
    'MSZoning', 'Alley', 'LandContour', 'LotConfig', 'BldgType', 'HouseStyle',
    'RoofStyle', 'MasVnrType', 'Foundation', 'GarageType', 'SaleType',
    'SaleCondition', 'Electrical', 'BsmtCond',
]
TARGET_ENC_FEATURES = ['Neighborhood', 'Exterior1st', 'Exterior2nd', 'MSSubClass', 'Condition1']


def clean_and_engineer(df):
    """Stages 1-2 from the original notebook: cleaning + feature engineering.
    Input: raw dataframe with the original Kaggle House Prices columns (minus Id/SalePrice OK either way).
    Output: cleaned + engineered dataframe, still with raw categorical strings (not yet encoded)."""
    df = df.copy()

    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    df = df.drop(columns=[c for c in ZERO_VAR_FEATURES if c in df.columns])
    df = df.drop(columns=[c for c in USELESS_FEATURES if c in df.columns])

    df['MSSubClass'] = df['MSSubClass'].astype(str)

    for col in NONE_FILL:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    for col in ZERO_FILL:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if 'LotFrontage' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

    if 'GarageYrBlt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

    for col in ['Electrical', 'MSZoning', 'SaleType']:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Any remaining nulls: numeric -> 0, categorical -> 'None' (safety net for app-generated rows)
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == object:
                df[col] = df[col].fillna('None')
            else:
                df[col] = df[col].fillna(0)

    # --- Feature engineering ---
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
    df = df.drop(columns=['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold'])

    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBathrooms'] = df['FullBath'] + df['BsmtFullBath'] + 0.5 * df['HalfBath'] + 0.5 * df.get('BsmtHalfBath', 0)
    df['TotalPorchSF'] = (
        df['OpenPorchSF'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'] + df['3SsnPorch']
    )
    df = df.drop(columns=['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], errors='ignore')

    df['QualCondScore'] = df['OverallQual'] * df['OverallCond']
    df['GarageQualCond'] = df['GarageQual'].map(QUAL_MAP).fillna(0) + df['GarageCond'].map(QUAL_MAP).fillna(0)
    df = df.drop(columns=['GarageQual', 'GarageCond'], errors='ignore')

    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df = df.drop(columns=['PoolArea', 'PoolQC'], errors='ignore')
    df['HasBasement'] = (df['BsmtQual'] != 'None').astype(int)
    df = df.drop(columns=['MoSold'], errors='ignore')

    return df


def encode_features(df):
    """Stage 5 from the original notebook: ordinal + one-hot encoding.
    Target encoding is handled separately since it needs a fitted encoder."""
    df = df.copy()

    for col in ORDINAL_QUAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].map(QUAL_MAP)
    if 'BsmtExposure' in df.columns:
        df['BsmtExposure'] = df['BsmtExposure'].map(BSMT_EXPOSURE_MAP)
    for col in ['BsmtFinType1', 'BsmtFinType2']:
        if col in df.columns:
            df[col] = df[col].map(BSMT_FIN_MAP)
    if 'GarageFinish' in df.columns:
        df['GarageFinish'] = df['GarageFinish'].map(GARAGE_FINISH_MAP)
    if 'LotShape' in df.columns:
        df['LotShape'] = df['LotShape'].map(LOT_SHAPE_MAP)
    if 'LandSlope' in df.columns:
        df['LandSlope'] = df['LandSlope'].map(LAND_SLOPE_MAP)
    if 'PavedDrive' in df.columns:
        df['PavedDrive'] = df['PavedDrive'].map(PAVED_DRIVE_MAP)
    if 'Fence' in df.columns:
        df['Fence'] = df['Fence'].map(FENCE_MAP)
    if 'Functional' in df.columns:
        df['Functional'] = df['Functional'].map(FUNCTIONAL_MAP)
    if 'CentralAir' in df.columns:
        df['CentralAir'] = df['CentralAir'].map(CENTRAL_AIR_MAP)

    ohe_cols_present = [c for c in OHE_FEATURES if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols_present, drop_first=True)

    return df


def align_columns(df, expected_columns):
    """Ensure a dataframe has exactly the columns the model was trained on
    (adds missing dummy columns as 0, drops extras, fixes order)."""
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    return df[expected_columns]