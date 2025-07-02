import pandas as pd

ALL_FEATURES = ['X2 house age', 'X3 distance to the nearest MRT station' , 'X4 number of convenience stores', 'new_house']

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df['new_house'] = df['X2 house age'].map(lambda x: x < 3)
    return df
