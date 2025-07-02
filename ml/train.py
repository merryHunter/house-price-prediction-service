import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib
import sys
from datetime import datetime
from features import create_features


def train_model(dataframe_path: str) -> None:
    """ Trains simple Ridge model with simple feature preprocessing """

    df = pd.read_csv(dataframe_path)
    df = create_features(df)
    
    X_train = df[['X2 house age', 'X3 distance to the nearest MRT station' , 'X4 number of convenience stores', 'new_house']]
    y_train = df[['Y house price of unit area']]
    print(f"Training Regression model with following features: \n{X_train.columns}")
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    print(f"Doing train_test_split. \nTrain dataset info: {X_train.shape[0]} samples, \nvalidation: {X_val.shape[0]}")
    ml_pipeline = make_pipeline(StandardScaler(), Ridge())
    ml_pipeline.fit(X_train, y_train)
    X_pred = ml_pipeline.predict(X_val)
    val_r2 = r2_score(X_pred, y_val)
    ts = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    print(f"Validation R^2 score: {val_r2}")
    model_save_path = f'models/price_model_{ts}.joblib'
    success = joblib.dump(ml_pipeline, model_save_path )
    if success:
        print(f"Model saved to: {model_save_path}")


if __name__ == "__main__":
    train_model(sys.argv[1])
