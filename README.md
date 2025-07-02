# House Pricing Prediction Service

This is a demo of a simple linear regression model.

## Pre requisites

`pip install -r requirements.txt`

## Training model

`python .\train.py '..\data\Real estate.csv' `

Output of model training will be a joblib model artifact (scikit-learn pipeline object) in folder `ml/models/house_price_<timestamp>.joblib`

## Running tests

`python -m pytest tests/test.py`

## Running FastAPI service

`python -m  fastapi run app/service.py`

To try out api, open [http://127.0.0.1:8000/docs#/default/predict_predict_post](http://127.0.0.1:8000/docs#/default/predict_predict_post) . Insert sample JSON:

```
{
    "X2 house age": 5,
    "X3 distance to the nearest MRT station": 11,
    "X4 number of convenience stores": 0
}
```


Endpoint input and prediction is written in local file `predictions.csv`.