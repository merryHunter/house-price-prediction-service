# House Pricing Prediction Service

This is a demo of a simple linear regression model. 

## Pre requisites

`python -m pip install -r requirements-dev.txt`

Main dependencies are Pandas and Scikit-learn.

## Training model

`python .\train.py '..\data\Real estate.csv' `

Rigde Regression model is traing on housing data. Output of model training will be a joblib model artifact (scikit-learn pipeline object) in folder `ml/models/house_price_<timestamp>.joblib`.

Feature engineering is extremely basic only to show re-use of feature transformation during training and inference stages. Potentially, pairwise features, distance
to closest city, more boolean flags can be computed. 

## Running tests

`python -m pytest tests/test.py`

## Running FastAPI service

From root directory, launch the following command:  
`python -m  fastapi run app/service.py`

Model loading is done by reading .env file containing path to the model. Sample model is already provided.

Input is validated for missing fields, boundary values and null values.

Service paylod and house price is written to `predictions.csv`.

To try out API service, open [http://127.0.0.1:8000/docs#/default/predict_predict_post](http://127.0.0.1:8000/docs#/default/predict_predict_post) . Insert sample JSON:

```
{
    "X2 house age": 5,
    "X3 distance to the nearest MRT station": 11,
    "X4 number of convenience stores": 0
}
```


Endpoint input and prediction is written in local file `predictions.csv`.

## Docker

Minimal docker image with launching multiple uvicorn workers can be built and run :

```
docker build -t ml_service .
docker run -p 8000:8000 ml_service
```

## Disclaimer on AI usage

AI assisted IDE Cursor was used only during test development, Github Actions and Dockerfile. A few Scikit learn, Pandas, pytest and FastAPI reference documentation pages were accessed during rest of the development.

## Recording

Development session is recorded, 10X speed up video is stored at `assets/recording_10x.mp4`.