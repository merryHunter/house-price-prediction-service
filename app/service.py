from fastapi import FastAPI
import joblib 
import os
import logging
from logging.handlers import RotatingFileHandler
import uuid
from ml.features import create_features, ALL_FEATURES
import pandas as pd
from dotenv import load_dotenv


### ----- LOGGING  ----- ###
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('house_price_service')
filehandler = RotatingFileHandler('app.log', maxBytes=1024 * 1024, backupCount=5)
logger.addHandler(filehandler)

### ----- ENV  ----- ###
load_dotenv()
MODEL_PATH = os.getenv('MODEL_PATH', None)

### ----- PREDICTOR  ----- ###
class HousePricePredictor:
    def __init__(self):
        self._model_loaded = False
        self._ml_pipeline = None
        self._model_version = None
        self.required_fields = ['X2 house age', 'X3 distance to the nearest MRT station' , 'X4 number of convenience stores']
        self._MAX_AGE=100
        self._MAX_STORES=20
        self._result_filename = 'predictions.csv'
        csv_header = ALL_FEATURES
        csv_header.append('predicted_price')
        csv_header.append('request_id')
        csv_header.append('\n')
        with open(self._result_filename, 'w') as f:
            f.writelines(','.join(csv_header))

    
    def load_model(self, model_path: str):
        """ Load model from disk. Loading is not allowed for already loaded model. """
        if self._model_loaded and model_path == self._model_version:
            logger.warning(f"Model '{model_path}' is already loaded!")
            return
        if model_path is None:
            logger.error("MODEL_PATH is not set in environment variables")
            return
        try:    
            self._ml_pipeline = joblib.load(model_path)
            self._model_version = model_path
            self._model_loaded = True
            logger.info(f"ML Model loaded from: {model_path}")
        except Exception as e:
            logger.error(f"ML Model could not be loaded from: {model_path}. Error: {str(e)}")

    def validate_boundaries(self, payload: dict):
        if payload['X2 house age'] > self._MAX_AGE or payload['X4 number of convenience stores'] > self._MAX_STORES:
            return False
        return True

    async def predict_price(self, request_id: str, data: dict) -> float:
        df = pd.DataFrame([data])
        df = create_features(df)
        logger.debug(f"{request_id} - features created, running predict.")
        result = self._ml_pipeline.predict(df)[0][0]
        await write_prediction(self._result_filename, list(df.iloc[0]), result, request_id)
        return result
            

async def write_prediction(filename: str, input_array: list, prediction: float, request_id: str):
    input_array.append(prediction)
    input_array.append(request_id)
    input_array.append('\n')
    input_array = [str(x) for x in input_array]
    with open(filename, 'a') as f:
        f.writelines(','.join(input_array))


### ----- FASTAPI  ----- ###
app = FastAPI()

predictor = HousePricePredictor()
predictor.load_model(MODEL_PATH)


@app.get('/health')
async def health():
    return 200

@app.post('/predict')
async def predict(payload: dict):
    """ Predict endpoint to run inference of House Prediction model. 
        Input: JSON payload. Currently supported features:
        "X2 house age", "X3 distance to the nearest MRT station", "X4 number of convenience stores".
        Sample JSON:
        {
            "X2 house age": 1,
            "X3 distance to the nearest MRT station": 2,
            "X4 number of convenience stores": 0
        }
        Returns unit price (float).
        Sample output:
            {
            "success": 1,
            "price": 10.0
            }
        If problems occured:
            {
            "success: 0,
            "price": -1.0,
            "message": "input contains errors: <str> | Unexpected exception, review logs."
            }
    """
    request_id = uuid.uuid4().hex
    logger.info(f"{request_id} - Received following payload: {payload}")
    message_validation = "Input contains errors"
    message_model = "Unexpected exception, review logs"
    # validate request payload
    for field in predictor.required_fields:
        if field not in payload:
            invalid_message = f"{message_validation}: '{field}' is missing as input field."
            logger.warning(f"{request_id} - {invalid_message}")
            return {"success" : 0, "price": -1.0, "message": invalid_message}
    
    # validate boundaries
    if not predictor.validate_boundaries(payload):
        invalid_message = f"{message_validation}: Invalid boundaries of input values"
        return {"success" : 0, "price": -1.0, "message": invalid_message}

    try:
        house_price = await predictor.predict_price(request_id, payload)    
        logger.info(f"{request_id} - predicted price: {house_price}")
        return {"success" : 1, "price": house_price}
    except Exception as e:
        logger.error(f"{request_id} - Unexpected exception: {str(e)}")
        return {"success" : 0, "price": -1.0, "message": message_model}