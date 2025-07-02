import pytest
import os
import tempfile
import pandas as pd
from fastapi.testclient import TestClient
from app.service import HousePricePredictor, app, write_prediction


@pytest.fixture
def predictor():
    """Create a fresh HousePricePredictor instance for testing"""
    return HousePricePredictor()


@pytest.fixture
def client():
    """Create a test client for FastAPI"""
    return TestClient(app)


@pytest.fixture
def existing_model_path():
    """Get path to an existing model file"""
    model_path = os.getenv('MODEL_PATH', None)
    if model_path is None:
        # set model path in .env file
        with open('.env', 'w') as f:
            f.write('MODEL_PATH=ml/models/price_model_2025-07-02-12-50-34.joblib')
        model_path = 'ml/models/price_model_2025-07-02-12-50-34.joblib'
    return model_path


class TestModelLoading:
    def test_model_loading_good_path(self, predictor, existing_model_path):
        """Test loading model from valid path"""
        assert not predictor._model_loaded
        predictor.load_model(existing_model_path)
        assert predictor._model_loaded
        assert predictor._model_version == existing_model_path
        assert predictor._ml_pipeline is not None

    def test_model_loading_nonexistent_path(self, predictor):
        """Test loading model from non-existent path"""
        nonexistent_path = "nonexistent_model.joblib"
        assert not predictor._model_loaded
        predictor.load_model(nonexistent_path)
        assert not predictor._model_loaded
        assert predictor._ml_pipeline is None

    def test_model_loading_already_loaded(self, predictor, existing_model_path):
        """Test loading same model twice"""
        predictor.load_model(existing_model_path)
        assert predictor._model_loaded
        
        # Try to load the same model again
        predictor.load_model(existing_model_path)
        assert predictor._model_loaded


class TestModelInference:
    @pytest.mark.asyncio
    async def test_model_inference_ok(self, predictor, existing_model_path):
        """Test inference with loaded model"""
        predictor.load_model(existing_model_path)
        
        test_data = {
            'X2 house age': 5,
            'X3 distance to the nearest MRT station': 100,
            'X4 number of convenience stores': 3
        }
        
        result = await predictor.predict_price("test_id", test_data)
        assert isinstance(result, float)
        assert result > 0  # Should return a positive price


class TestValidateBoundaries:
    def test_validate_boundaries_valid_input(self, predictor):
        """Test validation with valid boundaries"""
        valid_payload = {
            'X2 house age': 50,
            'X4 number of convenience stores': 10
        }
        assert predictor.validate_boundaries(valid_payload) is True

    def test_validate_boundaries_age_too_high(self, predictor):
        """Test validation with age exceeding maximum"""
        invalid_payload = {
            'X2 house age': 150,  # > MAX_AGE (100)
            'X4 number of convenience stores': 10
        }
        assert predictor.validate_boundaries(invalid_payload) is False

    def test_validate_boundaries_both_limits_exceeded(self, predictor):
        """Test validation with both limits exceeded"""
        invalid_payload = {
            'X2 house age': 150,
            'X4 number of convenience stores': 25
        }
        assert predictor.validate_boundaries(invalid_payload) is False

    def test_validate_boundaries_edge_cases(self, predictor):
        """Test validation at boundary values"""
        # Test exact limits (should be valid)
        edge_payload = {
            'X2 house age': 100,  # Exactly MAX_AGE
            'X4 number of convenience stores': 20  # Exactly MAX_STORES
        }
        assert predictor.validate_boundaries(edge_payload) is True


class TestFastAPIPredict:
    def test_predict_valid_input(self, client):
        """Test predict endpoint with valid input"""
        valid_payload = {
            "X2 house age": 5,
            "X3 distance to the nearest MRT station": 100,
            "X4 number of convenience stores": 3
        }
        
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == 1
        assert isinstance(data["price"], float)
        assert data["price"] > 0

    def test_predict_missing_required_field(self, client):
        """Test predict endpoint with missing required field"""
        incomplete_payload = {
            "X2 house age": 5,
            "X3 distance to the nearest MRT station": 100
            # Missing "X4 number of convenience stores"
        }
        
        response = client.post("/predict", json=incomplete_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == 0
        assert data["price"] == -1.0
        assert "X4 number of convenience stores" in data["message"]
        assert "missing" in data["message"]

    def test_predict_invalid_boundaries(self, client):
        """Test predict endpoint with invalid boundary values"""
        invalid_payload = {
            "X2 house age": 150,  # Exceeds MAX_AGE
            "X3 distance to the nearest MRT station": 100,
            "X4 number of convenience stores": 3
        }
        
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == 0
        assert data["price"] == -1.0
        assert "Invalid boundaries" in data["message"]

    def test_health_endpoint(self, client):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == 200


class TestWritePrediction:
    @pytest.mark.asyncio
    async def test_write_prediction_to_file(self):
        """Test that predictions are correctly written to file"""
        test_filename = "test_predictions.csv"
        test_input = [5, 100, 3, True]  # Sample feature values including new_house
        test_prediction = 42.5
        test_request_id = "test_request_123"
        
        await write_prediction(test_filename, test_input.copy(), test_prediction, test_request_id)
        
        # Verify correct data was written
        with open(test_filename, 'r') as f:
            content = f.read()
            assert "5,100,3,True,42.5,test_request_123" in content

    @pytest.mark.asyncio
    async def test_write_prediction_file_creation(self):
        """Test that prediction writing handles file operations correctly"""
        test_input = [10, 200, 5, False]
        test_prediction = 35.7
        test_request_id = "test_request_456"
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
            temp_filename = temp_file.name
        
        try:
            # Test actual file writing
            await write_prediction(temp_filename, test_input.copy(), test_prediction, test_request_id)
            
            # Verify file content
            with open(temp_filename, 'r') as f:
                content = f.read()
                assert "10,200,5,False,35.7,test_request_456" in content
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)