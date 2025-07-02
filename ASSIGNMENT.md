# Take-Home Task: Deploy a House Price Prediction Service (3-Hour Time Limit)

## Background
This assignment is designed to be completed within 3 hours. Focus on the core requirements first before attempting any bonus tasks.

You've joined a startup building microservices for real estate analytics. You're asked to build a service that predicts house prices based on basic features (e.g., metro location, location score, etc.). This service should be easy to test, deploy, and monitor in production.

## Task Overview
You will:
1. Train a simple model on a housing dataset.
2. Serve it via a basic REST API.
3. Write essential tests for the model.
4. Answer key deployment questions.

## Provided Resources
- A CSV file with house price data (included with this assignment).

## Assignment Structure
Please organize your submission with the following structure:
```
/
├── app/           # API service code
├── data/          # Data files (pre-filled with provided CSV)
├── docs/          # Documentation
├── ml/            # Model training code
└── tests/         # Test files
```

### 1. Model Training (in /ml)
- Load housing dataset.
- Implement minimal feature engineering and train a simple model (don't focus on scoring the model).
- Save the model to disk.

### 2. REST API (in /app)
- Build an API with any web framework of your choice.
- Endpoints:
  - `POST /predict` — accepts JSON input for features, returns predicted price.
  - `GET /health` — returns a static OK response.
- Ensure request input is validated.
- Implement basic logging functionality for API requests and responses.

### 3. Testing (in /tests)
- Write basic tests to verify model loading and prediction.

### 4. Deployment Considerations (in /docs/DEPLOYMENT.md)
Write brief answers (1-2 paragraphs each) to these two basic questions:
- What approach would you use to version and track different models in production?
- What key metrics would you monitor for this API service and the prediction model?

## Constraints
- Keep it under 500 lines of total code (including tests).
- You can use any libraries or frameworks of your choice. Include requirements.txt or package files as appropriate.

## Submission Instructions
1. Create a GitHub repository with your solution.
2. Ensure your repository is public or provide access to the reviewers.
3. Include a README.md with:
   - Brief description of your solution
   - Instructions for running the code and tests
   - Any dependencies required

## Bonus
- Set up a CI/CD workflow (.github/workflows/) to install dependencies and run tests.
- Containerize the application.

