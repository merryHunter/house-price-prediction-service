# Deployment considerations

## What approach would you use to version and track different models in production?

Model versioning can be achieved by using a consistent pattern of model_id generation, currently this is simply timestamp, but it can 
be more complex, to include branch name and follow calendar or semantic versioning. Ideally to store models in centralized
model registry  like MLFlow or W&B. Alternatively utilizing object storage like S3 with correctly generated model ID prefix for lineage back to 
training code and data versions.

Tracking models in production is done via writing relevant model_id together with storing predictions. In our case, we can add column `model_id`
to file `predictions.csv` where we write model predictions to track which model predicted. Similarly, in production environment we would have a central table
where multiple models could write predictions. Eventually, this table can be used for monitoring performance and compare model variants performance.

## What key metrics would you monitor for this API service and the prediction model?

As for API Service we need to monitor standard system metrics (health check, CPU load, disk usage load), 
input requests rate per second / hour, request latency response time.

We can also monitor number of invalid requests or errors.

For prediction model we can monitor latency, compute feature drift (especially for numerical features) and model prediction drift to identify any problems.
Drifts can be easily computed using aggregated windows and relative to a sample from the train set.

Model performance wise, R^2 score is a common metric for monitoring performance of a regression model. Dashboards can include residual plot to see error.
