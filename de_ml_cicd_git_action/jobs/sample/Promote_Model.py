import time

DEMO_MODEL_NAME = "de_ml-cicd-actions-model"

client = mlflow.tracking.MlflowClient()

try:
  client.create_registered_model(DEMO_MODEL_NAME)
except Exception as e:
  pass

model_version = client.create_model_version(
  DEMO_MODEL_NAME, 
  "%s/model" % best_run["artifact_uri"], best_run["run_id"]
)
time.sleep(4) # wait a bit for mlflow API 
client.update_model_version(
  DEMO_MODEL_NAME, model_version.version,
  description="Demo Random Forest Model"
)

client.transition_model_version_stage(DEMO_MODEL_NAME, model_version.version, stage="Staging")