# Databricks notebook source
import mlflow
import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

from mlflow.tracking.client import MlflowClient

client = MlflowClient()

model_name = "de_ml-cicd-actions-model"
# Transition a model version
client.transition_model_version_stage(
  name=model_name,
  version=1,
  stage="Production",
)
