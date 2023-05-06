# Databricks notebook source
# MAGIC %run ./includes/globals

# COMMAND ----------

promote_model = bool(True if str(dbutils.widgets.get('01.promote_model')).lower() == 'yes' else False)

print(promote_model)

# COMMAND ----------

# MAGIC %md
# MAGIC # import libraries

# COMMAND ----------

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import itertools
from prophet import Prophet, serialize
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.pyfunc
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import time

import matplotlib.pyplot as plt
import plotly.express as px

# COMMAND ----------

# MAGIC %md
# MAGIC # Archive and delete all previous models (optional)

# COMMAND ----------

# # Get the client object for the MLflow tracking server
# client = mlflow.tracking.MlflowClient()

# # List all registered models
# registered_models = client.list_registered_models()

# model_name = 'G01_model'

# # Get all model versions for the current registered model
# model_versions = client.search_model_versions(f"name='{model_name}'")

# # Loop through all model versions and set their stage to "Archived"
# for version in model_versions:
#     version_number = version.version
#     version_stage = version.current_stage
    
#     if version_stage != "Archived":
#         print(f"Archiving model {model_name}, version {version_number}")
#         client.transition_model_version_stage(
#             name=model_name,
#             version=version_number,
#             stage="Archived"
#         )

# client.delete_registered_model(model_name)
# print("Specified registered models have been deleted.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Load silver modeling table

# COMMAND ----------

delta_df = spark.read.format("delta").load(SILVER_MODELING_HISTORICAL_PATH)
data_df = delta_df.toPandas()

# COMMAND ----------

data_df.head()

# COMMAND ----------

# # split data into train/test
# train_ratio = 0.8

# # Calculate the split index
# split_index = int(len(data_df) * train_ratio)

# # Split the data into train and test sets
# train_df = data_df.iloc[:split_index]
# test_df = data_df.iloc[split_index:]

# COMMAND ----------

# MAGIC %md
# MAGIC # Models

# COMMAND ----------

## Helper routine to extract the parameters that were used to train a specific instance of the model
def extract_params(pr_model):
    return {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}

# COMMAND ----------

# MAGIC %md
# MAGIC ## setting up function for training base model

# COMMAND ----------

# Start an MLflow run
def train_base_model(ARTIFACT_PATH, params, data):
    with mlflow.start_run():

        # Initialize and fit the model
        base_model = Prophet(**params) 
        base_model.add_country_holidays(country_name='US')
        base_model.add_regressor('temperature')
        base_model.add_regressor('feels_like')
        base_model.add_regressor('precipitation')
        base_model.fit(data)

        # Cross-validation
        df_cv = cross_validation(
                                model=base_model,
                                horizon="7 days",
                                period="60 days",
                                initial="120 days",
                                parallel="threads",
                                #disable_tqdm=False,
                            )
        # Get model performance
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmse = df_p['rmse'].values[0]

        # Log the model
        mlflow.prophet.log_model(base_model, artifact_path=ARTIFACT_PATH)
        
        # Log the best hyperparameters
        params = extract_params(base_model)
        #remove this attribute because it is too long to be logged. throws "RestException: INVALID_PARAMETER_VALUE"
        del params["component_modes"]
        mlflow.log_params(params)

        # Log the RMSE metric
        mlflow.log_metric("rmse", rmse)

        model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)
        print(f"Model artifact logged to: {model_uri}")
    
    return model_uri

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting up function for training tuned model

# COMMAND ----------

#function for gridsearch
def gridsearch_run(param_grid, data):
    
    # Set up parameter grid. Generate all combinations of parameters
    rmses = [] 
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    for i, params in enumerate(all_params):
        
        print(f"\n\nTraining model {i+1}/{len(all_params)}:\n")

        with mlflow.start_run():
            # Initialize and fit the model
            model = Prophet(**params) 
            model.add_country_holidays(country_name='US')
            model.add_regressor('temperature')
            model.add_regressor('feels_like')
            model.add_regressor('precipitation')
            model.fit(data)
            
            # Cross-validation
            df_cv = cross_validation(
                                    model=model,
                                    horizon="7 days",
                                    period="60 days",
                                    initial="120 days",
                                    parallel="threads",
                                    #disable_tqdm=False,
                                )
            
            # Model performance
            df_p = performance_metrics(df_cv, rolling_window=1)

            # Save model performance metrics for this combination of hyper parameters
            rmses.append(df_p['rmse'].values[0])

            # Log the model
            mlflow.prophet.log_model(model, artifact_path=ARTIFACT_PATH)
            
            # Log the best hyperparameters
            params = extract_params(model)

            #remove this attribute because it is too long to be logged. throws "RestException: INVALID_PARAMETER_VALUE"
            del params["component_modes"]

            mlflow.log_params(params)

            # Log the RMSE metric
            mlflow.log_metric("rmse", df_p['rmse'])

    # getting parameters with the best RMSE
    min_index = np.argmin(rmses)
    min_rmse = rmses[min_index]
    best_params = all_params[min_index]
    
    return best_params, min_rmse

#param grid for gridsearch
param_grid_for_gridsearch = {  
    'changepoint_prior_scale': [0.001, 0.01, 1],
    'seasonality_prior_scale': [15.0, 20.0, 30.0], 
    # for all hyperparams below
    # we have determined these to be the best
    # setting them to fixed values for speed
    'holidays_prior_scale': [7], 
    'changepoint_range': [1],
    'daily_seasonality': [True],
    'weekly_seasonality': [False],
    'yearly_seasonality': [False],
    'growth': ['linear'],
    'seasonality_mode': ['additive']
}

def train_tuned_model(ARTIFACT_PATH, data):

    # Perform hyperparameter gridsearch
    best_params, min_rmse = gridsearch_run(param_grid_for_gridsearch, data)

    print(f"\n\nBest params: {best_params}")
    print(f"Best rmse: {min_rmse}\n\n")

    #fit the model with the best parameters
    with mlflow.start_run():
        # Train the best model using the best hyperparameters
        best_model = Prophet(**best_params) 
        best_model.add_country_holidays(country_name='US')
        best_model.add_regressor('temperature')
        best_model.add_regressor('feels_like')
        best_model.add_regressor('precipitation')
        best_model.fit(data)

        # Log the model
        mlflow.prophet.log_model(best_model, artifact_path=ARTIFACT_PATH)
        
        # Log the best hyperparameters
        params = extract_params(best_model)

        #remove this attribute because it is too long to be logged. throws "RestException: INVALID_PARAMETER_VALUE"
        del params["component_modes"]

        mlflow.log_params(params)

        # Log the RMSE metric
        mlflow.log_metric("rmse", min_rmse)
        
        model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)
        print(f"Model artifact logged to: {model_uri}")

    return model_uri

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting up function for registering models

# COMMAND ----------

def register_model(model_type, staging, model_uri):

    #register base model
    model_details = mlflow.register_model(model_uri=model_uri, name=ARTIFACT_PATH)

    # After creating a model version, it may take a short period of time to become ready. 
    def wait_until_ready(model_name, model_version):
        client = MlflowClient()
        for _ in range(10):
            model_version_details = client.get_model_version(
            name=model_name,
            version=model_version,
            )
            status = ModelVersionStatus.from_string(model_version_details.status)
            print("Model status: %s" % ModelVersionStatus.to_string(status))
            if status == ModelVersionStatus.READY:
                break
            time.sleep(1)
    
    wait_until_ready(model_details.name, model_details.version)

    #add model descriptions
    client = MlflowClient()
    client.update_registered_model(
    name=model_details.name,
    description="The model forecasts the number of bikes that will be rented in the near future.",
    )
    client.update_model_version(
    name=model_details.name,
    version=model_details.version,
    description=f"This model version is the {model_type} model."
    )

    # set model stage
    client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage=staging,
    )
    # model_version_details = client.get_model_version(
    # name=model_details.name,
    # version=model_details.version,
    # )

# COMMAND ----------

# MAGIC %md
# MAGIC ## training and registering models
# MAGIC If there are currently no production models, train a baseline model and register it.
# MAGIC
# MAGIC If there is a production model, train a tuned model and register it.

# COMMAND ----------

GROUP_NAME = 'G01'
ARTIFACT_PATH = f"{GROUP_NAME}_model"
params = {} #empty param dict for base model

client = MlflowClient()
registered_models = client.list_registered_models()

try:
    production_versions = client.get_latest_versions(ARTIFACT_PATH, stages=["Production"])

    if production_versions: #if there are production models
        print('\n\nThere is currently a production model. Proceeding to train a tuned model.\n\n')
        model_type = "tuned"
        staging = "Staging"
        model_uri = train_tuned_model(ARTIFACT_PATH, data_df)
        print('\n\nTraining complete. Registering the model as the staging model.\n\n')
        register_model(model_type, staging, model_uri)
    else: #if there are no production models
        print('\n\nThere are currently no production models. Proceeding to train a baseline model.\n\n')
        model_type = "baseline"
        staging = "Production"
        model_uri = train_base_model(ARTIFACT_PATH, params, data_df)
        print('\n\nTraining complete. Registering the model as the production model.\n\n')
        register_model(model_type, staging, model_uri)

except mlflow.exceptions.RestException: #if there are no models named G01_model
    print('\n\nThere are currently no models. Proceeding to train a baseline model.\n\n')
    model_type = "baseline"
    staging = "Production"
    params = {}
    model_uri = train_base_model(ARTIFACT_PATH, params, data_df)
    print('\n\nTraining complete. Registering the model as the production model.\n\n')
    register_model(model_type, staging, model_uri)


# COMMAND ----------

# MAGIC %md
# MAGIC # Load registered models

# COMMAND ----------

# GROUP_NAME = 'G01'
# ARTIFACT_PATH = f"{GROUP_NAME}_model"

# model_production_uri = "models:/{model_name}/production".format(model_name=ARTIFACT_PATH)

# print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_production_uri))

# model_production = mlflow.prophet.load_model(model_production_uri)

# model_staging_uri = "models:/{model_name}/staging".format(model_name=ARTIFACT_PATH)

# print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_staging_uri))

# model_staging = mlflow.prophet.load_model(model_staging_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC # Inferencing and residual plot (optional)

# COMMAND ----------

# production_forecast = model_production.predict(test_df)
# staging_forecast = model_staging.predict(test_df)

# COMMAND ----------

# production_forecast.head()

# COMMAND ----------

# temp_df = test_df[['ds', 'y']].reset_index(drop = True)

# results_production = pd.DataFrame()
# results_production['yhat'] = production_forecast['yhat']
# results_production['residual'] = production_forecast['yhat'] - temp_df['y']
# results_production['label'] = 'Production'

# results_staging = pd.DataFrame()
# results_staging['yhat'] = staging_forecast['yhat']
# results_staging['residual'] = staging_forecast['yhat'] - temp_df['y']
# results_staging['label'] = 'Staging'

# results = pd.concat([results_production, results_staging], axis=0, ignore_index=True)

# COMMAND ----------

# results.head()

# COMMAND ----------

# #plot the residuals
# fig = px.scatter(
#     results, x='yhat', y='residual',
#     marginal_y='violin',
#     trendline='ols',
#     color = 'label',
#     opacity=0.5 
# )
# fig.show()

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
