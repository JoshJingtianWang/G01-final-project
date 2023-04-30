# Databricks notebook source
#test

# COMMAND ----------

# MAGIC %run ./includes/includes

# COMMAND ----------

# MAGIC %run ./includes/globals

# COMMAND ----------

start_date = str(dbutils.widgets.get('01.start_date'))
end_date = str(dbutils.widgets.get('02.end_date'))
hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
promote_model = bool(True if str(dbutils.widgets.get('04.promote_model')).lower() == 'yes' else False)

print(start_date,end_date,hours_to_forecast, promote_model)
print("YOUR CODE HERE...")

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
# MAGIC # Archive and delete all previous models

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

# split data into train/test
train_ratio = 0.8

# Calculate the split index
split_index = int(len(data_df) * train_ratio)

# Split the data into train and test sets
train_df = data_df.iloc[:split_index]
test_df = data_df.iloc[split_index:]

# COMMAND ----------

# MAGIC %md
# MAGIC # Generate synthetic data

# COMMAND ----------

# def generate_bike_changes(start_date, num_days, num_intervals):
#     data = []
#     curr_date = start_date

#     # Define holidays (e.g., New Year's Day, Christmas Day)
#     holidays = [datetime(start_date.year, 1, 1), datetime(start_date.year, 12, 25)]

#     for day in range(num_days):
#         for interval in range(num_intervals):
#             # Add a sine wave component to the bike change values
#             cycle_length = 24
#             amplitude = 10
#             net_bike_change = amplitude * np.sin(2 * np.pi * interval / cycle_length)
#             net_bike_change += random.uniform(-3, 3)  # Add some random noise

#             # # Add holiday column
#             # is_holiday = 1 if curr_date.date() in [holiday.date() for holiday in holidays] else 0

#             # Add weather data (temperature and precipitation)
#             temperature = random.uniform(0, 100)  # Random temperature between 0 and 100 degrees Fahrenheit
#             precipitation = random.uniform(0, 1)  # Random precipitation between 0 and 1 inches

#             data.append((curr_date, net_bike_change, temperature, precipitation))
#             curr_date += timedelta(hours=1)

#     return data

# # Generate synthetic data
# start_date = datetime(2020, 1, 1)
# num_days = 365
# num_intervals = 24  # Hourly intervals

# random.seed(42)
# data = generate_bike_changes(start_date, num_days, num_intervals)

# # Create a Pandas dataframe
# data_df = pd.DataFrame(data, columns=["ds", "y", "temperature", "precipitation"])

# data_df.head()

# COMMAND ----------

# x = data_df.ds[:100]
# y = data_df.y[:100]
# plt.plot(x, y)
# plt.show()

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
# MAGIC # models

# COMMAND ----------

## Helper routine to extract the parameters that were used to train a specific instance of the model
def extract_params(pr_model):
    return {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}

# COMMAND ----------

# MAGIC %md
# MAGIC ## base model

# COMMAND ----------

GROUP_NAME = 'G01'
ARTIFACT_PATH = f"{GROUP_NAME}_model"
params = {} #empty param dict for base model
# Start an MLflow run
with mlflow.start_run():
    base_model = Prophet(**params) 
    base_model.add_country_holidays(country_name='US')
    base_model.add_regressor('temperature')
    base_model.add_regressor('feels_like')
    base_model.add_regressor('precipitation')
    base_model.fit(train_df)

    # Cross-validation
    df_cv = cross_validation(
                            model=base_model,
                            horizon="7 days",
                            period="60 days",
                            initial="120 days",
                            parallel="threads",
                            #disable_tqdm=False,
                        )
    # Model performance
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmse = df_p['rmse'].values[0]
    
    params = extract_params(base_model)

    # Log the best hyperparameters and the RMSE metric in MLflow
    mlflow.prophet.log_model(base_model, artifact_path=ARTIFACT_PATH)
    #mlflow.log_params(params)
    mlflow.log_metric("rmse", rmse)
    model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)
    print(f"Model artifact logged to: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### register base model

# COMMAND ----------

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

# COMMAND ----------

#add model descriptions
client = MlflowClient()
client.update_registered_model(
  name=model_details.name,
  description="The model forecasts the number of bikes that will be rented in the near future.",
)
client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version is the baseline model."
)

# COMMAND ----------

# set model stage
client.transition_model_version_stage(
  name=model_details.name,
  version=model_details.version,
  stage='Production',
)
model_version_details = client.get_model_version(
  name=model_details.name,
  version=model_details.version,
)
model_version_details.current_stage

# COMMAND ----------

model_version_details.version

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tuned model

# COMMAND ----------

#function for gridsearch
def gridsearch_run(param_grid, data):
    # Set up parameter grid
    # Generate all combinations of parameters
    rmses = [] 
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    for i, params in enumerate(all_params):
        print(f"Training model {i+1}/{len(all_params)}")
        model = Prophet(**params) 
        #holidays = pd.DataFrame({"ds": [], "holiday": []})
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

    min_index = np.argmin(rmses)
    min_rmse = rmses[min_index]
    best_params = all_params[min_index]
    
    return best_params, min_rmse

#param grid for gridsearch
param_grid_for_gridsearch = {  
    'changepoint_prior_scale': [0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0],
    'holidays_prior_scale': [0.01, 0.1, 1.0],
    'changepoint_range': [0.8, 0.9, 1.0],
    # 'daily_seasonality': [True, False],
    # 'weekly_seasonality': [True, False],
    # 'yearly_seasonality': [True, False],
    # 'growth': ['linear', 'flat'],
    # 'seasonality_mode': ['additive', 'multiplicative']
}

# COMMAND ----------

3*3*3*3

# COMMAND ----------

# Perform hyperparameter tuning with MLflow tracking

GROUP_NAME = 'G01'
ARTIFACT_PATH = f"{GROUP_NAME}_model"
optimize_method = 'gridsearch'


# Start an MLflow run
with mlflow.start_run():

    if optimize_method == 'hyperopt':
        best_params = hyperopt_run(search_space_for_hyperopt)

    elif optimize_method == 'gridsearch':
        best_params, min_rmse = gridsearch_run(param_grid_for_gridsearch, train_df)

    else:
        raise ValueError('optimize_method not supported')

    print()
    print()
    print()
    print("Best params:", best_params)

    # Train the best model using the best hyperparameters
    best_model = Prophet(**best_params) 
    best_model.add_country_holidays(country_name='US')
    best_model.add_regressor('temperature')
    best_model.add_regressor('feels_like')
    best_model.add_regressor('precipitation')
    best_model.fit(train_df)

    # Log the best hyperparameters and the RMSE metric in MLflow
    mlflow.prophet.log_model(best_model, artifact_path=ARTIFACT_PATH)
    mlflow.log_params(best_params)
    mlflow.log_metric("rmse", min_rmse)
    model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)
    print(f"Model artifact logged to: {model_uri}")

# COMMAND ----------

#register model
model_details = mlflow.register_model(model_uri=model_uri, name=ARTIFACT_PATH)

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

# COMMAND ----------

# add model description

client = MlflowClient()
client.update_registered_model(
  name=model_details.name,
  description="The model forecasts the number of bikes that will be rented in the near future.",
)
client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version has been tuned via Gridsearch."
)

# COMMAND ----------

# set stage to "Staging"
client.transition_model_version_stage(
  name=model_details.name,
  version=model_details.version,
  stage='Staging',
)
model_version_details = client.get_model_version(
  name=model_details.name,
  version=model_details.version,
)
model_version_details.current_stage

# COMMAND ----------

model_version_details.version

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transition to production

# COMMAND ----------

# client.transition_model_version_stage(
#   name=model_details.name,
#   version=model_details.version,
#   stage='Production',
# )

# COMMAND ----------

# MAGIC %md
# MAGIC # Load registered models

# COMMAND ----------

GROUP_NAME = 'G01'
ARTIFACT_PATH = f"{GROUP_NAME}_model"

model_production_uri = "models:/{model_name}/production".format(model_name=ARTIFACT_PATH)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_production_uri))

model_production = mlflow.prophet.load_model(model_production_uri)

model_staging_uri = "models:/{model_name}/staging".format(model_name=ARTIFACT_PATH)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_staging_uri))

model_staging = mlflow.prophet.load_model(model_staging_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC # Inferencing and residual plot

# COMMAND ----------

test_df.head()

# COMMAND ----------

production_forecast = model_production.predict(test_df)
staging_forecast = model_staging.predict(test_df)

# COMMAND ----------

production_forecast.head()

# COMMAND ----------

# results=production_forecast[['ds','yhat']].join(test_df, on = "ds")
#                                                 #lsuffix='_caller', 
#                                                 # rsuffix='_other'
temp_df = test_df[['ds', 'y']].reset_index(drop = True)      
# results['yhat_production'] = production_forecast['yhat']                                          
# results['residual_production'] = production_forecast['yhat'] - results['y']
# results['yhat_staging'] = production_forecast['yhat']        
# results['residual_staging'] = staging_forecast['yhat'] - results['y']

results_production = pd.DataFrame()
results_production['yhat'] = production_forecast['yhat']
results_production['residual'] = production_forecast['yhat'] - temp_df['y']
results_production['label'] = 'Production'

results_staging = pd.DataFrame()
results_staging['yhat'] = staging_forecast['yhat']
results_staging['residual'] = staging_forecast['yhat'] - temp_df['y']
results_staging['label'] = 'Staging'

results = pd.concat([results_production, results_staging], axis=0, ignore_index=True)


# COMMAND ----------

results.head()

# COMMAND ----------

results.tail()

# COMMAND ----------

#plot the residuals
fig = px.scatter(
    results, x='yhat', y='residual',
    marginal_y='violin',
    trendline='ols',
    color = 'label',
    opacity=0.5 
)
fig.show()

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
