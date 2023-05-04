# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import time

# COMMAND ----------

# DBTITLE 0,YOUR APPLICATIONS CODE HERE...
#start_date = str(dbutils.widgets.get('01.start_date'))
#end_date = str(dbutils.widgets.get('02.end_date'))
#hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
#promote_model = bool(True if str(dbutils.widgets.get('04.promote_model')).lower() == 'yes' else False)

#print(start_date,end_date,hours_to_forecast, promote_model)



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The following shows the current timestamp when the notebook is run (now).

# COMMAND ----------

from datetime import datetime

now = datetime.now()
print("Current timestamp:", now)

# COMMAND ----------

# MAGIC %md The following shows the current Production Model version. 

# COMMAND ----------

 # Get the client object for the MLflow tracking server
 client = mlflow.tracking.MlflowClient()

 # List all registered models
 registered_models = client.search_registered_models()

 model_name = 'G01_model'

 # Get all model versions for the current registered model
 model_versions = client.search_model_versions(f"name='{model_name}'")
 model_versions

# COMMAND ----------

 # Get the client object for the MLflow tracking server
 client = mlflow.tracking.MlflowClient()

 # List all registered models
 registered_models = client.list_registered_models()

 model_name = 'G01_model'

 # Get all model versions for the current registered model
 model_versions = client.search_model_versions(f"name='{model_name}'")

 # Loop through all model versions and set their stage to "Archived"
 for version in model_versions:
     version_number = version.version
     version_stage = version.current_stage
    
     if version_stage != "Archived":
         print(f"Archiving model {model_name}, version {version_number}")
         client.transition_model_version_stage(
             name=model_name,
             version=version_number,
             stage="Archived"
         )

 #client.delete_registered_model(model_name)
 #print("Specified registered models have been deleted.")

# COMMAND ----------

GROUP_NAME = 'G01'
ARTIFACT_PATH = f"{GROUP_NAME}_model"
model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)

# COMMAND ----------



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

# MAGIC %md The following shpws the current Staging Model version.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC The following map shows the location of our station, with a marker. When the marker is clicked, the station name pops up: W 21 St & 6 Ave.

# COMMAND ----------

# MAGIC %pip install folium

# COMMAND ----------

import folium

# Latitude and longitude of the station
lat = 40.7417409809474
lon = -73.99415683746338

# Create the map object centered on the station location
map = folium.Map(location=[lat, lon], zoom_start=15)

# Add a marker to the station location
folium.Marker([lat, lon], popup='W 21 St & 6 Ave').add_to(map)

# Display the map
display(map)

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
