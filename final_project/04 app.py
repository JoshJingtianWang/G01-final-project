# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

from datetime import datetime

now = datetime.now()
print("Current timestamp:", now)

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

# DBTITLE 0,YOUR APPLICATIONS CODE HERE...
#start_date = str(dbutils.widgets.get('01.start_date'))
#end_date = str(dbutils.widgets.get('02.end_date'))
#hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
#promote_model = bool(True if str(dbutils.widgets.get('04.promote_model')).lower() == 'yes' else False)

#print(start_date,end_date,hours_to_forecast, promote_model)

from datetime import datetime

import holidays
import pyspark.sql.functions as F
from pyspark.sql.functions import col, expr, from_unixtime
from pyspark.sql.functions import hour, minute, second, to_date
from pyspark.sql.types import *

status_data = (
    spark
    .read
    .format("delta")
    .load(BRONZE_STATION_STATUS_PATH)
)

#display(status_data)

info_data = (
    spark
    .read
    .format("delta")
    .load(BRONZE_STATION_INFO_PATH)
)

#display(info_data)

weather_df = (
    spark
    .read
    .format("delta")
    .load(SILVER_MODELING_HISTORICAL_PATH)
)


# COMMAND ----------

import pyspark.sql.functions as F

# Join the bronze tables to get weather and station data
bronze_df = (
    status_data
    .join(info_data, "station_id")
    .select("station_id", "timestamp", "temperature", "precipitation", "total_docks", "bikes_available")
)

# Aggregate the silver table to get the historical data
historical_data_df = (
    weather_df
    #.groupBy("station_id", F.window("timestamp", "4 hours"))
    .agg(F.avg("temperature").alias("avg_temperature"), F.sum("precipitation").alias("total_precipitation"), F.avg("bikes_available").alias("avg_bikes_available"))
    .select("station_id", "window.start", "window.end", "avg_temperature", "total_precipitation", "avg_bikes_available")
    .withColumnRenamed("window.start", "timestamp")
)

# Join the bronze and historical data to create the gold table
gold_df = (
    bronze_df
    .crossjoin(historical_data_df)
    .withColumnRenamed("avg_bikes_available", "actual_availability")
)

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

import pandas as pd

# Create a datetime index for the next 4 hours
start_time = now
end_time = start_time + pd.Timedelta(hours=4)
index = pd.date_range(start=start_time, end=end_time, freq="30T")

# Make predictions using the staging model
staging_predictions = []
for t in index:
    # Get the historical data for this time step
    historical_data = gold_df.loc[:t, :]

    # Make a prediction using the staging model
    input_data = prepare_input_data(historical_data)
    prediction = staging_model.predict(input_data)
    staging_predictions.append(prediction)

# Convert the predictions to a pandas DataFrame
staging_predictions_df = pd.DataFrame(staging_predictions, index=index, columns=["predicted_availability"])

# Check for stock-out or full station conditions
stock_out = staging_predictions_df["predicted_availability"] == 0
full_station = staging_predictions_df["predicted_availability"] == total_docks

# Combine the predictions with the stock-out and full station flags
predictions_with_flags = staging_predictions_df.join(stock_out).join(full_station)

# Define a function to return the predictions with flags
def predict_bike_availability():
    return predictions_with_flags



# COMMAND ----------

import matplotlib.pyplot as plt

# Get the residuals for the staging model
staging_residuals = staging_predictions_df - gold_df["actual_availability"].values.reshape(-1, 1)

# Plot the residuals
fig, ax = plt.subplots()
ax.plot(staging_residuals)
ax.axhline(y=0, color="gray", linestyle="--")
ax.set_xlabel("Time")
ax.set_ylabel("Residual")
ax.set_title("Staging Model Residual Plot")
plt.show()

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
