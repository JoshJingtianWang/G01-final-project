# Databricks notebook source
# MAGIC %pip install folium

# COMMAND ----------

# MAGIC %run ./includes/globals

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import time

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The following shows the current timestamp when the notebook is run (now).

# COMMAND ----------

from datetime import datetime
import pytz

now = datetime.now(pytz.timezone('America/New_York'))
print("Current timestamp:", now)

# COMMAND ----------

# MAGIC %md The following shows the current Production Model version and Staging Model version.

# COMMAND ----------

# Get the client object for the MLflow tracking server
client = mlflow.tracking.MlflowClient()

# List all registered models
registered_models = client.search_registered_models()

model_name = 'G01_model_temp'

# Get all model versions for the current registered model
model_versions = client.search_model_versions(f"name='{model_name}'")

model_info = [(model.version, model.current_stage) for model in model_versions]
staging = [model for model in model_info if model[1] == "Staging"]

production = [model for model in model_info if model[1] == "Production"]

most_recent_staging = max(staging, key=lambda x: x[0])

most_recent_production = max(production, key=lambda x: x[0])

print(f"Most recent production model version: {most_recent_production[0]}")
print(f"Most recent staging model version: {most_recent_staging[0]}")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC The following map shows the location of our station, with a marker. When the marker is clicked, the station name pops up: W 21 St & 6 Ave.

# COMMAND ----------

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

info_data = (
    spark
    .read
    .format("delta")
    .load(BRONZE_STATION_INFO_PATH)
)

weather_df = (
    spark
    .read
    .format("delta")
    .load(BRONZE_NYC_WEATHER_PATH)
)

# COMMAND ----------

import folium

# Extract latitude and longitude for our station
location_values = info_data.filter(col("name") == GROUP_STATION_ASSIGNMENT)[["lat", "lon"]].collect()[0]
latitude = location_values["lat"]
longitude = location_values["lon"]

# Create the map object centered on the station location
station_map = folium.Map(location=[latitude, longitude], zoom_start=15)

# Add a marker to the station location
folium.Marker([latitude, longitude], popup=GROUP_STATION_ASSIGNMENT).add_to(station_map)

# Display the map
display(station_map)

# COMMAND ----------

current_weather = (
    weather_df
    .withColumn("description", col("weather").getItem("description"))
    .orderBy(col("time"), ascending=False)
    .limit(1)
    .select("time", "description", "`rain.1h`")
    .collect()
)[0]

print(
    f"The most recent weather record as of {current_weather['time']} is:\n"
    f"\tConditions: {current_weather['description'][0]}\n"
    f"\tRain: {current_weather['rain.1h']} {'' if not current_weather['rain.1h'] else ' mm/hr'}"
)

# COMMAND ----------

display(info_data)

# COMMAND ----------

station_id = (
    info_data
    .filter(col("name") == GROUP_STATION_ASSIGNMENT)
    .collect()
)[0]["station_id"]

# COMMAND ----------

dock_info = (
    status_data
    .filter(col("station_id") == station_id)
    .orderBy(col("last_reported"), ascending=False)
    .limit(1)
    .withColumn("total_docks", col("num_docks_available") + col("num_docks_disabled"))
    .select("num_docks_available", "num_docks_disabled", "total_docks")
    .collect()
)[0]

print(
    f"With {dock_info['num_docks_available']} docks available and {dock_info['num_docks_disabled']} "
    f"docks disabled, there are {dock_info['total_docks']} docks in total."
)

# COMMAND ----------

bike_info = (
    status_data
    .filter(col("station_id") == station_id)
    .orderBy(col("last_reported"), ascending=False)
    .limit(1)
    # .withColumn("total_docks", col("num_docks_available") + col("num_docks_disabled"))
    .select("num_bikes_available", "num_ebikes_available", "num_bikes_disabled")
    .collect()
)[0]

print(
    f"As of the most recent update:\n"
    f"\tRegular bikes available: {bike_info['num_bikes_available']}\n"
    f"\tElectric bikes available: {bike_info['num_ebikes_available']}\n"
    f"\tDisabled bikes: {bike_info['num_bikes_disabled']}"
)

# COMMAND ----------

# MAGIC %md insert

# COMMAND ----------

# import pyspark.sql.functions as F

# # Join the bronze tables to get weather and station data
# bronze_df = (
#     status_data
#     .join(info_data, "station_id")
#     .select("station_id", "timestamp", "temperature", "precipitation", "total_docks", "bikes_available")
# )

# # Aggregate the silver table to get the historical data
# historical_data_df = (
#     weather_df
#     #.groupBy("station_id", F.window("timestamp", "4 hours"))
#     .agg(F.avg("temperature").alias("avg_temperature"), F.sum("precipitation").alias("total_precipitation"), F.avg("bikes_available").alias("avg_bikes_available"))
#     .select("station_id", "window.start", "window.end", "avg_temperature", "total_precipitation", "avg_bikes_available")
#     .withColumnRenamed("window.start", "timestamp")
# )

# # Join the bronze and historical data to create the gold table
# gold_df = (
#     bronze_df
#     .crossjoin(historical_data_df)
#     .withColumnRenamed("avg_bikes_available", "actual_availability")
# )

# COMMAND ----------

silver_weather_df = (
    weather_df
    # Create new "temp" columns for each temperature unit
    .withColumnRenamed("temp", "tempK")
    .withColumn("tempC", col("tempK") - 273.15)
    .withColumn("tempF", (9/5) * col("tempC") + 32)
    # Create new "feels_like" columns for each temperature unit
    .withColumnRenamed("feels_like", "feels_likeK")
    .withColumn("feels_likeC", col("feels_likeK") - 273.15)
    .withColumn("feels_likeF", (9/5) * col("feels_likeC") + 32)
    # Convert "wind_speed" from m/s to mph
    .withColumnRenamed("wind_speed", "wind_speed_mps")
    .withColumn("wind_speed_mph", col("wind_speed_mps") * 2.23694)
    # Convert "dt" from unix time to a timestamp and extract components
    .withColumn("dt", from_unixtime("dt").cast("timestamp"))
    .withColumn("dt_date", to_date(col("dt")))
    .withColumn("dt_month", F.month(col("dt_date")))
    .withColumn("dt_hour", hour(col("dt")))
    .withColumn("dt_minute", minute(col("dt")))
    .withColumn("dt_second", second(col("dt")))
    # Rename the data evolution "_rescued_data" column
    .withColumnRenamed("_rescued_data", "weather._rescued_data")
)

# COMMAND ----------

aggregation_df = (
    silver_weather_df
    # Extract only the columns relevant to modeling
    .select(
        col("dt_date"),
        col("dt_hour"),
        col("tempF"),
        col("feels_likeF"),
        col("`rain.1h`"),
    )
    # Group by date and hour
    .groupBy(
        col("dt_date"),
        col("dt_hour"),
    )
    # Perform aggregations
    .agg(
        F.avg("tempF").alias("temperature"),
        F.avg("feels_likeF").alias("feels_like"),
        F.avg("`rain.1h`").alias("precipitation"),
    )
    # Combine the date and hour columns into datetime format
    .withColumn("ds", F.concat_ws(" ", "dt_date", "dt_hour"))
    .withColumn("ds", F.to_timestamp("ds", "yyyy-MM-dd H"))
    # Fill any null values with 0
    .na.fill(0, ["precipitation"])
    # Drop any unnecessary columns (done this way to keep pipeline dynamic)
    .drop(
        "dt_date",
        "dt_hour",
    )
    # Order by datetime
    .orderBy(col("ds"), ascending=True)
)

display(aggregation_df)

# COMMAND ----------

GROUP_NAME = 'G01'
# ARTIFACT_PATH = f"{GROUP_NAME}_model"
ARTIFACT_PATH = f"{GROUP_NAME}_model_temp"

model_production_uri = "models:/{model_name}/production".format(model_name=ARTIFACT_PATH)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_production_uri))

model_production = mlflow.prophet.load_model(model_production_uri)

model_staging_uri = "models:/{model_name}/staging".format(model_name=ARTIFACT_PATH)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_staging_uri))

model_staging = mlflow.prophet.load_model(model_staging_uri)

# COMMAND ----------

future = model_staging.make_future_dataframe(periods=4, freq="H")
future["ds"] = future["ds"] + pd.DateOffset(hours=5)
future

# COMMAND ----------

silver_historical_modeling_df = (
    spark
    .read
    .format("delta")
    .load(SILVER_MODELING_HISTORICAL_PATH)
)

# COMMAND ----------

min_ds = future["ds"].min()
max_ds = future["ds"].max()

# COMMAND ----------

test_df = (
    silver_historical_modeling_df
    .filter((min_ds <= col("ds")) & (col("ds") <= max_ds))
    .drop("y")
)

display(test_df)

# COMMAND ----------

forecast = model_staging.predict(test_df.toPandas())

# COMMAND ----------

forecast[forecast["ds"] >= (max_ds - pd.DateOffset(hours=5+4))]["yhat"]

# COMMAND ----------

# import pandas as pd

# # Create a datetime index for the next 4 hours
# start_time = datetime.now()
# end_time = start_time + pd.Timedelta(hours=4)
# index = pd.date_range(start=start_time, end=end_time, freq="30T")

# # Make predictions using the staging model
# staging_predictions = []
# for t in index:
#     # Get the historical data for this time step
#     historical_data = gold_df.loc[:t, :]

#     # Make a prediction using the staging model
#     input_data = prepare_input_data(historical_data)
#     prediction = staging_model.predict(input_data)
#     staging_predictions.append(prediction)

# # Convert the predictions to a pandas DataFrame
# staging_predictions_df = pd.DataFrame(staging_predictions, index=index, columns=["predicted_availability"])

# # Check for stock-out or full station conditions
# stock_out = staging_predictions_df["predicted_availability"] == 0
# full_station = staging_predictions_df["predicted_availability"] == total_docks

# # Combine the predictions with the stock-out and full station flags
# predictions_with_flags = staging_predictions_df.join(stock_out).join(full_station)

# # Define a function to return the predictions with flags
# def predict_bike_availability():
#     return predictions_with_flags



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
