# Databricks notebook source
# MAGIC %md
# MAGIC # Application

# COMMAND ----------

# MAGIC %md
# MAGIC # Initial setup

# COMMAND ----------

# MAGIC %md
# MAGIC We need to run `pip install` before importing the globals since it restarts the Python interpreter.

# COMMAND ----------

# MAGIC %pip install folium

# COMMAND ----------

# MAGIC %run ./includes/globals

# COMMAND ----------

start_date = str(dbutils.widgets.get('01.start_date'))
end_date = str(dbutils.widgets.get('02.end_date'))
hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
promote_model = bool(True if str(dbutils.widgets.get('04.promote_model')).lower() == 'yes' else False)

print(start_date,end_date,hours_to_forecast, promote_model)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we can import the necessary modules and define useful constants.

# COMMAND ----------

import time
from datetime import datetime

import holidays
import folium
import mlflow
import pyspark.sql.functions as F
import pytz
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from pyspark.sql.functions import col, expr, from_unixtime
from pyspark.sql.functions import hour, minute, second, to_date
from pyspark.sql.types import *

# COMMAND ----------

# ARTIFACT_PATH = f"{GROUP_NAME}_model"
ARTIFACT_PATH = f"{GROUP_NAME}_model_temp"
MODEL_NAME = ARTIFACT_PATH

# COMMAND ----------

# MAGIC %md
# MAGIC # Load existing bronze tables

# COMMAND ----------

# Read the station status bronze table
status_data = (
    spark
    .read
    .format("delta")
    .load(BRONZE_STATION_STATUS_PATH)
)

# Read the station info bronze table
info_data = (
    spark
    .read
    .format("delta")
    .load(BRONZE_STATION_INFO_PATH)
)

# Read the NYC weather bronze table
weather_df = (
    spark
    .read
    .format("delta")
    .load(BRONZE_NYC_WEATHER_PATH)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create gold table

# COMMAND ----------

current_weather_df = (
    weather_df
    # Limit to the most recent data
    .orderBy(col("time"), ascending=False)
    .limit(1)
    # Add the name of the station for joining purposes
    .withColumn("name", F.lit(GROUP_STATION_ASSIGNMENT))
    # Convert "temperature" column to Fahrenheit
    .withColumn("temperature", (9/5) * (col("temp") - 273.15) + 32)
    # Convert "feels like" column to Fahrenheit
    .withColumn("feels_like", (9/5) * (col("feels_like") - 273.15) + 32)
    # Convert "wind_speed" from m/s to mph
    .withColumn("wind_speed", col("wind_speed") * 2.23694)
    # Convert precipitation from mm/hr to in/hr
    .withColumn("rain.1h", col("`rain.1h`") * 0.0393700787)
    # Extract weather condition data
    .withColumn("weather", col("weather").getItem(0))
    .withColumn("weather_main", col("weather").getItem("main"))
    .withColumn("weather_description", col("weather").getItem("description"))
    # Rename columns
    .withColumnRenamed("rain.1h", "precipitation")
    # Select only the relevant columns
    .select(
        "name", "time", "weather_main", "weather_description", "temperature",
        "feels_like", "precipitation", "wind_speed",
    )
)

# COMMAND ----------

gold_df = (
    # Start with the station status bronze table
    status_data
    # Left join it with the station info bronze table
    .join(
        info_data, "station_id", "left"
    )
    # Filter out other stations
    .filter(col("station_id") == station_id)
    # Limit to only the most recent data
    .orderBy(col("last_reported"), ascending=False)
    .limit(1)
    # Calculate the total number of docks
    .withColumn("total_docks", col("num_docks_available") + col("num_docks_disabled"))
    # Rename columns
    .withColumnRenamed("lat", "latitude")
    .withColumnRenamed("lon", "longitude")
    # Extract only the appropriate columns
    .select(
        "station_id", "name", "latitude", "longitude", "capacity", "num_docks_available", "num_docks_disabled", "total_docks", "num_bikes_available", "num_ebikes_available", "num_bikes_disabled",
    )
    # Join with the current weather table
    .join(
        current_weather_df, "name"
    )
)

display(gold_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Current timestamp

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The following shows the current timestamp when the notebook is run (i.e., the time right now).

# COMMAND ----------

now = datetime.now(pytz.timezone('America/New_York'))
print("Current time:", now)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Production and staging models

# COMMAND ----------

# MAGIC %md The following shows the current production model version and staging model version.

# COMMAND ----------

# Get the client object for the MLflow tracking server
client = mlflow.tracking.MlflowClient()

# List all registered models
registered_models = client.search_registered_models()

# Get all model versions for the current registered model
model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

# Get a list of all models, their stages, and their versions
model_info = [(model.version, model.current_stage) for model in model_versions]

# COMMAND ----------

# Get information about the most recent staging and production model
html_rows = []
for stage in ["Production", "Staging"]:
    stage_models = [model for model in model_info if model[1] == stage]
    most_recent_model = max(stage_models, key=lambda x: x[0])
    html_rows.append(f"<tr><td>{stage}</td><td>{most_recent_model[0]}</td></tr>")

# Display the HTML table
displayHTML(f"""
    <table border=1>
    <tr><td><b>Model</b></td><td><b>Version</b></td></tr>
    {''.join(html_rows)}
    </table>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Station location

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC The following map shows the location of our station, with a marker. When the marker is clicked, the station name pops up: W 21 St & 6 Ave.

# COMMAND ----------

# Extract latitude and longitude for our station
location_values = gold_df.select("latitude", "longitude").collect()[0]
latitude = location_values["latitude"]
longitude = location_values["longitude"]

# Create the map object centered on the station location
station_map = folium.Map(location=[latitude, longitude], zoom_start=15)

# Add a marker to the station location
folium.Marker([latitude, longitude], popup=f"<h4>{GROUP_STATION_ASSIGNMENT}</h4>").add_to(station_map)

# Display the map
display(station_map)

# COMMAND ----------

# MAGIC %md
# MAGIC # Current weather

# COMMAND ----------

# Extract current weather data
current_weather = gold_df.select(
    "weather_description",
    "precipitation",
    "temperature",
    "feels_like",
    "wind_speed",
).collect()[0]

# COMMAND ----------

# Prepare to display the current weather
weather_info = {
    "Conditions": ("weather_description", ""),
    "Temperature": ("temperature", "F"),
    "Feels like": ("feels_like", "F"),
    "Wind speed": ("wind_speed", "mph"),
    "Precipitation": ("precipitation", "in/hr"),
}

html_rows = []
for key in weather_info:
    name = weather_info[key][0]
    try:
        value = f"{float(current_weather[name]):0.2f}"
    except ValueError:
        value = current_weather[name]
    units = weather_info[key][1]
    html_rows.append(f"<tr><td>{key}</td><td>{value}</td><td>{units}</td></tr>")

# Display the HTML table
displayHTML(f"""
    <table border=1>
    <tr><td><b>Description</b></td><td><b>Value</b></td><td><b>Units</b></td></tr>
    {''.join(html_rows)}
    </table>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Dock availability

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we can determine the number of docks that are available at our station.

# COMMAND ----------

# Extract current weather data
dock_info = gold_df.select(
    "num_docks_available",
    "num_docks_disabled",
    "total_docks",
).collect()[0]

# COMMAND ----------

# Display the HTML table
displayHTML(f"""
    <table border=1>
    <tr><td><b>Description</b></td><td><b>Value</b></td></tr>
    <tr><td>Number of docks available</td><td>{dock_info['num_docks_available']}</td></tr>
    <tr><td>Number of docks disabled</td><td>{dock_info['num_docks_disabled']}</td></tr>
    <tr><td><i>Total number of docks<i></td><td><i>{dock_info['total_docks']}<i></td></tr>
    </table>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Bike availability

# COMMAND ----------

# MAGIC %md
# MAGIC The following shows the availability of bikes at our station, assuming that the number of bikes available is distinctly independent from the number of e-bikes available.

# COMMAND ----------

# Extract current weather data
bike_info = gold_df.select(
    "num_bikes_available",
    "num_ebikes_available",
    "num_bikes_disabled",
).collect()[0]

# COMMAND ----------

# Display the HTML table
displayHTML(f"""
    <table border=1>
    <tr><td><b>Bike Type</b></td><td><b>Available</b></td></tr>
    <tr><td>Regular</td><td>{bike_info['num_bikes_available']}</td></tr>
    <tr><td>Electric</td><td>{bike_info['num_ebikes_available']}</td></tr>
    <tr><td>Disabled</td><td>{bike_info['num_bikes_disabled']}</td></tr>
    </table>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Forecasting net bike change

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

# import matplotlib.pyplot as plt

# # Get the residuals for the staging model
# staging_residuals = staging_predictions_df - gold_df["actual_availability"].values.reshape(-1, 1)

# # Plot the residuals
# fig, ax = plt.subplots()
# ax.plot(staging_residuals)
# ax.axhline(y=0, color="gray", linestyle="--")
# ax.set_xlabel("Time")
# ax.set_ylabel("Residual")
# ax.set_title("Staging Model Residual Plot")
# plt.show()

# COMMAND ----------

display(status_data)

# COMMAND ----------

display(
    status_data
    .withColumn("last_reported", from_unixtime("last_reported").cast("timestamp"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Clean up
# MAGIC
# MAGIC Finally, we can perform operations that will clean up and exit the notebook.

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
