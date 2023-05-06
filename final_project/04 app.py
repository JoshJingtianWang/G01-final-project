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

# start_date = str(dbutils.widgets.get('01.start_date'))
# end_date = str(dbutils.widgets.get('02.end_date'))
# hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
# promote_model = bool(True if str(dbutils.widgets.get('04.promote_model')).lower() == 'yes' else False)

# print(start_date,end_date,hours_to_forecast, promote_model)

PROMOTE_MODEL = False

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we can import the necessary modules and define useful constants.

# COMMAND ----------

import time
from datetime import datetime, timedelta

import holidays
import folium
import mlflow
import plotly.express as px
import pyspark.sql.functions as F
import pytz
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from pyspark.sql.functions import col, expr, from_unixtime
from pyspark.sql.functions import hour, minute, second, to_date
from pyspark.sql.types import *

# COMMAND ----------

ARTIFACT_PATH = f"{GROUP_NAME}_model"
# ARTIFACT_PATH = f"{GROUP_NAME}_model_temp"
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

# COMMAND ----------

# Read the station info bronze table
info_data = (
    spark
    .read
    .format("delta")
    .load(BRONZE_STATION_INFO_PATH)
)

# COMMAND ----------

# Read the NYC weather bronze table
weather_df = (
    spark
    .read
    .format("delta")
    .load(BRONZE_NYC_WEATHER_PATH)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load the silver modeling table

# COMMAND ----------

# Load the silver modeling historical table
silver_historical_modeling_df = (
    spark
    .read
    .format("delta")
    .load(SILVER_MODELING_HISTORICAL_PATH)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create gold tables

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

# Use our station's name to determine its ID
station_id = info_data.filter(col("name") == GROUP_STATION_ASSIGNMENT).collect()[0]["station_id"]

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
        "station_id", "name", "latitude", "longitude", "capacity", "num_docks_available",
        "num_docks_disabled", "total_docks", "num_bikes_available", "num_ebikes_available",
        "num_bikes_disabled",
    )
    # Join with the current weather table
    .join(
        current_weather_df, "name"
    )
)

# COMMAND ----------

gold_forecast_df = (
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

# COMMAND ----------

gold_availability_df = (
    status_data
    .filter(col("station_id") == station_id)
    .withColumn("dt", from_unixtime("last_reported"))
    .select(
        "dt",
        "num_bikes_available",
        "num_ebikes_available",
    )
    .withColumn("bikes_available", col("num_bikes_available") + col("num_ebikes_available"))
)

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

# Get the latest versions of the production and staging models
latest_versions = client.get_latest_versions(MODEL_NAME, stages=["Production", "Staging"])

# Create HTML for each row in the table
info_html = [f"<tr><td>{v.current_stage}</td><td>{v.version}</td></tr>" for v in latest_versions]

# Display the HTML table
displayHTML(f"""
    <table border=1>
    <tr><td><b>Model</b></td><td><b>Version</b></td></tr>
    {''.join(info_html)}
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

# Create HTML for each row in the table
html_rows = []
for key in weather_info:
    name = weather_info[key][0]
    try:
        # Round to two decimal places if possible
        value = f"{float(current_weather[name]):0.2f}"
    except TypeError:
        # If a value is None, replace with 0
        value = "0"
    except ValueError:
        # If the value is a string, it is the weather conditions
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
# MAGIC # Total docks at station

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we can determine the total number of docks that are at our station, taking into account the number of docks available and the number of docks disabled.

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

# MAGIC %md
# MAGIC ## Preparing the data

# COMMAND ----------

# MAGIC %md
# MAGIC In this section, we will prepare the data for forecasting, and then prepare the forecast for plotting. A visualization of the forecast will be shown in the next section.

# COMMAND ----------

# Load the production model
model_production_uri = "models:/{model_name}/production".format(model_name=ARTIFACT_PATH)
print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_production_uri))
model_production = mlflow.prophet.load_model(model_production_uri)
print("Successfully loaded the production model.\n")

# Try to load the staging model
try:
    model_staging_uri = "models:/{model_name}/staging".format(model_name=ARTIFACT_PATH)
    print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_staging_uri))
    model_staging = mlflow.prophet.load_model(model_staging_uri)
except mlflow.exceptions.MlflowException:
    # Staging model might not have been created yet
    print("No staging model was found.")
    model_staging = None

# COMMAND ----------

# Get the earliest time in the historical data
start_time = silver_historical_modeling_df.orderBy("ds", ascending=True).limit(1).collect()[0]["ds"]

# Get a datetime object of 4 hours in the future from the current time
end_time = datetime.now() + timedelta(hours=4)

# Create a date range by hour from the start of the data to 4 hours from now
prediction_ds = pd.date_range(start=start_time, end=end_time, freq='H')
# Correct for timezones
prediction_ds = prediction_ds - pd.DateOffset(hours=5)

# COMMAND ----------

# Get the earliest timestamp
min_ds = prediction_ds.min()
# Get the latest timestamp
max_ds = prediction_ds.max()

# COMMAND ----------

# Columns of the dataframes must be in the same order to perform a union
column_order = ["ds", "temperature", "feels_like", "precipitation"]

# Get the appropriate data from the historical table
historical_test_df = (
    silver_historical_modeling_df
    .filter((min_ds <= col("ds")) & (col("ds") <= max_ds))
    .drop("y")
    .select(column_order)
)

# Get the appropriate data from the unseen table
unseen_test_df = (
    gold_forecast_df
    .filter((min_ds <= col("ds")) & (col("ds") <= max_ds))
    .select(column_order)
)

# COMMAND ----------

# The union of the two dataframes will be used for predictions
prediction_df = historical_test_df.union(unseen_test_df).orderBy("ds", ascending=False)

# COMMAND ----------

# Use the production model to make predictions
forecast_df = model_production.predict(prediction_df.toPandas())

# Correct for timezones
forecast_df["ds"] = forecast_df["ds"] + pd.DateOffset(hours=5)

# COMMAND ----------

# Determine the number of bikes available at the station before forecasting
bikes_available = bike_info["num_bikes_available"] + bike_info["num_ebikes_available"]

# Get the last four hours worth of predictions
net_changes = forecast_df.tail(4)["yhat"].tolist()

# Use the net change forecast to determine how many bikes will be at the station
bikes_forecast = []
for net_change in net_changes:
    previous_availability = bikes_available if not bikes_forecast else bikes_forecast[-1]
    # Add the previous availability to the net change (rounded towards zero)
    bikes_forecast.append(previous_availability + int(net_change))

# COMMAND ----------

# Pull the station's capacity from the gold table
capacity = gold_df.collect()[0]["capacity"]
print(f"Our station has a capacity of {capacity} bikes.")

# COMMAND ----------

# Look back four hours before the current time
last_four_hours = forecast_df.iloc[-8:-4]["ds"]

# Get the inventory at various points over those four hours
historical_inventory_df = (
    gold_availability_df
    .filter((last_four_hours.min() <= col("dt")) & (col("dt") <= last_four_hours.max()))
    .withColumnRenamed("dt", "time")
    .withColumnRenamed("bikes_available", "inventory")
    .select("time", "inventory")
    .orderBy("time", ascending=True)
)

# COMMAND ----------

# Create a dataframe to store the forecasted data
forecasted_inventory_df = pd.DataFrame({
    "time": forecast_df.tail(4)["ds"].tolist(),
    "inventory": bikes_forecast,
})

# COMMAND ----------

# Concatenate the two dataframes to prepare for plotting
production_forecast_df = pd.concat([
    historical_inventory_df.toPandas().rename({"dt": "time", "bikes_available": "inventory"}),
    forecasted_inventory_df
], axis=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualizing the forecast

# COMMAND ----------

fig = px.line(production_forecast_df, x="time", y="inventory", title="Forecast of Bike Inventory")
fig.add_hline(y=0, line_width=3, line_dash="dash", line_color="gray")
fig.add_hline(y=capacity, line_width=3, line_dash="dash", line_color="gray")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Residual plot

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we can plot the residuals for both the production and staging forecasts. Only the historical time period is plotted due to lack of data, but this should be enough to provide a general idea of how well the models are performing.
# MAGIC
# MAGIC If there is no model currently staged, the visualization will only include residuals for the production model.

# COMMAND ----------

# Get the actual net change values from the historical modeling table
actual_df = silver_historical_modeling_df.toPandas()[["ds", "y"]].reset_index(drop=True)

# Instantiate a list to store the resulting dataframes for production and staging
result_dfs = []

# Get the forecasts for the historical time period
production_forecast = forecast_df[(actual_df["ds"].min() <= forecast_df["ds"]) & (forecast_df["ds"] <= actual_df["ds"].max())]

# Calculate the residuals for the production forecast
results_production = pd.DataFrame({
    "yhat": production_forecast["yhat"],
    "residual": production_forecast["yhat"] - actual_df["y"],
    "label": "Production",
})
result_dfs.append(results_production)

# Only plot staging residuals if there is a staging model
if model_staging:
    # Use the production model to make predictions
    staging_forecast_df = model_staging.predict(prediction_df.toPandas())

    # Correct for timezones
    staging_forecast_df["ds"] = staging_forecast_df["ds"] + pd.DateOffset(hours=5)

    # Get the forecasts for the historical time period
    staging_forecast = staging_forecast_df[(actual_df["ds"].min() <= staging_forecast_df["ds"]) & (staging_forecast_df["ds"] <= actual_df["ds"].max())]

    # Calculate the residuals for the staging forecast
    results_staging = pd.DataFrame({
        "yhat": staging_forecast["yhat"],
        "residual": staging_forecast["yhat"] - actual_df["y"],
        "label": "Staging",
    })
    result_dfs.append(results_staging)
else:
    print("The residual plot will only contain production data because there is currently no staging model.")

# Concatenate the results in preparation for plotting
results = pd.concat(result_dfs, axis=0, ignore_index=True)

# COMMAND ----------

# Plot the residuals
fig = px.scatter(
    results,
    x="yhat",
    y="residual",
    marginal_y="violin",
    trendline="ols",
    color="label",
    opacity=0.5 
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Archive and promote models

# COMMAND ----------

# MAGIC %md
# MAGIC If desired by the user, archive the current production model and promote the staging model. This is controlled by a widget in the top-level notebook.

# COMMAND ----------

# Promote the staging model if desired
if PROMOTE_MODEL and model_staging:

    # Get the current production model
    latest_production = [model for model in latest_versions if model.current_stage=="Production"][0]

    # Archive the current production model
    client.transition_model_version_stage(
        name=latest_production.name,
        version=latest_production.version,
        stage="Archived",
    )

    # Get the current staging model
    latest_staging = [model for model in latest_versions if model.current_stage=="Staging"][0]

    # Promote the staging model to production
    client.transition_model_version_stage(
        name=latest_staging.name,
        version=latest_staging.version,
        stage="Production",
    )

    print("The staging model has been successfully promoted to production.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Clean up
# MAGIC
# MAGIC Finally, we can perform operations that will clean up and exit the notebook.

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
