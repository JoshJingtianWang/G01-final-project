# Databricks notebook source
# MAGIC %md
# MAGIC # Extract, Transform, Load

# COMMAND ----------

# MAGIC %md
# MAGIC Initialize the database and various global variables.

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
# MAGIC # Initial setup
# MAGIC 
# MAGIC First, we will import necessary modules, change configuration settings, and define useful constants.

# COMMAND ----------

from datetime import datetime

import holidays
import pyspark.sql.functions as F
from pyspark.sql.functions import col, expr, from_unixtime
from pyspark.sql.functions import hour, minute, second, to_date
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md
# MAGIC # Load existing bronze tables

# COMMAND ----------

# MAGIC %md
# MAGIC First, we can easily load the premade Delta tables into their own dataframes for processing and eventually storing in silver tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Station info table

# COMMAND ----------

# MAGIC %md
# MAGIC The following is the schema for the station info bronze table:
# MAGIC - `has_kiosk`: boolean (nullable = true)
# MAGIC - `station_type`: string (nullable = true)
# MAGIC - `region_id`: string (nullable = true)
# MAGIC - `short_name`: string (nullable = true)
# MAGIC - `lat`: double (nullable = true)
# MAGIC - `electric_bike_surcharge_waiver`: boolean (nullable = true)
# MAGIC - `capacity`: long (nullable = true)
# MAGIC - `legacy_id`: string (nullable = true)
# MAGIC - `station_id`: string (nullable = true)
# MAGIC - `eightd_has_key_dispenser`: boolean (nullable = true)
# MAGIC - `external_id`: string (nullable = true)
# MAGIC - `rental_methods`: array (nullable = true)
# MAGIC   - `element`: string (containsNull = true)
# MAGIC - `lon`: double (nullable = true)
# MAGIC - `name`: string (nullable = true)
# MAGIC - `rental_uris.ios`: string (nullable = true)
# MAGIC - `rental_uris.android`: string (nullable = true)
# MAGIC 
# MAGIC The path to the table is stored in the following global variable: `BRONZE_STATION_INFO_PATH`.

# COMMAND ----------

bronze_station_info_df = (
    spark
    .read
    .format("delta")
    .load(BRONZE_STATION_INFO_PATH)
)

display(bronze_station_info_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Station status table

# COMMAND ----------

# MAGIC %md
# MAGIC The following is the schema for the station status bronze table:
# MAGIC - `num_ebikes_available`: long (nullable = true)
# MAGIC - `is_installed`: long (nullable = true)
# MAGIC - `num_docks_available`: long (nullable = true)
# MAGIC - `num_scooters_unavailable`: double (nullable = true)
# MAGIC - `num_scooters_available`: double (nullable = true)
# MAGIC - `station_id`: string (nullable = true)
# MAGIC - `last_reported`: long (nullable = true)
# MAGIC - `num_docks_disabled`: long (nullable = true)
# MAGIC - `is_renting`: long (nullable = true)
# MAGIC - `is_returning`: long (nullable = true)
# MAGIC - `eightd_has_available_keys`: boolean (nullable = true)
# MAGIC - `station_status`: string (nullable = true)
# MAGIC - `num_bikes_disabled`: long (nullable = true)
# MAGIC - `legacy_id`: string (nullable = true)
# MAGIC - `num_bikes_available`: long (nullable = true)
# MAGIC - `valet.region`: string (nullable = true)
# MAGIC - `valet.off_dock_capacity`: double (nullable = true)
# MAGIC - `valet.active`: boolean (nullable = true)
# MAGIC - `valet.dock_blocked_count`: double (nullable = true)
# MAGIC - `valet.off_dock_count`: double (nullable = true)
# MAGIC - `valet.station_id`: string (nullable = true)
# MAGIC - `valet.valet_revision`: double (nullable = true)
# MAGIC 
# MAGIC The path to the table is stored in the following global variable: `BRONZE_STATION_STATUS_PATH`.

# COMMAND ----------

bronze_station_status_df = (
    spark
    .read
    .format("delta")
    .load(BRONZE_STATION_STATUS_PATH)
)

display(bronze_station_status_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## NYC weather table

# COMMAND ----------

# MAGIC %md
# MAGIC The following is the schema for the NYC weather data bronze table:
# MAGIC - `dt`: long (nullable = true)
# MAGIC - `temp`: double (nullable = true)
# MAGIC - `feels_like`: double (nullable = true)
# MAGIC - `pressure`: long (nullable = true)
# MAGIC - `humidity`: long (nullable = true)
# MAGIC - `dew_point`: double (nullable = true)
# MAGIC - `uvi`: double (nullable = true)
# MAGIC - `clouds`: long (nullable = true)
# MAGIC - `visibility`: long (nullable = true)
# MAGIC - `wind_speed`: double (nullable = true)
# MAGIC - `wind_deg`: long (nullable = true)
# MAGIC - `wind_gust`: double (nullable = true)
# MAGIC - `weather`: array (nullable = true)
# MAGIC   - `element`: struct (containsNull = true)
# MAGIC     - `description`: string (nullable = true)
# MAGIC     - `icon`: string (nullable = true)
# MAGIC     - `id`: long (nullable = true)
# MAGIC     - `main`: string (nullable = true)
# MAGIC - `pop`: double (nullable = true)
# MAGIC - `rain.1h`: double (nullable = true)
# MAGIC - `time`: string (nullable = true)
# MAGIC 
# MAGIC The path to the table is stored in the following global variable: `BRONZE_NYC_WEATHER_PATH`.

# COMMAND ----------

bronze_nyc_weather_df = (
    spark
    .read
    .format("delta")
    .load(BRONZE_NYC_WEATHER_PATH)
)

display(bronze_nyc_weather_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create new bronze tables

# COMMAND ----------

# MAGIC %md
# MAGIC ## Historical station table

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The following is the schema for the historical station data bronze table:
# MAGIC - `ride_id`: string (nullable = true)
# MAGIC - `rideable_type`: string (nullable = true)
# MAGIC - `started_at`: string (nullable = true)
# MAGIC - `ended_at`: string (nullable = true)
# MAGIC - `start_station_name`: string (nullable = true)
# MAGIC - `start_station_id`: string (nullable = true)
# MAGIC - `end_station_name`: string (nullable = true)
# MAGIC - `end_station_id`: string (nullable = true)
# MAGIC - `start_lat`: string (nullable = true)
# MAGIC - `start_lng`: string (nullable = true)
# MAGIC - `end_lat`: string (nullable = true)
# MAGIC - `end_lng`: string (nullable = true)
# MAGIC - `member_casual`: string (nullable = true)
# MAGIC - `_rescued_data`: string (nullable = true)
# MAGIC 
# MAGIC The path to the table is stored in the following global variable: `BIKE_TRIP_DATA_PATH`.
# MAGIC 
# MAGIC The directory its checkpoints are saved at are stored in the following global variable: `BRONZE_STATION_HISTORY_CHECKPOINTS`.

# COMMAND ----------

# MAGIC %md
# MAGIC We can begin by streaming the files in the appropriate directory one-by-one, while setting the `mergeSchema` option to *True* and supplying a checkpoint path for the `cloudFiles.schemaLocation` option.

# COMMAND ----------

bronze_station_history_df = (
    spark
    .readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    # .option("maxFilesPerTrigger", 1)
    .option("header", "true")
    .option("mergeSchema", "true")
    .option("cloudFiles.schemaLocation", BRONZE_STATION_HISTORY_CHECKPOINTS)
    .load(BIKE_TRIP_DATA_PATH)
)

display(bronze_station_history_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have started the stream, we can write the stream to a new bronze table.
# MAGIC 
# MAGIC We use the `.outputMode("append")` here to ensure that only new data is written to the table.

# COMMAND ----------

bronze_station_history_query = (
    bronze_station_history_df
    .writeStream
    .format("delta")
    .outputMode("append")
    .trigger(once=True)
    .option("checkpointLocation", BRONZE_STATION_HISTORY_CHECKPOINTS)
    .start(BRONZE_STATION_HISTORY_PATH)
)

# COMMAND ----------

display(spark.sql(f"OPTIMIZE delta.`{BRONZE_STATION_HISTORY_PATH}`"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Historical weather table

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The following is the schema for the historical weather data bronze table:
# MAGIC - `dt`: string (nullable = true)
# MAGIC - `temp`: string (nullable = true)
# MAGIC - `feels_like`: string (nullable = true)
# MAGIC - `pressure`: string (nullable = true)
# MAGIC - `humidity`: string (nullable = true)
# MAGIC - `dew_point`: string (nullable = true)
# MAGIC - `uvi`: string (nullable = true)
# MAGIC - `clouds`: string (nullable = true)
# MAGIC - `visibility`: string (nullable = true)
# MAGIC - `wind_speed`: string (nullable = true)
# MAGIC - `wind_deg`: string (nullable = true)
# MAGIC - `pop`: string (nullable = true)
# MAGIC - `snow_1h`: string (nullable = true)
# MAGIC - `id`: string (nullable = true)
# MAGIC - `main`: string (nullable = true)
# MAGIC - `description`: string (nullable = true)
# MAGIC - `icon`: string (nullable = true)
# MAGIC - `loc`: string (nullable = true)
# MAGIC - `lat`: string (nullable = true)
# MAGIC - `lon`: string (nullable = true)
# MAGIC - `timezone`: string (nullable = true)
# MAGIC - `timezone_offset`: string (nullable = true)
# MAGIC - `rain_1h`: string (nullable = true)
# MAGIC - `_rescued_data`: string (nullable = true)
# MAGIC 
# MAGIC The path to the table is stored in the following global variable: `NYC_WEATHER_FILE_PATH`.
# MAGIC 
# MAGIC The directory its checkpoints are saved at are stored in the following global variable: `BRONZE_WEATHER_HISTORY_CHECKPOINTS`.

# COMMAND ----------

# MAGIC %md
# MAGIC We can begin by streaming the files in the appropriate directory one-by-one, while setting the `mergeSchema` option to *True* and supplying a checkpoint path for the `cloudFiles.schemaLocation` option.

# COMMAND ----------

bronze_weather_history_df = (
    spark
    .readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    # .option("maxFilesPerTrigger", 1)
    .option("header", "true")
    .option("mergeSchema", "true")
    .option("cloudFiles.schemaLocation", BRONZE_WEATHER_HISTORY_CHECKPOINTS)
    .load(NYC_WEATHER_FILE_PATH)
)

display(bronze_station_history_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have started the stream, we can write the data to a new bronze table.
# MAGIC 
# MAGIC We use the `.mode("append")` here to ensure that only new data is written to the table.

# COMMAND ----------

bronze_weather_history_query = (
    bronze_weather_history_df
    .writeStream
    .format("delta")
    .outputMode("append")
    .trigger(once=True)
    .option("checkpointLocation", BRONZE_WEATHER_HISTORY_CHECKPOINTS)
    .start(BRONZE_WEATHER_HISTORY_PATH)
)

# COMMAND ----------

display(spark.sql(f"OPTIMIZE delta.`{BRONZE_WEATHER_HISTORY_PATH}`"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Create silver tables

# COMMAND ----------

# MAGIC %md
# MAGIC To create a silver table for each bronze table, we must read the raw data that is stored in the bronze table, perform any cleaning and transformations, then finally save the transformed data as new silver tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Station info table

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The following is the schema for the station info silver table:
# MAGIC - `has_kiosk`: boolean (nullable = true)
# MAGIC - `station_type`: string (nullable = true)
# MAGIC - `region_id`: string (nullable = true)
# MAGIC - `short_name`: string (nullable = true)
# MAGIC - `lat`: double (nullable = true)
# MAGIC - `electric_bike_surcharge_waiver`: boolean (nullable = true)
# MAGIC - `capacity`: long (nullable = true)
# MAGIC - `legacy_id`: string (nullable = true)
# MAGIC - `station_id`: string (nullable = true)
# MAGIC - `eightd_has_key_dispenser`: boolean (nullable = true)
# MAGIC - `external_id`: string (nullable = true)
# MAGIC - `rental_methods`: array (nullable = true)
# MAGIC   - `element`: string (containsNull = true)
# MAGIC - `lon`: double (nullable = true)
# MAGIC - `name`: string (nullable = true)
# MAGIC - `rental_uris`.ios: string (nullable = true)
# MAGIC - `rental_uris`.android: string (nullable = true)
# MAGIC - `uses_key`: boolean (nullable = true)
# MAGIC - `uses_method_credit_card`: boolean (nullable = true)
# MAGIC 
# MAGIC The path to the table is stored in the following global variable: `SILVER_STATION_INFO_PATH`.
# MAGIC 
# MAGIC The directory its checkpoints are saved at are stored in the following global variable: `SILVER_STATION_INFO_CHECKPOINTS`.

# COMMAND ----------

# MAGIC %md
# MAGIC There are other transformations that can be done on the station info silver table, but currently are not being done. This is because these transformations do not seem to serve a purpose for the rest of the team, and will essentially only slow down the project by adding unnecessary overhead and computations. Examples of these transformations are:
# MAGIC 
# MAGIC - There are null values in columns such as `region_id` and could benefit from data imputation or filtering
# MAGIC - The `rental_methods` columns is a struct and could be exploded into multiple rows

# COMMAND ----------

silver_station_info_df = (
    bronze_station_info_df
    # Create new columns for the rental methods
    .withColumn("uses_key", F.array_contains(col("rental_methods"), "KEY"))
    .withColumn("uses_method_credit_card", F.array_contains(col("rental_methods"), "CREDITCARD"))
)

display(silver_station_info_df)

# COMMAND ----------

silver_station_status_query = (
    silver_station_status_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("checkpointLocation", SILVER_STATION_INFO_CHECKPOINTS)
    .save(SILVER_STATION_INFO_PATH)
)

# COMMAND ----------

display(spark.sql(f"OPTIMIZE delta.`{SILVER_STATION_INFO_PATH}`"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Station status table

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The following is the schema for the station status silver table:
# MAGIC - `num_ebikes_available`: long (nullable = true)
# MAGIC - `is_installed`: long (nullable = true)
# MAGIC - `num_docks_available`: long (nullable = true)
# MAGIC - `num_scooters_unavailable`: double (nullable = true)
# MAGIC - `num_scooters_available`: double (nullable = true)
# MAGIC - `station_id`: string (nullable = true)
# MAGIC - `last_reported`: timestamp (nullable = true)
# MAGIC - `num_docks_disabled`: long (nullable = true)
# MAGIC - `is_renting`: long (nullable = true)
# MAGIC - `is_returning`: long (nullable = true)
# MAGIC - `eightd_has_available_keys`: boolean (nullable = true)
# MAGIC - `station_status`: string (nullable = true)
# MAGIC - `num_bikes_disabled`: long (nullable = true)
# MAGIC - `legacy_id`: string (nullable = true)
# MAGIC - `num_bikes_available`: long (nullable = true)
# MAGIC - `valet.region`: string (nullable = true)
# MAGIC - `valet.off_dock_capacity`: double (nullable = true)
# MAGIC - `valet.active`: boolean (nullable = true)
# MAGIC - `valet.dock_blocked_count`: double (nullable = true)
# MAGIC - `valet.off_dock_count`: double (nullable = true)
# MAGIC - `valet.station_id`: string (nullable = true)
# MAGIC - `valet.valet_revision`: double (nullable = true)
# MAGIC - `last_reported_date`: date (nullable = true)
# MAGIC - `last_reported_hour`: integer (nullable = true)
# MAGIC - `last_reported_minute`: integer (nullable = true)
# MAGIC - `last_reported_second`: integer (nullable = true)
# MAGIC 
# MAGIC The path to the table is stored in the following global variable: `SILVER_STATION_STATUS_PATH`.
# MAGIC 
# MAGIC The directory its checkpoints are saved at are stored in the following global variable: `SILVER_STATION_STATUS_CHECKPOINTS`.

# COMMAND ----------

# MAGIC %md
# MAGIC There are other transformations that can be done on the station staus silver table, but currently are not being done. This is because these transformations do not seem to serve a purpose for the rest of the team, and will essentially only slow down the project by adding unnecessary overhead and computations. Examples of these transformations are:
# MAGIC 
# MAGIC - The `num_scooters` column has some null values and could benefit from data imputation or filtering
# MAGIC - The `valet` columns have a lot of null values and could benefit from data imputation or filtering

# COMMAND ----------

silver_station_status_df = (
    bronze_station_status_df
    # Convert "last_reported" from unix time to a timestamp and extract components
    .withColumn("last_reported", from_unixtime("last_reported").cast("timestamp"))
    .withColumn("last_reported_date", to_date(col("last_reported")))
    .withColumn("last_reported_hour", hour(col("last_reported")))
    .withColumn("last_reported_minute", minute(col("last_reported")))
    .withColumn("last_reported_second", second(col("last_reported")))
)

display(silver_station_status_df)

# COMMAND ----------

silver_station_status_query = (
    silver_station_status_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("checkpointLocation", SILVER_STATION_STATUS_CHECKPOINTS)
    .save(SILVER_STATION_STATUS_PATH)
)

# COMMAND ----------

display(spark.sql(f"OPTIMIZE delta.`{SILVER_STATION_STATUS_PATH}`"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## NYC weather table

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The following is the schema for the NYC weather silver table:
# MAGIC - `dt`: timestamp (nullable = true)
# MAGIC - `tempK`: double (nullable = true)
# MAGIC - `feels_likeK`: double (nullable = true)
# MAGIC - `pressure`: long (nullable = true)
# MAGIC - `humidity`: long (nullable = true)
# MAGIC - `dew_point`: double (nullable = true)
# MAGIC - `uvi`: double (nullable = true)
# MAGIC - `clouds`: long (nullable = true)
# MAGIC - `visibility`: long (nullable = true)
# MAGIC - `wind_speed_mps`: double (nullable = true)
# MAGIC - `wind_deg`: long (nullable = true)
# MAGIC - `wind_gust_mps`: double (nullable = true)
# MAGIC - `weather`: struct (nullable = true)
# MAGIC   - `description`: string (nullable = true)
# MAGIC   - `icon`: string (nullable = true)
# MAGIC   - `id`: long (nullable = true)
# MAGIC   - `main`: string (nullable = true)
# MAGIC - `pop`: double (nullable = true)
# MAGIC - `rain`.1h: double (nullable = true)
# MAGIC - `time`: string (nullable = true)
# MAGIC - `tempC`: double (nullable = true)
# MAGIC - `tempF`: double (nullable = true)
# MAGIC - `feels_likeC`: double (nullable = true)
# MAGIC - `feels_likeF`: double (nullable = true)
# MAGIC - `wind_speed_mph`: double (nullable = true)
# MAGIC - `wind_gust_mph`: double (nullable = true)
# MAGIC - `dt_date`: date (nullable = true)
# MAGIC - `dt_hour`: integer (nullable = true)
# MAGIC - `dt_minute`: integer (nullable = true)
# MAGIC - `dt_second`: integer (nullable = true)
# MAGIC - `weather_description`: string (nullable = true)
# MAGIC - `weather_icon`: string (nullable = true)
# MAGIC - `weather_id`: long (nullable = true)
# MAGIC - `weather_main`: string (nullable = true)
# MAGIC 
# MAGIC The path to the table is stored in the following global variable: `SILVER_NYC_WEATHER_PATH`.
# MAGIC 
# MAGIC The directory its checkpoints are saved at are stored in the following global variable: `SILVER_NYC_WEATHER_CHECKPOINTS`.

# COMMAND ----------

silver_nyc_weather_df = (
    bronze_nyc_weather_df
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
    # Convert "wind_gust" from m/s to mph
    .withColumnRenamed("wind_gust", "wind_gust_mps")
    .withColumn("wind_gust_mph", col("wind_gust_mps") * 2.23694)
    # Convert "dt" from unix time to a timestamp and extract components
    .withColumn("dt", from_unixtime("dt").cast("timestamp"))
    .withColumn("dt_date", to_date(col("dt")))
    .withColumn("dt_hour", hour(col("dt")))
    .withColumn("dt_minute", minute(col("dt")))
    .withColumn("dt_second", second(col("dt")))
    # Separate weather map column into multiple columns
    .withColumn("weather", col("weather").getItem(0))
    .withColumn("weather_description", col("weather").getItem("description"))
    .withColumn("weather_icon", col("weather").getItem("icon"))
    .withColumn("weather_id", col("weather").getItem("id"))
    .withColumn("weather_main", col("weather").getItem("main"))
)

display(silver_nyc_weather_df)

# COMMAND ----------

silver_nyc_weather_query = (
    silver_nyc_weather_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("checkpointLocation", SILVER_NYC_WEATHER_CHECKPOINTS)
    .save(SILVER_NYC_WEATHER_PATH)
)

# COMMAND ----------

display(spark.sql(f"OPTIMIZE delta.`{SILVER_NYC_WEATHER_PATH}`"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Historical station table

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The following is the schema for the historical station silver table:
# MAGIC - `ride_id`: string (nullable = true)
# MAGIC - `rideable_type`: string (nullable = true)
# MAGIC - `started_at`: timestamp (nullable = true)
# MAGIC - `ended_at`: timestamp (nullable = true)
# MAGIC - `start_station_name`: string (nullable = true)
# MAGIC - `start_station_id`: string (nullable = true)
# MAGIC - `end_station_name`: string (nullable = true)
# MAGIC - `end_station_id`: string (nullable = true)
# MAGIC - `start_lat`: string (nullable = true)
# MAGIC - `start_lng`: string (nullable = true)
# MAGIC - `end_lat`: string (nullable = true)
# MAGIC - `end_lng`: string (nullable = true)
# MAGIC - `member_casual`: string (nullable = true)
# MAGIC - `station._rescued_data`: string (nullable = true)
# MAGIC - `started_at_date`: date (nullable = true)
# MAGIC - `started_at_hour`: integer (nullable = true)
# MAGIC - `started_at_minute`: integer (nullable = true)
# MAGIC - `started_at_second`: integer (nullable = true)
# MAGIC - `ended_at_date`: date (nullable = true)
# MAGIC - `ended_at_hour`: integer (nullable = true)
# MAGIC - `ended_at_minute`: integer (nullable = true)
# MAGIC - `ended_at_second`: integer (nullable = true)
# MAGIC - `is_holiday`: boolean (nullable = true)
# MAGIC 
# MAGIC The path to the table is stored in the following global variable: `SILVER_STATION_HISTORY_PATH`.
# MAGIC 
# MAGIC The directory its checkpoints are saved at are stored in the following global variable: `SILVER_STATION_HISTORY_CHECKPOINTS`.

# COMMAND ----------

# Get list of all US holidays in the last 10 years
current_year = datetime.now().year
us_holidays = holidays.US(years=range(current_year-10, current_year+1))

# COMMAND ----------

silver_station_history_df = (
    bronze_station_history_df
    # Convert "started_at" to a timestamp and extract components
    .withColumn("started_at", col("started_at").cast("timestamp"))
    .withColumn("started_at_date", to_date(col("started_at")))
    .withColumn("started_at_hour", hour(col("started_at")))
    .withColumn("started_at_minute", minute(col("started_at")))
    .withColumn("started_at_second", second(col("started_at")))
    # Convert "ended_at" to a timestamp and extract components
    .withColumn("ended_at", col("ended_at").cast("timestamp"))
    .withColumn("ended_at_date", to_date(col("ended_at")))
    .withColumn("ended_at_hour", hour(col("ended_at")))
    .withColumn("ended_at_minute", minute(col("ended_at")))
    .withColumn("ended_at_second", second(col("ended_at")))
    # Determine whether "started_at_date" is a holiday
    .withColumn("is_holiday", col("started_at_date").isin(list(us_holidays)))
    # # Filter to only show our group's station
    # .filter(col("start_station_name") == GROUP_STATION_ASSIGNMENT)
    # Rename the data evolution "_rescued_data" column
    .withColumnRenamed("_rescued_data", "station._rescued_data")
    # Add a watermark
    .withWatermark("started_at", "1 minutes")
)

display(silver_station_history_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Historical weather table

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The following is the schema for the historical weather silver table:
# MAGIC - `dt`: timestamp (nullable = true)
# MAGIC - `tempK`: string (nullable = true)
# MAGIC - `feels_likeK`: string (nullable = true)
# MAGIC - `pressure`: string (nullable = true)
# MAGIC - `humidity`: string (nullable = true)
# MAGIC - `dew_point`: string (nullable = true)
# MAGIC - `uvi`: string (nullable = true)
# MAGIC - `clouds`: string (nullable = true)
# MAGIC - `visibility`: string (nullable = true)
# MAGIC - `wind_speed_mps`: string (nullable = true)
# MAGIC - `wind_deg`: string (nullable = true)
# MAGIC - `pop`: string (nullable = true)
# MAGIC - `snow_1h`: string (nullable = true)
# MAGIC - `id`: string (nullable = true)
# MAGIC - `main`: string (nullable = true)
# MAGIC - `description`: string (nullable = true)
# MAGIC - `icon`: string (nullable = true)
# MAGIC - `loc`: string (nullable = true)
# MAGIC - `lat`: string (nullable = true)
# MAGIC - `lon`: string (nullable = true)
# MAGIC - `timezone`: string (nullable = true)
# MAGIC - `timezone_offset`: string (nullable = true)
# MAGIC - `rain_1h`: string (nullable = true)
# MAGIC - `weather._rescued_data`: string (nullable = true)
# MAGIC - `tempC`: double (nullable = true)
# MAGIC - `tempF`: double (nullable = true)
# MAGIC - `feels_likeC`: double (nullable = true)
# MAGIC - `feels_likeF`: double (nullable = true)
# MAGIC - `wind_speed_mph`: double (nullable = true)
# MAGIC - `dt_date`: date (nullable = true)
# MAGIC - `dt_hour`: integer (nullable = true)
# MAGIC - `dt_minute`: integer (nullable = true)
# MAGIC - `dt_second`: integer (nullable = true)
# MAGIC 
# MAGIC 
# MAGIC The path to the table is stored in the following global variable: `SILVER_WEATHER_HISTORY_PATH`.
# MAGIC 
# MAGIC The directory its checkpoints are saved at are stored in the following global variable: `SILVER_WEATHER_HISTORY_CHECKPOINTS`.

# COMMAND ----------

silver_weather_history_df = (
    bronze_weather_history_df
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
    .withColumn("dt_hour", hour(col("dt")))
    .withColumn("dt_minute", minute(col("dt")))
    .withColumn("dt_second", second(col("dt")))
    # Rename the data evolution "_rescued_data" column
    .withColumnRenamed("_rescued_data", "weather._rescued_data")
    # Add a watermark
    .withWatermark("dt", "1 minutes")
)

display(silver_weather_history_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Joined historical table

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The following is the schema for the joined historical silver table:
# MAGIC - `ride_id`: string (nullable = true)
# MAGIC - `rideable_type`: string (nullable = true)
# MAGIC - `started_at`: timestamp (nullable = true)
# MAGIC - `ended_at`: timestamp (nullable = true)
# MAGIC - `start_station_name`: string (nullable = true)
# MAGIC - `start_station_id`: string (nullable = true)
# MAGIC - `end_station_name`: string (nullable = true)
# MAGIC - `end_station_id`: string (nullable = true)
# MAGIC - `start_lat`: string (nullable = true)
# MAGIC - `start_lng`: string (nullable = true)
# MAGIC - `end_lat`: string (nullable = true)
# MAGIC - `end_lng`: string (nullable = true)
# MAGIC - `member_casual`: string (nullable = true)
# MAGIC - `station._rescued_data`: string (nullable = true)
# MAGIC - `started_at_date`: date (nullable = true)
# MAGIC - `started_at_hour`: integer (nullable = true)
# MAGIC - `started_at_minute`: integer (nullable = true)
# MAGIC - `started_at_second`: integer (nullable = true)
# MAGIC - `ended_at_date`: date (nullable = true)
# MAGIC - `ended_at_hour`: integer (nullable = true)
# MAGIC - `ended_at_minute`: integer (nullable = true)
# MAGIC - `ended_at_second`: integer (nullable = true)
# MAGIC - `is_holiday`: boolean (nullable = true)
# MAGIC - `dt`: timestamp (nullable = true)
# MAGIC - `tempK`: string (nullable = true)
# MAGIC - `feels_likeK`: string (nullable = true)
# MAGIC - `pressure`: string (nullable = true)
# MAGIC - `humidity`: string (nullable = true)
# MAGIC - `dew_point`: string (nullable = true)
# MAGIC - `uvi`: string (nullable = true)
# MAGIC - `clouds`: string (nullable = true)
# MAGIC - `visibility`: string (nullable = true)
# MAGIC - `wind_speed_mps`: string (nullable = true)
# MAGIC - `wind_deg`: string (nullable = true)
# MAGIC - `pop`: string (nullable = true)
# MAGIC - `snow_1h`: string (nullable = true)
# MAGIC - `id`: string (nullable = true)
# MAGIC - `main`: string (nullable = true)
# MAGIC - `description`: string (nullable = true)
# MAGIC - `icon`: string (nullable = true)
# MAGIC - `loc`: string (nullable = true)
# MAGIC - `lat`: string (nullable = true)
# MAGIC - `lon`: string (nullable = true)
# MAGIC - `timezone`: string (nullable = true)
# MAGIC - `timezone_offset`: string (nullable = true)
# MAGIC - `rain_1h`: string (nullable = true)
# MAGIC - `weather._rescued_data`: string (nullable = true)
# MAGIC - `tempC`: double (nullable = true)
# MAGIC - `tempF`: double (nullable = true)
# MAGIC - `feels_likeC`: double (nullable = true)
# MAGIC - `feels_likeF`: double (nullable = true)
# MAGIC - `wind_speed_mph`: double (nullable = true)
# MAGIC - `dt_date`: date (nullable = true)
# MAGIC - `dt_hour`: integer (nullable = true)
# MAGIC - `dt_minute`: integer (nullable = true)
# MAGIC - `dt_second`: integer (nullable = true)
# MAGIC 
# MAGIC The path to the table is stored in the following global variable: `SILVER_HISTORICAL_PATH`.
# MAGIC 
# MAGIC The directory its checkpoints are saved at are stored in the following global variable: `SILVER_HISTORICAL_CHECKPOINTS`.

# COMMAND ----------

silver_historical_df = silver_station_history_df.join(
    silver_weather_history_df,
    # Joining two streaming tables requires a watermark
    expr("""
        started_at_date = dt_date AND started_at_hour = dt_hour AND started_at_minute BETWEEN 0 and 59
    """),
    # Perform a left join so that all station rows are kept
    how="left",
)

display(silver_historical_df)

# COMMAND ----------

silver_historical_query = (
    silver_historical_df
    .writeStream
    .format("delta")
    .outputMode("append")
    .trigger(once=True)
    .option("mergeSchema", "true")
    .option("checkpointLocation", SILVER_HISTORICAL_CHECKPOINTS)
    .start(SILVER_HISTORICAL_PATH)
)

# COMMAND ----------

display(spark.sql(f"OPTIMIZE delta.`{SILVER_HISTORICAL_PATH}`"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Clean up
# MAGIC 
# MAGIC Finally, we can perform operations that will clean up and exit the notebook.

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
