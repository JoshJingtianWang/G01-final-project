# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

display(dbutils.fs.ls(f"{GROUP_DATA_PATH}"))

# COMMAND ----------

#YC_WEATHER_FILE_PATH = 'dbfs:/FileStore/tables/raw/weather/'

dbutils.fs.ls('dbfs:/FileStore/tables/raw/weather/NYC_Weather_Data.csv')

# COMMAND ----------

start_date = str(dbutils.widgets.get('01.start_date'))
end_date = str(dbutils.widgets.get('02.end_date'))
hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
promote_model = bool(True if str(dbutils.widgets.get('04.promote_model')).lower() == 'yes' else False)

print(start_date,end_date,hours_to_forecast, promote_model)
print("YOUR CODE HERE...")

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import from_unixtime
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import from_unixtime, col
from pyspark.sql.functions import from_unixtime, to_utc_timestamp, date_format
from pyspark.sql.functions import date_format

BRONZE_STATION_INFO_PATH = "dbfs:/FileStore/tables/bronze_station_info.delta"
BRONZE_STATION_STATUS_PATH = "dbfs:/FileStore/tables/bronze_station_status.delta"
BRONZE_NYC_WEATHER_PATH = "dbfs:/FileStore/tables/bronze_nyc_weather.delta"


station_info_df = spark.read.format("delta").load(BRONZE_STATION_INFO_PATH)
station_status_df = spark.read.format("delta").load(BRONZE_STATION_STATUS_PATH)
nyc_weather_df = spark.read.format("delta").load(BRONZE_NYC_WEATHER_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC # Looking at Started_At
# MAGIC 
# MAGIC Right here, I'm trying to just set up the basic graphs for one dataset, but when the ETL comes in, it should change.

# COMMAND ----------

from pyspark.sql.functions import to_timestamp, month, dayofmonth, hour, dayofweek


data = spark.read.format('csv').option('header', 'true').load('dbfs:/FileStore/tables/raw/bike_trips/202302_citibike_tripdata.csv')
data = data.filter(data['start_station_name'] == "W 21 St & 6 Ave")

data = data.withColumn("timestamp_started_at", to_timestamp("started_at", "yyyy-MM-dd HH:mm:ss"))
data = data.withColumn("timestamp_ended_at", to_timestamp("ended_at", "yyyy-MM-dd HH:mm:ss"))

data = data.withColumn("month_started_at", month("timestamp_started_at"))
data = data.withColumn("day_started_at", dayofmonth("timestamp_started_at"))
data = data.withColumn("hour_started_at", hour("timestamp_started_at"))
data = data.withColumn("dayofweek_started_at", dayofweek("timestamp_started_at"))

data = data.withColumn("month_ending_at", month("timestamp_ended_at"))
data = data.withColumn("day_ending_at", dayofmonth("timestamp_ended_at"))
data = data.withColumn("hour_ending_at", hour("timestamp_ended_at"))
data = data.withColumn("dayofweek_ending_at", dayofweek("timestamp_ended_at"))


# COMMAND ----------

display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC We notice that the week-days have the most bike rentals, so let's look at the hours specifically for each week day.

# COMMAND ----------

monday = data.filter(data['dayofweek_started_at'] == 1)
tuesday = data.filter(data['dayofweek_started_at'] == 2)
wednesday = data.filter(data['dayofweek_started_at'] == 3)
thursday = data.filter(data['dayofweek_started_at'] == 4)
friday = data.filter(data['dayofweek_started_at'] == 5)


# COMMAND ----------

display(monday)

# COMMAND ----------

display(tuesday)

# COMMAND ----------

display(wednesday)

# COMMAND ----------

display(thursday)

# COMMAND ----------

display(friday)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Gonna make an assumption here and I'm gonna say that 9-5 is the work schedule of most workers in this area, therefore, it seems like people on weekdays are generally getting their bikes when they go home, around the 4PM to 8PM time.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## ending_at
# MAGIC 
# MAGIC Gonna look at the same exact code but for the ending_at

# COMMAND ----------

display(data)

# COMMAND ----------

monday = data.filter(data['dayofweek_ending_at'] == 1)
tuesday = data.filter(data['dayofweek_ending_at'] == 2)
wednesday = data.filter(data['dayofweek_ending_at'] == 3)
thursday = data.filter(data['dayofweek_ending_at'] == 4)
friday = data.filter(data['dayofweek_ending_at'] == 5)


# COMMAND ----------

display(monday)

# COMMAND ----------

display(tuesday)

# COMMAND ----------

display(wednesday)

# COMMAND ----------

display(thursday)

# COMMAND ----------

display(friday)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The same trends are valid for the ending times as well. It seems like overall the bikes are being used to go home rather than go to work. I don't have the monthly trends yet, nor can I look at seasonal trends, but this is my current hypothesis.

# COMMAND ----------

weather_data = spark.read.format('csv').option('header', 'true').load('dbfs:/FileStore/tables/raw/weather/NYC_Weather_Data.csv')
weather_data = weather_data.withColumn("temp_f", (col("temp") - 273.15) * 9/5 + 32)

weather_data = weather_data.withColumn("time", from_unixtime("dt"))

weather_data = weather_data.withColumn("timestamp", to_timestamp("time", "yyyy-MM-dd HH:mm:ss"))

weather_data = weather_data.withColumn("month", month("timestamp"))
weather_data = weather_data.withColumn("day", dayofmonth("timestamp"))
weather_data = weather_data.withColumn("hour", hour("timestamp"))
weather_data = weather_data.withColumn("dayofweek", dayofweek("timestamp"))


# COMMAND ----------

display(weather_data)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC For now, let's look at trends with temperature.

# COMMAND ----------

display(weather_data)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC This data alone won't really do anything, but we can notice that summer basically isn't in the dataset. Let's try to combine our current weather datasets with bike dataset.

# COMMAND ----------

display(data)
display(weather_data)



# COMMAND ----------

# Convert the started_at column in data to TimestampType
data = data.withColumn("started_at", to_timestamp("started_at", "yyyy-MM-dd HH:mm:ss"))

# Convert the TimestampType column to the desired format
data = data.withColumn("started_at", date_format("started_at", "yyyy-MM-dd HH:00:00"))


# COMMAND ----------

display(data)

# COMMAND ----------


# Join the two DataFrames on the renamed datetime column and the started_at column
joined_data = data.join(weather_data, data.started_at == weather_data.time)


# COMMAND ----------

display(joined_data)

# COMMAND ----------

display(data)
display(weather_data)



# COMMAND ----------

display(data)
data = spark.read.format('csv').option('header', 'true').load('dbfs:/FileStore/tables/raw/weather/NYC_Weather_Data.csv')

display(data)

# COMMAND ----------

display(data.groupBy('rideable_type').count().orderBy(F.desc('count')))


# COMMAND ----------



# COMMAND ----------

from pyspark.sql.functions import to_timestamp

station_info_df = station_info_df.filter(station_info_df['name'] == "W 21 St & 6 Ave")
station_status_df = station_status_df.filter(station_status_df['station_id'] == "66dc120f-0aca-11e7-82f6-3863bb44ef7c")
station_df = station_info_df.join(station_status_df, on='station_id')
station_df = station_df.withColumn('time', to_timestamp('last_reported'))
nyc_weather_df = nyc_weather_df.withColumn('time', to_timestamp('dt'))


# COMMAND ----------

from pyspark.sql.functions import to_timestamp

station_df = station_df.withColumn('time', to_timestamp('last_reported'))

# COMMAND ----------

from pyspark.sql.functions import col

station_df = station_df.withColumn("num_vehicles_used", col("capacity") - col("num_docks_available"))
station_df = station_df.withColumn("num_bikes_used", col("capacity") - col("num_ebikes_available") - col("num_docks_available"))
station_df = station_df.withColumn("num_ebikes_used", col("capacity") - col("num_bikes_available") - col("num_docks_available"))

# COMMAND ----------

from pyspark.sql.functions import to_timestamp, month, dayofmonth, hour, dayofweek

station_df = station_df.withColumn("timestamp", to_timestamp("time", "yyyy-MM-dd'T'HH:mm:ss.SSSZ"))

station_df = station_df.withColumn("month", month("timestamp"))
station_df = station_df.withColumn("day", dayofmonth("timestamp"))
station_df = station_df.withColumn("hour", hour("timestamp"))
station_df = station_df.withColumn("dayofweek", dayofweek("timestamp"))


# COMMAND ----------

display(station_df)

# COMMAND ----------

display(station_df)

# COMMAND ----------

display(station_df)

# COMMAND ----------

from pyspark.sql.functions import to_timestamp, month, dayofmonth, hour, dayofweek

nyc_weather_df = nyc_weather_df.withColumn("timestamp", to_timestamp("time", "yyyy-MM-dd'T'HH:mm:ss.SSSZ"))

nyc_weather_df = nyc_weather_df.withColumn("month", month("timestamp"))
nyc_weather_df = nyc_weather_df.withColumn("day", dayofmonth("timestamp"))
nyc_weather_df = nyc_weather_df.withColumn("hour", hour("timestamp"))
nyc_weather_df = nyc_weather_df.withColumn("dayofweek", dayofweek("timestamp"))


# COMMAND ----------

from pyspark.sql.functions import col

nyc_weather_df = nyc_weather_df.withColumn("temp_f", (col("temp") - 273.15) * 9/5 + 32)


# COMMAND ----------

display(nyc_weather_df)





# COMMAND ----------

display(nyc_weather_df)


# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))

# COMMAND ----------


