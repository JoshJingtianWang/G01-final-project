# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 1) get a start_at_month column
# MAGIC 2) ohe the rideable type
# MAGIC 3) add a started_at_season column
# MAGIC 4) start the visualization process, today's goal is to just simply understand what is going on, don't have to have meaningful outputs

# COMMAND ----------

df = spark.read.format("delta").load("dbfs:/FileStore/tables/G01/silver_historical.delta/")



# COMMAND ----------

display(df)

# COMMAND ----------

from pyspark.sql.functions import month

# Assuming your DataFrame is named 'df' and the column is named 'started_at_date'
df = df.withColumn("month", month(df.started_at_date))


# COMMAND ----------

from pyspark.sql.functions import when

# Assuming your DataFrame is named 'df' and the column is named 'rideable_type'
df = df.withColumn("total_bikes", when(df.rideable_type == "classic_bike", 1).\
                  when(df.rideable_type == "electric_bike", 1).\
                  when(df.rideable_type == "docked_bike", 0).\
                  otherwise(0))



# COMMAND ----------

display(df)

# COMMAND ----------

from pyspark.sql.functions import col, when

df = df.withColumn("seasons", when((col("month") == 12) | (col("month") <= 2), "Winter").\
                  when((col("month") >= 3) & (col("month") <= 5), "Spring").\
                  when((col("month") >= 6) & (col("month") <= 8), "Summer").\
                  when((col("month") >= 9) & (col("month") <= 11), "Fall").\
                  otherwise("Unknown"))


# COMMAND ----------

display(df)

# COMMAND ----------

from pyspark.sql.functions import dayofweek, dayofmonth

# Assuming your DataFrame is named 'df' and the date column is named 'started_at_date'
df = df.withColumn("day_of_week", dayofweek("started_at_date"))
df = df.withColumn("day_of_month", dayofmonth("started_at_date"))


# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Going to look at every season except for Summer, since I know the data isn't there. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Winter

# COMMAND ----------

winter = df.filter(col('seasons') == 'Winter')
display(winter)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Seems like December is the most active month for bikes, so I am going to visualize these in two groups: December, and then January and February. Added the other graphs just for awareness, nothing to really get from there until we dive deeper.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## December

# COMMAND ----------

december = df.filter(col('month') == 12)
display(december)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Curious why there are dips in general on: 3rd, 6th, 9th, 10th, 15th, 16th, and 18th. After that, it's the holiday season, so it's understandable why there aren't as many bikers.

# COMMAND ----------

# Define the desired day of month values
day_values = [3, 6, 9, 10, 15, 16, 18]

# Filter the 'winter' DataFrame by the desired day of month values
december_filtered_neg = december.filter(col("day_of_month").isin(day_values))
december_filtered_pos = december.filter(~ col("day_of_month").isin(day_values))


# COMMAND ----------

display(december_filtered_neg)

# COMMAND ----------

display(december_filtered_pos)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In December, we are able to see that evenings regardless of dips or peaks, are when bikes are being used. However, there are dips more on weekends, and on weekdays, bikes are being used more. Curious if this a trend or not.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Jan and Feb

# COMMAND ----------

other_winter = df.filter((col('month') == 1) | (col('month') == 2))
display(other_winter)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Again, there are dips and peaks. Let's take a look at them.

# COMMAND ----------

# Define the desired day of month values
day_values = [3, 7, 11, 12, 14, 19, 20, 24, 22, 25, 29,31]

# Filter the 'winter' DataFrame by the desired day of month values
other_filtered_neg = other_winter.filter(col("day_of_month").isin(day_values))
other_filtered_pos = other_winter.filter(~ col("day_of_month").isin(day_values))


# COMMAND ----------

display(other_filtered_neg)

# COMMAND ----------

display(other_filtered_pos)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## WINTER RECAP
# MAGIC 
# MAGIC We learned that weekdays around the evening are most important times. We will look at holidays more in depth, but in December, the holiday time period made bikes go down. Weekdays + Evenings is the current trend, let's look at other seasons to see if it's still valid.

# COMMAND ----------

# MAGIC %md 
# MAGIC # Spring

# COMMAND ----------

spring = df.filter(col('seasons') == 'Spring')
display(spring)

# COMMAND ----------

# MAGIC %md
# MAGIC Gonna split it up into 2 groups again: March & April AND May

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## March & April

# COMMAND ----------

main_spring = df.filter((col('month') == 3) | (col('month') == 4))
display(main_spring)


# COMMAND ----------

# Define the desired day of month values
day_values = [3, 6, 9, 12, 17, 19, 23, 24, 24, 28, 31]

# Filter the 'winter' DataFrame by the desired day of month values
other_filtered_neg = main_spring.filter(col("day_of_month").isin(day_values))
other_filtered_pos = main_spring.filter(~ col("day_of_month").isin(day_values))


# COMMAND ----------

display(other_filtered_neg)

# COMMAND ----------

display(other_filtered_pos)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC There are more bikes being used on weekends on the negative dips. Same trends

# COMMAND ----------

may = df.filter((col('month') == 5))
display(may)


# COMMAND ----------

# Define the desired day of month values
day_values = [2, 6, 7, 14, 16, 18, 22, 24, 24, 27, 28]

# Filter the 'winter' DataFrame by the desired day of month values
other_filtered_neg = may.filter(col("day_of_month").isin(day_values))
other_filtered_pos = may.filter(~ col("day_of_month").isin(day_values))


# COMMAND ----------

display(other_filtered_neg)

# COMMAND ----------

display(other_filtered_pos)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC SAME TREND! May most likely had more bikes used, just because of how warm it got, we will check later.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # FALL

# COMMAND ----------

fall = df.filter(col('seasons') == 'Fall')
display(fall)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Gonna look at Fall as one group.

# COMMAND ----------

# Define the desired day of month values
day_values = [11, 13, 15, 14, 18, 18, 24, 25, 31]

# Filter the 'winter' DataFrame by the desired day of month values
other_filtered_neg = fall.filter(col("day_of_month").isin(day_values))
other_filtered_pos = fall.filter(~ col("day_of_month").isin(day_values))


# COMMAND ----------

display(other_filtered_neg)

# COMMAND ----------

display(other_filtered_pos)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #LEGACY

# COMMAND ----------

start_date = str(dbutils.widgets.get('01.start_date'))
end_date = str(dbutils.widgets.get('02.end_date'))
hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
promote_model = bool(True if str(dbutils.widgets.get('04.promote_model')).lower() == 'yes' else False)

print(start_date,end_date,hours_to_forecast, promote_model)
print("YOUR CODE HERE...")

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


