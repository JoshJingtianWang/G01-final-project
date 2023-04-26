# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

# MAGIC %md 
# MAGIC # Imports

# COMMAND ----------

from pyspark.sql.functions import col, unix_timestamp, avg, expr, ceil, count, hour, count, when, col, month, expr, desc, date_format, dayofweek
import matplotlib.pyplot as plt
from pyspark.sql.functions import array, round
import seaborn as sns



# COMMAND ----------

# MAGIC %md 
# MAGIC # Load-In Dataset

# COMMAND ----------


all_bike_rentals = spark.read.format("delta").load("dbfs:/FileStore/tables/G01/silver_historical.delta/")
station_trips = all_bike_rentals.filter(all_bike_rentals.start_station_name == 'W 21 St & 6 Ave')


# COMMAND ----------

display(all_bike_rentals)

# COMMAND ----------

# MAGIC %md
# MAGIC # What is the distribution of rideable types?

# COMMAND ----------


rental_counts = station_trips.groupby("rideable_type").agg(
  count("rideable_type").alias("count")
).withColumn("percentage", round(col("count") / bike_rentals.count() * 100, 2))

rental_counts.toPandas().plot(kind="bar", x="rideable_type", y="percentage", legend=None)
plt.title("Percentage of Bike Rentals by Rideable Type")
plt.xlabel("Rideable Type")
plt.ylabel("Percentage (%)")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # How long are the rentals?

# COMMAND ----------


# Calculate the duration of each rental in seconds
station_trips = station_trips.withColumn(
  "duration_sec",
  unix_timestamp("ended_at") - unix_timestamp("started_at")
)

# Calculate the average duration of rentals in seconds, minutes, hours, days, weeks, months, and overall
duration_avg = station_trips.agg(
  avg(col("duration_sec")).alias("avg_duration_sec"),
  expr("avg(duration_sec) / 60").alias("avg_duration_min"),
).collect()[0]

# Print the average durations
print(f"Average rental duration in seconds: {duration_avg['avg_duration_sec']:.2f}")
print(f"Average rental duration in minutes: {duration_avg['avg_duration_min']:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # What is the average duration by bike type?

# COMMAND ----------

station_trips = station_trips.withColumn("start_time", unix_timestamp(col("started_at"), "yyyy-MM-dd HH:mm:ss"))
station_trips = station_trips.withColumn("end_time", unix_timestamp(col("ended_at"), "yyyy-MM-dd HH:mm:ss"))
station_trips = station_trips.withColumn("trip_duration", (col("end_time") - col("start_time"))/60)

display(station_trips.groupBy("rideable_type").agg(avg("trip_duration").alias("avg_trip_duration (min)"), count("*").alias("trip_count")))


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # What are the most popular end stations for rides that start at this station?

# COMMAND ----------

popular_end_stations = station_trips.groupBy('end_station_name').count().orderBy(desc('count'))
display(popular_end_stations)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # What is the average duration of rides that start at this station?

# COMMAND ----------

display(station_trips.agg(avg(expr("unix_timestamp(ended_at) - unix_timestamp(started_at)")).alias("avg_duration")))


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # What are the seasonal trip trends for each season?

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## How many bike rides are done each season?

# COMMAND ----------

station_trips = station_trips.withColumn("season", when((month(col("started_at")) >= 3) & (month(col("started_at")) <= 5), "Spring")
.when((month(col("started_at")) >= 6) & (month(col("started_at")) <= 8), "Summer")
.when((month(col("started_at")) >= 9) & (month(col("started_at")) <= 11), "Fall")
.otherwise("Winter"))

# Create a bar chart to show the total number of trips for each season
season_counts = station_trips.groupBy("season").count().orderBy("season").toPandas()
sns.barplot(x="season", y="count", data=season_counts)
plt.title("Total Number of Trips by Season")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Here we can see that Winter has the most bike rides with both Summer and Spring behind it. Fall doesn't seem to have that much data. It's hard to get a pattern here because we only have one year of data.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## What is the average trip duration by season?

# COMMAND ----------

# Create a boxplot to show the distribution of trip durations for each season
season_durations = station_trips.groupBy("season").agg(avg("trip_duration").alias("avg_duration")).orderBy("season").toPandas()
sns.barplot(x="season", y="avg_duration", data=season_durations)
plt.title("Average Trip Duration by Season")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We can see that both Fall and Summer have slightly longer trips, while both Spring and Winter are slightly lower. We can take a look at that average temperature to see if that's what is causing the different durations.

# COMMAND ----------

season_temps = station_trips.groupBy("season").agg(avg("tempF").alias("avg_temp")).orderBy("season").toPandas()
sns.barplot(x="season", y="avg_temp", data=season_temps)
plt.title("Average Temperature by Season")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC I am guessing that the weather conditions in Spring had elements of Winter making it harder for people to bike as long, while in Fall, there were elements of Summer. What I mean by this is that: People wanted to bike longer when there was better weather conditions, while when there was worse conditions, people just wanted to get quickly to their destination. We can look at the actual weather conditions to see if this is the case. Additionally, there is a lack of Fall data, so that could skew the visuals.

# COMMAND ----------


# Create a heatmap to show the number of trips by day of week and season
season_weekdays = station_trips.groupBy("season", date_format("started_at", "EEEE").alias("day_of_week")).count().orderBy(["season", "day_of_week"]).toPandas()
season_weekdays = season_weekdays.pivot(index="day_of_week", columns="season", values="count")
sns.heatmap(season_weekdays, cmap="Blues")
plt.title("Number of Trips by Day of Week and Season")
plt.show()


# COMMAND ----------

for season in ["Spring", "Summer", "Fall", "Winter"]:
    season_trips = station_trips.filter(station_trips.season == season)
    season_hourly = season_trips.groupBy(date_format("started_at", "EEEE").alias("day_of_week"), hour("started_at").alias("hour")).count().orderBy(["day_of_week", "hour"]).toPandas()
    sns.lineplot(x="hour", y="count", hue="day_of_week", data=season_hourly)
    plt.title(f"Number of Trips by Day of Week and Hour ({season})")
    plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # What are the monthly trip trends for each season? 

# COMMAND ----------


# Group the data by season and month, and count the number of trips
season_month_trips = station_trips.groupBy("season", month("started_at").alias("month")).count()

# Order the data by season and month
season_month_trips = season_month_trips.orderBy(["season", "month"])

# Convert to a pandas DataFrame and plot a line chart
season_month_trips_pd = season_month_trips.toPandas()
sns.barplot(x="month", y="count", hue="season", data=season_month_trips_pd)
plt.title("Total Number of Trips by Month and Season")
plt.show()


# COMMAND ----------

# Group the data by season, month, and start station, and calculate the average trip duration
season_month_duration = station_trips.groupBy("season", month("started_at").alias("month"), "start_station_name").agg(avg("trip_duration").alias("avg_duration"))

# Order the data by season, month, and average duration
season_month_duration = season_month_duration.orderBy(["season", "month", desc("avg_duration")])

# Convert to a pandas DataFrame and plot a line chart
season_month_duration_pd = season_month_duration.toPandas()
sns.barplot(x="month", y="avg_duration", hue="season", data=season_month_duration_pd)
plt.title("Average Trip Duration by Month and Season")
plt.show()


# COMMAND ----------

# Define the desired order of months and weekdays
month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Group the data by season, month, and day of week, and count the number of trips
season_month_weekday_trips = station_trips.groupBy("season", month("started_at").alias("month"), dayofweek("started_at").alias("weekday")).count()

# Order the data by season, month, and weekday
season_month_weekday_trips = season_month_weekday_trips.orderBy(["season", month("started_at"), dayofweek("started_at")])

# Convert to a pandas DataFrame and plot a heatmap
season_month_weekday_trips_pd = season_month_weekday_trips.toPandas()
season_month_weekday_trips_pd["month"] = pd.Categorical(season_month_weekday_trips_pd["month"].apply(lambda x: month_order[x-1]), categories=month_order)
season_month_weekday_trips_pd["weekday"] = pd.Categorical(season_month_weekday_trips_pd["weekday"].apply(lambda x: weekday_order[x-1]), categories=weekday_order)
sns.heatmap(data=season_month_weekday_trips_pd.pivot(index="weekday", columns=["month", "season"], values="count"), cmap="YlGnBu")
plt.title("Total Number of Trips by Month, Season, and Day of Week")
plt.show()


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

station_trips_pandas = station_trips.toPandas()
station_trips_pandas['started_at_month'] = pd.to_datetime(station_trips['started_at']).dt.month

# Group the data by month and user type, and count the number of trips
monthly_user_trips = station_trips_pandas.groupby(['started_at_month', 'member_casual']).size().reset_index(name='count')

# Pivot the data to have user types as columns and months as rows
monthly_user_trips_pivot = monthly_user_trips.pivot(index='started_at_month', columns='member_casual', values='count')

# Plot the data as a line graph
monthly_user_trips_pivot.plot(kind='line', figsize=(10,5))
plt.title('Monthly Trend by User Type')
plt.xlabel('Month')
plt.ylabel('Number of Trips')
plt.show()


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Convert the started_at column to datetime type and extract the month and hour
station_trips_pandas['started_at_month'] = pd.to_datetime(station_trips['started_at']).dt.month
station_trips_pandas['started_at_hour'] = pd.to_datetime(station_trips['started_at']).dt.hour

# Group the data by month and hour, and count the number of trips
hourly_trips = station_trips.groupby(['started_at_month', 'started_at_hour']).size().reset_index(name='count')

# Pivot the data to have hours as columns and months as rows
hourly_trips_pivot = hourly_trips.pivot(index='started_at_month', columns='started_at_hour', values='count')

# Plot the data as a line graph
hourly_trips_pivot.plot(kind='line', figsize=(10,5))
plt.title('Hourly Trend by Month')
plt.xlabel('Month')
plt.ylabel('Number of Trips')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # What are the seasonal trip trends for each season?

# COMMAND ----------

display(station_trips)

# COMMAND ----------

weather_counts = station_trips['main'].value_counts()
weather_counts.plot(kind='bar')


# COMMAND ----------


# Convert the `started_at` column to a pandas datetime object
station_trips['started_at'] = pd.to_datetime(station_trips['started_at'])

# Group the data by month and weather type, and count the number of trips for each group
monthly_weather_trips = station_trips.groupby([station_trips['started_at'].dt.month, 'main'])['ride_id'].count().reset_index(name='num_trips')

# Plot a line chart showing the trend of trips over different weather types for each month
sns.lineplot(data=monthly_weather_trips, x='started_at', y='num_trips', hue='main')


# COMMAND ----------

sns.countplot(data=station_trips, x='member_casual', hue='main')


# COMMAND ----------


# Group data by hour of day and weather type
hourly_weather = station_trips.groupby([station_trips['started_at'].dt.hour, 'main'])['ride_id'].count().reset_index(name='num_trips')

# Create line plot
sns.lineplot(data=hourly_weather, x='hour', y='tempF', hue='main')
sns.lineplot(data=hourly_weather, x='hour', y='ride_count', hue='main')


# COMMAND ----------

dbutils.fs.ls('dbfs:/FileStore/tables/G01/')


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 1) get a start_at_month column
# MAGIC 2) ohe the rideable type
# MAGIC 3) add a started_at_season column
# MAGIC 4) start the visualization process, today's goal is to just simply understand what is going on, don't have to have meaningful outputs

# COMMAND ----------

bike_rentals = spark.read.format("delta").load("dbfs:/FileStore/tables/G01/silver_historical.delta/")



# COMMAND ----------

display(bike_rentals)

# COMMAND ----------



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



# COMMAND ----------



# COMMAND ----------

from pyspark.sql.functions import count

# Assuming your DataFrame is named 'df' and your columns are named 'month' and 'bike'
# Group by the 'month' column and compute the average of the 'bike' column
df_avg_bike = df.groupBy('month').agg(count('total_bikes').alias('avg_bike')).

# Show the result
df_avg_bike.show()


# COMMAND ----------

display(df.avg_bike)

# COMMAND ----------

from pyspark.sql.functions import hour
from pyspark.sql.functions import count
from pyspark.sql.functions import hour, count, when, col


display(df.groupBy(col('dt_hour')).agg(count(when(col("total_bikes") == 1, 1))))


# COMMAND ----------

from pyspark.sql.functions import hour, count, when, col

capacity = 78

usage_by_hour = df.groupBy(col('dt_hour')).agg(count(when(col("total_bikes") == 1, 1)).alias("num_bikes_used"))
usage_by_hour = usage_by_hour.withColumn("capacity_used_percent", usage_by_hour["num_bikes_used"] / capacity * 100)



# COMMAND ----------

from pyspark.sql.functions import col, least, count, when, lit

capacity = 78

usage_by_hour_minute = df.groupBy(col("dt_hour"), col("dt_minute")).agg(least(count(when(col("total_bikes") == 1, 1)), lit(capacity)).alias("num_bikes_used"))
usage_by_hour_minute = usage_by_hour_minute.withColumn("capacity_used_percent", usage_by_hour_minute["num_bikes_used"] / lit(capacity) * 100)


# COMMAND ----------

display(usage_by_hour_minute)

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

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))

# COMMAND ----------


