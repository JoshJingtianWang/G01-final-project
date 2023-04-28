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
# MAGIC
# MAGIC LOOK INTO: DISABLED BIKES

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
).withColumn("percentage", round(col("count") / station_trips.count() * 100, 2))

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



# COMMAND ----------

# code that looks at different weather patterns by season
from pyspark.sql.functions import desc, dense_rank
from pyspark.sql.window import Window

# group by season and main weather condition, and count the number of occurrences
season_weather = station_trips.groupBy("season", "main").count()

# rank the weather conditions within each season by their count in descending order
season_weather_ranked = season_weather.withColumn("rank", dense_rank().over(Window.partitionBy("season").orderBy(desc("count"))))

# filter to only include the top 5 weather conditions for each season
season_weather_top5 = season_weather_ranked.filter("rank <= 5")

# convert to pandas dataframe for visualization
season_weather_top5_pd = season_weather_top5.orderBy("season", "rank").toPandas()

# plot the data using seaborn
sns.catplot(x="season", y="count", hue="main", kind="bar", data=season_weather_top5_pd)
plt.title("Top 5 Main Weather Conditions by Season")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC In both Spring and Winter, there is snow AND rain, with both having considerably less clear skies. Additionally, Summer seems to have more clear weather, while having less "hazardous weather." Now we want to look at the difference in number of trips AND average trip duration by each day of the week. 

# COMMAND ----------

import calendar

# Create a heatmap to show the number of trips by day of week and season
season_weekdays = station_trips.groupBy("season", date_format("started_at", "EEEE").alias("day_of_week")).count().orderBy(["season", "day_of_week"]).toPandas()

# pivot the data and reorder the rows to start with Monday
season_weekdays_pivot = season_weekdays.pivot(index="day_of_week", columns="season", values="count")
days_of_week = list(calendar.day_name)
season_weekdays_pivot = season_weekdays_pivot.reindex(days_of_week)

# plot the data using seaborn
sns.heatmap(season_weekdays_pivot, cmap="Blues", annot=True, fmt="d")
plt.title("Number of Trips by Day of Week and Season")
plt.xlabel("Season")
plt.ylabel("Day of Week")
plt.show()


# COMMAND ----------

import calendar

# Create a heatmap to show the average trip duration by day of week and season
season_weekdays = station_trips.groupBy("season", date_format("started_at", "EEEE").alias("day_of_week")).agg(avg("trip_duration").alias("avg_duration")).orderBy(["season", "day_of_week"]).toPandas()

# pivot the data and reorder the rows to start with Monday
season_weekdays_pivot = season_weekdays.pivot(index="day_of_week", columns="season", values="avg_duration")
days_of_week = list(calendar.day_name)
season_weekdays_pivot = season_weekdays_pivot.reindex(days_of_week)

# plot the data using seaborn
sns.heatmap(season_weekdays_pivot, cmap="Blues", annot=True, fmt=".1f")
plt.title("Average Trip Duration by Day of Week and Season")
plt.xlabel("Season")
plt.ylabel("Day of Week")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Here we can easily see that weekdays have more trips overall, while weekends tend to have less. Additionally, the average durations are longer on weekends (except for Mondays in Fall but I'll consider that a lack of data skewing it). My initial hypothesis is that the majority of use of bikes in the station are for people using it to get to work on WEEKDAYS, but on WEEKENDS, it is used primarily for leisure. We will look at the seasons by an hourly basis.

# COMMAND ----------

for season in ["Spring", "Summer", "Fall", "Winter"]:
    season_trips = station_trips.filter(station_trips.season == season)
    season_hourly = season_trips.groupBy(date_format("started_at", "EEEE").alias("day_of_week"), hour("started_at").alias("hour")).count().orderBy(["day_of_week", "hour"]).toPandas()
    sns.lineplot(x="hour", y="count", hue="day_of_week", data=season_hourly)
    plt.title(f"Number of Trips by Day of Week and Hour ({season})")
    plt.show()


# COMMAND ----------

for season in ["Spring", "Summer", "Fall", "Winter"]:
    season_trips = station_trips.filter(station_trips.season == season)
    season_hourly = season_trips.groupBy(date_format("started_at", "EEEE").alias("day_of_week"), hour("started_at").alias("hour")).agg(avg("trip_duration").alias("avg_duration")).orderBy(["day_of_week", "hour"]).toPandas()
    sns.lineplot(x="hour", y="avg_duration", hue="day_of_week", data=season_hourly)
    plt.title(f"Average Duration of Trips by Day of Week and Hour ({season})")
    plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Count of Trips: We can see that evenings in all 4 seasons are more prevalent. 
# MAGIC
# MAGIC Average Duration: Generally speaking it seems like trip durations are longer in the mornings across all 4 seasons, while evenings have shorter bike rides. The real mystery is why there are random peaks in all 4 seasons. 

# COMMAND ----------

import seaborn as sns

sns.boxplot(x="season", y="duration_sec", data=station_trips.toPandas())
plt.title("Boxplot of Trip Durations by Season")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We are going to look at these outliers.

# COMMAND ----------

high_duration_trips = station_trips.filter(station_trips.duration_sec > 0.2)
display(high_duration_trips)


# COMMAND ----------

median_duration = high_duration_trips.selectExpr("percentile_approx(duration_sec, 0.5)").collect()[0][0]
print("Median trip duration: {:.2f} seconds".format(median_duration))


# COMMAND ----------

from pyspark.sql.functions import mean, median, stddev

mean_duration = high_duration_trips.agg(mean("duration_sec")).collect()[0][0]
median_duration = high_duration_trips.approxQuantile("duration_sec", [0.5], 0.25)[0]
stddev_duration = high_duration_trips.agg(stddev("duration_sec")).collect()[0][0]

print(f"Mean Duration: {mean_duration}")
print(f"Median Duration: {median_duration}")
print(f"Standard Deviation Duration: {stddev_duration}")


# COMMAND ----------

import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import seaborn as sns
import calendar


# Calculate average trip duration by season
season_duration = (
    high_duration_trips
    .groupBy("season")
    .agg(F.avg("duration_sec").alias("avg_duration"))
    .orderBy("season")
    .toPandas()
)
sns.barplot(x="season", y="avg_duration", data=season_duration)
plt.title("Average Trip Duration by Season")
plt.show()


# COMMAND ----------

condition_duration = (
    high_duration_trips
    .groupBy("main")
    .agg(F.avg("duration_sec").alias("avg_duration"))
    .orderBy("avg_duration", ascending=False)
    .toPandas()
)
sns.barplot(x="main", y="avg_duration", data=condition_duration)
plt.title("Average Trip Duration by Weather Condition")
plt.xticks(rotation=45, ha='right')
plt.show()


# COMMAND ----------

duration_by_user_type = (
    high_duration_trips
    .groupBy("member_casual")
    .agg(F.avg("duration_sec").alias("avg_duration"))
    .orderBy("member_casual")
    .toPandas()
)
sns.barplot(x="member_casual", y="avg_duration", data=duration_by_user_type)
plt.title("Average Trip Duration by User Type")
plt.show()


# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.functions import to_timestamp

# Convert the timestamp column to a datetime object
high_duration_trips = high_duration_trips.withColumn("started_at", to_timestamp("started_at"))

# Create a scatter plot of trip duration vs. start time, colored by user type
sns.scatterplot(x="started_at", y="duration_sec", hue="member_casual", data=high_duration_trips.toPandas())

# Add a vertical line at the cutoff value
plt.axhline(y=200000, color="red", linestyle="--", label="Cutoff Value")

# Filter the data for durations above the cutoff value
outliers = high_duration_trips.filter(high_duration_trips.duration_sec > 200000).toPandas()

# Plot the outliers as red points
sns.scatterplot(x="started_at", y="duration_sec", hue="member_casual", data=outliers, color="red", s=100, label="Outliers")

# Set the y-axis limits to include the full range of values
plt.ylim(bottom=0, top=high_duration_trips.select(F.max("duration_sec")).collect()[0][0])

plt.title("Trip Duration vs. Start Time, Colored by User Type")
plt.legend()
plt.show()





# COMMAND ----------

display(outliers)

# COMMAND ----------

duration_by_user_type = (
    high_duration_trips
    .groupBy("rideable_type")
    .agg(F.avg("duration_sec").alias("avg_duration"))
    .orderBy("rideable_type")
    .toPandas()
)
sns.barplot(x="rideable_type", y="avg_duration", data=duration_by_user_type)
plt.title("Average Trip Duration by User Type")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Seems like docked_bikes are causing the avg. duration to be really high, but I guess that makes sense?

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # What are the monthly trip trends for each season? 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC First we are going to look at the total number of rentals by month.

# COMMAND ----------

import matplotlib.pyplot as plt

monthly_rentals = station_trips.groupBy(month("started_at").alias("month")).agg(count("*").alias("total_rentals")).orderBy("month")

# Extract the month and total rentals columns from the DataFrame
months = monthly_rentals.select("month").rdd.flatMap(lambda x: x).collect()
rentals = monthly_rentals.select("total_rentals").rdd.flatMap(lambda x: x).collect()

# Plot the data using a bar chart
plt.bar(months, rentals)
plt.title("Monthly Rentals")
plt.xlabel("Month")
plt.ylabel("Total Rentals")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Next we are going to look at the average duration by month.

# COMMAND ----------

# Calculate average monthly duration
monthly_duration = station_trips.groupBy(month("started_at").alias("month")).agg(avg("duration_sec").alias("avg_duration")).orderBy("month")

# Extract the month and average duration columns from the DataFrame
months_duration = monthly_duration.select("month").rdd.flatMap(lambda x: x).collect()
avg_durations = monthly_duration.select("avg_duration").rdd.flatMap(lambda x: x).collect()

# Plot the data using a line chart
plt.plot(months, avg_durations)
plt.title("Average Monthly Duration")
plt.xlabel("Month")
plt.ylabel("Average Duration (seconds)")
plt.show()




# COMMAND ----------

# MAGIC %md 
# MAGIC Now let's look at the average monthly temperature.

# COMMAND ----------

import matplotlib.pyplot as plt

# Calculate average monthly temperature
monthly_temp = station_trips.groupBy(month("started_at").alias("month")).agg(avg("tempF").alias("avg_tempF"))

# Extract the month and average temperature columns from the DataFrame
months_temp = monthly_temp.select("month").rdd.flatMap(lambda x: x).collect()
avg_tempF = monthly_temp.select("avg_tempF").rdd.flatMap(lambda x: x).collect()

plt.bar(months, avg_tempF)
plt.title("Average Monthly Temperature")
plt.xlabel("Month")
plt.ylabel("Average Temp (F)")
plt.show()




# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Let's see both monthly temperature and duration on the same graph.

# COMMAND ----------


# Plot the data using a bar chart for temperature and line chart for duration
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Month')
ax1.set_ylabel('Average Duration (seconds)', color=color)
ax1.plot(months_duration, avg_durations, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Average Temperature (Fahrenheit)', color=color)
ax2.bar(months_temp, avg_tempF, color=color, alpha=0.5)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Monthly Trends')
plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Now let's look at the proportion of member vs. casual riders by month

# COMMAND ----------

monthly_member_casual = station_trips.groupBy(month("started_at").alias("month")) \
                                     .agg(sum(when(station_trips.member_casual == "member", 1)).alias("total_members"), 
                                          sum(when(station_trips.member_casual == "casual", 1)).alias("total_casuals")) \
                                     .withColumn("total_rentals", col("total_members") + col("total_casuals")) \
                                     .withColumn("member_proportion", col("total_members") / col("total_rentals")) \
                                     .withColumn("casual_proportion", col("total_casuals") / col("total_rentals")) \
                                     .orderBy("month")
monthly_member_casual.show()


# COMMAND ----------

import matplotlib.pyplot as plt

# Extract the month and total members/casuals columns from the DataFrame
months = monthly_member_casual.select("month").rdd.flatMap(lambda x: x).collect()
total_members = monthly_member_casual.select("total_members").rdd.flatMap(lambda x: x).collect()
total_casuals = monthly_member_casual.select("total_casuals").rdd.flatMap(lambda x: x).collect()

# Plot the data using a line chart
plt.plot(months, total_members, label="Total Members")
plt.plot(months, total_casuals, label="Total Casuals")
plt.title("Total Monthly Rentals by Rider Type")
plt.xlabel("Month")
plt.ylabel("Total Rentals")
plt.legend()
plt.show()


# COMMAND ----------

# MAGIC %md We are just going to look at the rental count by hour and day of week and month

# COMMAND ----------

from pyspark.sql.types import IntegerType

monthly_day_weekday = station_trips.groupBy(month("started_at").alias("month"), dayofweek("started_at").alias("day_of_week")) \
                        .agg(count("*").alias("rental_count")) \
                        .withColumn("month", col("month").cast(IntegerType())) \
                        .orderBy("month", "day_of_week")


monthly_hourly = station_trips.groupBy(month("started_at").alias("month"), hour("started_at").alias("hour")) \
                    .agg(count("*").alias("rental_count")) \
                    .withColumn("month", col("month").cast(IntegerType())) \
                    .orderBy("month", "hour")





# COMMAND ----------




import seaborn as sns
import matplotlib.pyplot as plt

monthly_day_weekday_pd = monthly_day_weekday.toPandas()

# Reshape the dataframe to have months as rows and hours as columns
monthly_weekly_pivot = monthly_day_weekday_pd = monthly_day_weekday.toPandas().pivot("month", "day_of_week", "rental_count")

# Order the months
monthly_weekly_pivot = monthly_weekly_pivot.reindex([1,2,3,4,5,6,7,8,9,10,11,12])

# Create the heatmap
plt.figure(figsize=(12,8))
sns.heatmap(monthly_weekly_pivot, cmap="YlGnBu")
plt.title("Day Of Week Rentals by Month")
plt.xlabel("Day Of Week")
plt.ylabel("Month")
plt.show()



# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

monthly_hourly_pd = monthly_hourly.toPandas()

# Reshape the dataframe to have months as rows and hours as columns
monthly_hourly_pivot = monthly_hourly_pd.pivot("month", "hour", "rental_count")

# Order the months
monthly_hourly_pivot = monthly_hourly_pivot.reindex([1,2,3,4,5,6,7,8,9,10,11,12])

# Create the heatmap
plt.figure(figsize=(12,8))
sns.heatmap(monthly_hourly_pivot, cmap="YlGnBu")
plt.title("Hourly Rentals by Month")
plt.xlabel("Hour of Day")
plt.ylabel("Month")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##What are the daily trip trends for your given station?

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC How many daily trips are there for the given station?

# COMMAND ----------

from pyspark.sql.functions import countDistinct, to_date, date_format

# Add a new column to the station_trips dataframe with the date
station_trips = station_trips.withColumn("date", to_date("started_at"))

# Group by date to count daily trips for each day
daily_trips_by_day = station_trips.groupBy("date").agg(countDistinct("ride_id").alias("daily_trips")).orderBy("date")

# Convert Spark dataframe to Pandas dataframe
daily_trips_by_day_pd = daily_trips_by_day.toPandas()

# Create a plot of daily trip trends
plt.plot(daily_trips_by_day_pd["date"], daily_trips_by_day_pd["daily_trips"])

# Add labels and title
plt.xlabel("Date")
plt.ylabel("Number of Daily Trips")
plt.title("Daily Trip Trends")

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the chart
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC What is the average trip duration per day for the given station?

# COMMAND ----------

from pyspark.sql.functions import avg, to_date

# Add a new column to the station_trips dataframe with the date
station_trips = station_trips.withColumn("date", to_date("started_at"))

# Group by date to calculate the average trip duration for each day
avg_duration_by_day = station_trips.groupBy("date").agg(avg("trip_duration").alias("avg_duration")).orderBy("date")

# Convert Spark dataframe to Pandas dataframe
avg_duration_by_day_pd = avg_duration_by_day.toPandas()

# Create a plot of daily average trip duration trends with a log y-axis
plt.plot(avg_duration_by_day_pd["date"], np.log(avg_duration_by_day_pd["avg_duration"]))

# Add labels and title
plt.xlabel("Date")
plt.ylabel("Log Average Trip Duration (seconds)")
plt.title("Daily Average Trip Duration Trends (log scale)")

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the chart
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC How do trip trends vary by weekday and weekend for the given station?

# COMMAND ----------

from pyspark.sql.functions import countDistinct, to_date, date_format, when

# Define a function to categorize days as either weekday or weekend
def weekday_or_weekend(date):
    day_of_week = date_format(date, "EEEE")
    return when(day_of_week == "Saturday", "weekend").when(day_of_week == "Sunday", "weekend").otherwise("weekday")

# Add a new column to the station_trips dataframe with the day category
station_trips = station_trips.withColumn("day_category", weekday_or_weekend(to_date("started_at")))

# Group by day category and date to count daily trips for each category
daily_trips_by_day_category = station_trips.groupBy("day_category", to_date("started_at").alias("date")).agg(countDistinct("ride_id").alias("daily_trips")).orderBy("date")

# Convert Spark dataframe to Pandas dataframe
daily_trips_by_day_category_pd = daily_trips_by_day_category.toPandas()

# Create separate plots for weekdays and weekends
weekday_trips = daily_trips_by_day_category_pd[daily_trips_by_day_category_pd["day_category"] == "weekday"]
weekend_trips = daily_trips_by_day_category_pd[daily_trips_by_day_category_pd["day_category"] == "weekend"]

plt.plot(weekday_trips["date"], weekday_trips["daily_trips"], label="Weekdays")
plt.plot(weekend_trips["date"], weekend_trips["daily_trips"], label="Weekends")

# Add labels and title
plt.xlabel("Date")
plt.ylabel("Number of Daily Trips")
plt.title("Daily Trip Trends by Day Category")
plt.xticks(rotation=45, ha='right')

# Add legend
plt.legend()

# Show the chart
plt.show()


# COMMAND ----------

from pyspark.sql.functions import countDistinct, to_date, date_format, when, round, avg
import numpy as np

# Define a function to categorize days as either weekday or weekend
def weekday_or_weekend(date):
    day_of_week = date_format(date, "EEEE")
    return when(day_of_week == "Saturday", "weekend").when(day_of_week == "Sunday", "weekend").otherwise("weekday")

# Add a new column to the station_trips dataframe with the day category
station_trips = station_trips.withColumn("day_category", weekday_or_weekend(to_date("started_at")))

# Round duration to the nearest minute and add a new column
station_trips = station_trips.withColumn("duration_minutes", round(station_trips.trip_duration/60))

# Group by day category, date, and duration to calculate the average trip duration for each category, date, and duration
avg_duration_by_day_category_duration = station_trips.groupBy("day_category", to_date("started_at").alias("date"), "duration_minutes").agg(avg("trip_duration").alias("avg_duration")).orderBy("date")

# Convert Spark dataframe to Pandas dataframe
avg_duration_by_day_category_duration_pd = avg_duration_by_day_category_duration.toPandas()

# Create separate plots for weekdays and weekends
weekday_duration = avg_duration_by_day_category_duration_pd[avg_duration_by_day_category_duration_pd["day_category"] == "weekday"]
weekend_duration = avg_duration_by_day_category_duration_pd[avg_duration_by_day_category_duration_pd["day_category"] == "weekend"]

# Plot weekday trip duration trends
plt.plot(weekday_duration["date"], np.log(weekday_duration["avg_duration"]), label="Weekday")

# Plot weekend trip duration trends
plt.plot(weekend_duration["date"], np.log(weekend_duration["avg_duration"]), label="Weekend")

# Add labels and title
plt.xlabel("Date")
plt.ylabel("Average Trip Duration (log scale)")
plt.title("Daily Average Trip Duration Trends by Day Category and Duration")
plt.xticks(rotation=45, ha='right')

# Add legend
plt.legend()

# Show the chart
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC How do trip trends vary by member type?

# COMMAND ----------

from pyspark.sql.functions import countDistinct

# Group by member type and date to count daily trips for each member type
daily_trips_by_member_type = station_trips.groupBy("member_casual", to_date("started_at").alias("date")).agg(countDistinct("ride_id").alias("daily_trips")).orderBy("date")

# Convert Spark dataframe to Pandas dataframe
daily_trips_by_member_type_pd = daily_trips_by_member_type.toPandas()

# Create separate plots for member types
casual_trips = daily_trips_by_member_type_pd[daily_trips_by_member_type_pd["member_casual"] == "casual"]
member_trips = daily_trips_by_member_type_pd[daily_trips_by_member_type_pd["member_casual"] == "member"]

plt.plot(casual_trips["date"], casual_trips["daily_trips"], label="Casual")
plt.plot(member_trips["date"], member_trips["daily_trips"], label="Member")

# Add labels and title
plt.xlabel("Date")
plt.ylabel("Number of Daily Trips")
plt.title("Daily Trip Trends by Member Type")
plt.xticks(rotation=45, ha='right')

# Add legend
plt.legend()

# Show the chart
plt.show()


# COMMAND ----------

from pyspark.sql.functions import avg

# Group by member type and date to calculate average trip duration per day for each member type
daily_trips_by_member_type = station_trips.groupBy("member_casual", to_date("started_at").alias("date")).agg(avg("trip_duration").alias("avg_duration")).orderBy("date")

# Convert Spark dataframe to Pandas dataframe
daily_trips_by_member_type_pd = daily_trips_by_member_type.toPandas()

# Create separate plots for member types
casual_trips = daily_trips_by_member_type_pd[daily_trips_by_member_type_pd["member_casual"] == "casual"]
member_trips = daily_trips_by_member_type_pd[daily_trips_by_member_type_pd["member_casual"] == "member"]

plt.plot(casual_trips["date"], casual_trips["avg_duration"], label="Casual")
plt.plot(member_trips["date"], member_trips["avg_duration"], label="Member")

# Add labels and title
plt.xlabel("Date")
plt.ylabel("Average Trip Duration (seconds)")
plt.title("Daily Trip Duration Trends by Member Type")
plt.xticks(rotation=45, ha='right')

# Add legend
plt.legend()
plt.yscale('log')
# Show the chart
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## How does a holiday affect the daily (non-holiday) system use trend?

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Do markdown later

# COMMAND ----------

from pyspark.sql.functions import avg, count
import matplotlib.pyplot as plt

# group the data by holiday status and calculate the mean number of rentals and trip duration
rentals_by_holiday = (station_trips
                      .groupBy('is_holiday')
                      .agg(count('ride_id').alias('count_rentals'))
                      .orderBy('is_holiday'))
trip_duration_by_holiday = (station_trips
                            .groupBy('is_holiday')
                            .agg(avg('trip_duration').alias('avg_trip_duration'))
                            .orderBy('is_holiday'))

# create a bar plot to visualize the differences in bike rentals during holidays vs. non-holidays
rentals_by_holiday_pd = rentals_by_holiday.toPandas()
rentals_by_holiday_pd.plot(kind='bar', x='is_holiday', y='count_rentals', color=['blue', 'green'])
plt.title('Average Daily Bike Rentals During Holidays vs. Non-Holidays')
plt.xlabel('Holiday Status')
plt.ylabel('Average Daily Bike Rentals')

# create a box plot to visualize the differences in trip duration during holidays vs. non-holidays
station_trips_pd = station_trips.toPandas()
station_trips_pd.boxplot(column='trip_duration', by='is_holiday')
plt.title('Distribution of Trip Duration During Holidays vs. Non-Holidays')
plt.xlabel('Holiday Status')
plt.ylabel('Trip Duration (seconds)')
plt.show()


# COMMAND ----------

holiday_trips = station_trips.filter(station_trips.is_holiday == True)


# COMMAND ----------

import matplotlib.pyplot as plt
from pyspark.sql.functions import col

# Filter for holidays only
holiday_trips = station_trips.filter(station_trips.is_holiday == True)

# Group by date and count trips
daily_trip_count = (holiday_trips
                   .groupBy('started_at_date')
                   .agg({'ride_id': 'count'})
                   .withColumnRenamed('count(ride_id)', 'trip_count')
                   .orderBy(col('started_at_date')))

# Convert to Pandas dataframe
daily_trip_count_pandas = daily_trip_count.toPandas()

# Create a line plot of daily trip count
plt.plot(daily_trip_count_pandas['started_at_date'], daily_trip_count_pandas['trip_count'])

# Format the plot
plt.title('Daily Bike Rental Count for Holidays')
plt.xlabel('Date')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45, ha='right')
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
from pyspark.sql.functions import col

# Filter for holidays only
holiday_trips = station_trips.filter(station_trips.is_holiday == True)

# Group by date and count trips
daily_average_duration = (holiday_trips
                   .groupBy('started_at_date')
                   .agg({'trip_duration': 'average'})
                   .withColumnRenamed('average(trip_duration)', 'trip_duration')
                   .orderBy(col('started_at_date')))

# Convert to Pandas dataframe
daily_average_duration_pandas = daily_average_duration.toPandas()

# Create a line plot of daily trip count
plt.plot(daily_average_duration_pandas['started_at_date'], daily_average_duration_pandas['avg(trip_duration)'])

# Format the plot
plt.title('Daily Average Trip Duration for Holidays')
plt.xlabel('Date')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45, ha='right')
plt.show()


# COMMAND ----------

holiday_trips

# COMMAND ----------

display(holiday_trips)

# COMMAND ----------

holiday_trips = holiday_trips.toPandas()

# COMMAND ----------

import matplotlib.pyplot as plt

# Group the data by hour and count the number of trips
hourly_trip_count = holiday_trips.groupby(holiday_trips.started_at_hour).size().reset_index(name='count')

# Create a bar plot of hourly trip count
plt.plot(hourly_trip_count['started_at_hour'], hourly_trip_count['count'])

# Format the plot
plt.title('Hourly Bike Rental Count on Holidays')
plt.xlabel('Hour')
plt.ylabel('Number of Trips')
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt

# Group the data by hour and calculate the average trip duration
hourly_trip_duration = holiday_trips.groupby(holiday_trips.started_at_hour)['trip_duration'].mean().reset_index(name='avg_duration')

# Create a bar plot of hourly average trip duration
plt.plot(hourly_trip_duration['started_at_hour'], hourly_trip_duration['avg_duration'])

# Format the plot
plt.title('Hourly Average Bike Rental Duration on Holidays')
plt.xlabel('Hour')
plt.ylabel('Average Duration (minutes)')
plt.yscale('log')

plt.show()


# COMMAND ----------

import seaborn as sns

# Create a new dataframe grouped by hour and member_casual with count of trips
hourly_trip_count_member_casual = holiday_trips.groupby(['started_at_hour', 'member_casual'])['ride_id'].count().reset_index(name='count')

# Create a line plot of hourly trip count with hue of member_casual
sns.lineplot(data=hourly_trip_count_member_casual, x='started_at_hour', y='count', hue='member_casual')

# Format the plot
plt.title('Hourly Bike Rental Count during Holidays')
plt.xlabel('Hour')
plt.ylabel('Number of Trips')

plt.show()


# COMMAND ----------

# Create a new dataframe grouped by hour and member_casual with mean trip duration
hourly_trip_duration_member_casual = holiday_trips.groupby(['started_at_hour', 'member_casual'])['trip_duration'].mean().reset_index(name='mean_duration')

# Create a line plot of hourly mean trip duration with hue of member_casual
sns.lineplot(data=hourly_trip_duration_member_casual, x='started_at_hour', y='mean_duration', hue='member_casual')

# Format the plot
plt.title('Hourly Average Trip Duration during Holidays')
plt.xlabel('Hour')
plt.ylabel('Average Trip Duration (minutes)')

plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## How does weather affect the daily/hourly trend of system use?

# COMMAND ----------

from pyspark.sql.functions import date_format, hour

# Group the data by date, hour, and weather type
daily_trips = (
    station_trips
    .groupBy(date_format('started_at', 'yyyy-MM-dd').alias('date'),
             hour('started_at').alias('hour'),
             'main')
)

# Aggregate the number of trips and weather conditions
daily_trips = (
    daily_trips
    .agg({'ride_id': 'count',
          'tempF': 'mean',
          'wind_speed_mph': 'mean',
          'trip_duration': 'mean'})
    .withColumnRenamed('count(ride_id)', 'num_trips')
    .withColumnRenamed('avg(tempF)', 'avg_tempF')
    .withColumnRenamed('avg(wind_speed_mph)', 'avg_wind_speed_mph')
    .withColumnRenamed('avg(trip_duration)', 'trip_duration')

)


# COMMAND ----------

display(daily_trips)

# COMMAND ----------

import seaborn as sns

# Convert the dataframe to pandas
daily_trips_pd = daily_trips.toPandas()

# Create an area plot with multiple series
fig, ax = plt.subplots(figsize=(10, 6))
sns.set_palette('bright')
sns.lineplot(x='hour', y='trip_duration', hue='main', data=daily_trips_pd, ax=ax, ci=None, estimator='mean', alpha=0.8, linewidth=2.5)
ax.set_xlabel('Hour')
ax.set_ylabel('Average Trip Duration')
ax.legend()
plt.show()


# COMMAND ----------

import seaborn as sns

# Convert the dataframe to pandas
daily_trips_pd = daily_trips.toPandas()

# Create an area plot with multiple series
fig, ax = plt.subplots(figsize=(10, 6))
sns.set_palette('bright')
sns.lineplot(x='hour', y='avg_wind_speed_mph', hue='main', data=daily_trips_pd, ax=ax, ci=None, estimator='mean', alpha=0.8, linewidth=2.5)
ax.set_xlabel('Hour')
ax.set_ylabel('Average Wind Speed')
ax.legend()
plt.show()


# COMMAND ----------

import seaborn as sns

# Convert the dataframe to pandas
daily_trips_pd = daily_trips.toPandas()

# Create an area plot with multiple series
fig, ax = plt.subplots(figsize=(10, 6))
sns.set_palette('bright')
sns.lineplot(x='hour', y='avg_tempF', hue='main', data=daily_trips_pd, ax=ax, ci=None, estimator='mean', alpha=0.8, linewidth=2.5)
ax.set_xlabel('Hour')
ax.set_ylabel('Average Temperature (F)')
ax.legend()
plt.show()


# COMMAND ----------

import seaborn as sns

# Convert the dataframe to pandas
daily_trips_pd = daily_trips.toPandas()

# Create an area plot with multiple series
fig, ax = plt.subplots(figsize=(10, 6))
sns.set_palette('bright')
sns.lineplot(x='hour', y='num_trips', hue='main', data=daily_trips_pd, ax=ax, ci=None, estimator='mean', alpha=0.8, linewidth=2.5)
ax.set_xlabel('Hour')
ax.set_ylabel('Number of bike rentals')
ax.legend()
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt

# Convert the dataframe to pandas
daily_trips_pd = daily_trips.toPandas()

# Create a scatter plot of temperature vs. number of bike rentals
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(daily_trips_pd['avg_tempF'], daily_trips_pd['num_trips'], alpha=0.5, color='blue')
ax.set_xlabel('Average temperature (°F)')
ax.set_ylabel('Number of bike rentals')
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt

# Convert the dataframe to pandas
daily_trips_pd = daily_trips.toPandas()

# Create a scatter plot of temperature vs. number of bike rentals
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(daily_trips_pd['avg_wind_speed_mph'], daily_trips_pd['num_trips'], alpha=0.5, color='blue')
ax.set_xlabel('Average temperature (°F)')
ax.set_ylabel('Number of bike rentals')
plt.show()


# COMMAND ----------


