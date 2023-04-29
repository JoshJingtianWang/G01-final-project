# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

# MAGIC %md 
# MAGIC # Imports

# COMMAND ----------

from pyspark.sql.functions import col, unix_timestamp, avg, expr, ceil, count, hour, count, when, col, month, expr, desc, date_format, dayofweek, avg
import matplotlib.pyplot as plt
from pyspark.sql.functions import array, round
import seaborn as sns
import calendar
import pyspark.sql.functions as F
from pyspark.sql.functions import dense_rank
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType
import pandas as pd
from pyspark.sql.functions import count, mean


# COMMAND ----------

# MAGIC %md 
# MAGIC # Load-In Dataset

# COMMAND ----------


all_bike_rentals = spark.read.format("delta").load("dbfs:/FileStore/tables/G01/silver_historical.delta/") # use this dataset if you wanna look at all the stations
station_trips = all_bike_rentals.filter(all_bike_rentals.start_station_name == 'W 21 St & 6 Ave') # this is the main dataset


# COMMAND ----------

display(station_trips)

# COMMAND ----------

# MAGIC %md 
# MAGIC #Initial Analysis
# MAGIC
# MAGIC We are take a look at that the dataset and analyze it with simple visualizations just to get a feel for the data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bike Rentals by Rideable Type

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
# MAGIC
# MAGIC We see that classic_bike is the most prevalent bike type here. Although this doesn't necessarily say that classic_bikes are more popular, it does say that the data has a majoirty of classic_bikes.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Average Trip Duration by Rideable Type

# COMMAND ----------

station_trips = station_trips.withColumn("start_time", unix_timestamp(col("started_at"), "yyyy-MM-dd HH:mm:ss"))
station_trips = station_trips.withColumn("end_time", unix_timestamp(col("ended_at"), "yyyy-MM-dd HH:mm:ss"))
station_trips = station_trips.withColumn("trip_duration", (col("end_time") - col("start_time"))/60)

display(station_trips.groupBy("rideable_type").agg(avg("trip_duration").alias("avg_trip_duration (min)"), count("*").alias("trip_count")))


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC We can see that docked_bikes have a very trip duration, while both electric_bikes and classic_bikes are around 12 minutes.

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Count of Trips by User Type

# COMMAND ----------

count_by_user_type = (
    station_trips
    .groupBy("member_casual")
    .agg(F.count("rideable_type").alias("trip_count"))
    .orderBy("member_casual")
    .toPandas()
)
sns.barplot(x="member_casual", y="trip_count", data=count_by_user_type)
plt.title("Count of Trips by User Type")
plt.show()


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Average Trip Duration by User Type

# COMMAND ----------

duration_by_user_type = (
    station_trips
    .groupBy("member_casual")
    .agg(F.avg("trip_duration").alias("avg_duration"))
    .orderBy("member_casual")
    .toPandas()
)
sns.barplot(x="member_casual", y="avg_duration", data=duration_by_user_type)
plt.title("Average Trip Duration by User Type")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We can see that most of the bike trips are done by members, but casual user types tend to have longer bike rides.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Popular End Stations Based on Trip Counts

# COMMAND ----------

popular_end_stations = station_trips.groupBy('end_station_name').count().orderBy(desc('count'))
display(popular_end_stations)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC If there was more time, we would've loved to look into the end_stations, however, it isn't worth pursuing.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # What are the seasonal trip trends for each season?
# MAGIC
# MAGIC Now we are going to look at the data grouped by each season.

# COMMAND ----------

station_trips = station_trips.withColumn("season", when((month(col("started_at")) >= 3) & (month(col("started_at")) <= 5), "Spring")
                                              .when((month(col("started_at")) >= 6) & (month(col("started_at")) <= 8), "Summer")
                                              .when((month(col("started_at")) >= 9) & (month(col("started_at")) <= 11), "Fall")
                                              .otherwise("Winter"))


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Total Number of Trips by Season

# COMMAND ----------

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
# MAGIC ## Average Duration by Season

# COMMAND ----------

season_durations = station_trips.groupBy("season").agg(avg("trip_duration").alias("avg_duration")).orderBy("season").toPandas()
sns.barplot(x="season", y="avg_duration", data=season_durations)
plt.title("Average Trip Duration by Season")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We can see that both Fall and Summer have slightly longer trips, while both Spring and Winter are slightly lower. We can take a look at that average temperature to see if that's what is causing the different durations.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Average Temperature by Season

# COMMAND ----------

season_temps = station_trips.groupBy("season").agg(avg("tempF").alias("avg_temp")).orderBy("season").toPandas()
sns.barplot(x="season", y="avg_temp", data=season_temps)
plt.title("Average Temperature by Season")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Summer has warmer weather, while Spring and Winter have lower temperatures. Fall also has lower temperatures, but it is still in the 50s. We can take a look at the actual weather conditions during these seasons to further analyze the difference in durations. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Weather Conditions by Season

# COMMAND ----------

season_weather = station_trips.groupBy("main", "season").count()
season_weather_ranked = season_weather.withColumn("rank", dense_rank().over(Window.partitionBy("season").orderBy(desc("count"))))
season_weather_top5 = season_weather_ranked.filter("rank <= 5").orderBy(["season", "rank"]).toPandas()
season_weather_pivot = season_weather_top5.pivot(index="season", columns="main", values="count")
season_weather_pivot.plot(kind="bar", stacked=True)
plt.title("Top 5 Main Weather Conditions by Season")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC In both Spring and Winter, there is snow and rain, with both having considerably less clear skies. Additionally, Summer seems to have more clear weather, while having less "hazardous weather." Now we want to look at the difference in number of trips and average trip duration by each day of the week. 
# MAGIC
# MAGIC In both Spring and Winter, the weather conditions force people to only use bikes for getting to work rather than leisure. We can take a look at seasons in an hourly and weekly basis.

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Number of Trips by Day of Week and Season

# COMMAND ----------


season_weekdays = station_trips.groupBy("season", date_format("started_at", "EEEE").alias("day_of_week")).count().orderBy(["season", "day_of_week"]).toPandas()

season_weekdays_pivot = season_weekdays.pivot(index="day_of_week", columns="season", values="count")
days_of_week = list(calendar.day_name)
season_weekdays_pivot = season_weekdays_pivot.reindex(days_of_week)

sns.heatmap(season_weekdays_pivot, cmap="Blues", annot=True, fmt="d")
plt.title("Number of Trips by Day of Week and Season")
plt.xlabel("Season")
plt.ylabel("Day of Week")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Average Trip Duration by Day of Week and Season

# COMMAND ----------

season_weekdays = station_trips.groupBy("season", date_format("started_at", "EEEE").alias("day_of_week")).agg(avg("trip_duration").alias("avg_duration")).orderBy(["season", "day_of_week"]).toPandas()

season_weekdays_pivot = season_weekdays.pivot(index="day_of_week", columns="season", values="avg_duration")
days_of_week = list(calendar.day_name)
season_weekdays_pivot = season_weekdays_pivot.reindex(days_of_week)

sns.heatmap(season_weekdays_pivot, cmap="Blues", annot=True, fmt=".1f")
plt.title("Average Trip Duration by Day of Week and Season")
plt.xlabel("Season")
plt.ylabel("Day of Week")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We can see that weekdays generally have more bike rides in general, but weekends especially in Summer have a longer duration. This leads me to believe that people are generally using bikes for getting to work (or going home), and if they use it on weekends, it's for leisure. We can take a look at the hourly basis by duration and number of bikes to further explore this.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Number of Trips and Average Duration by Day of Week and Hour for Each Season

# COMMAND ----------

for season in ["Spring", "Summer", "Fall", "Winter"]:
    season_trips = station_trips.filter(station_trips.season == season)
    
    season_hourly_counts = season_trips.groupBy(date_format("started_at", "EEEE").alias("day_of_week"), hour("started_at").alias("hour")).count().orderBy(["day_of_week", "hour"]).toPandas()
    sns.lineplot(x="hour", y="count", hue="day_of_week", data=season_hourly_counts)
    plt.title(f"Number of Trips by Day of Week and Hour ({season})")
    plt.show()
    
    season_hourly_avg_duration = season_trips.groupBy(date_format("started_at", "EEEE").alias("day_of_week"), hour("started_at").alias("hour")).agg(avg("trip_duration").alias("avg_duration")).orderBy(["day_of_week", "hour"]).toPandas()
    sns.lineplot(x="hour", y="avg_duration", hue="day_of_week", data=season_hourly_avg_duration)
    plt.title(f"Average Duration of Trips by Day of Week and Hour ({season})")
    plt.show()


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC From the number of trips graphs, we can see that they all follow the same pattern where people are using the bikes in the evenings. From previous analysis, we learned that people are mostly using bikes to travel to work. Using these graphs, we can conclude that people are using bikes to leave work but not to go to work. Additionally, Each season follows its own pattern for average duration, and it makes sense mostly, as Winter has longer durations, while Summer has shorter durations. However, in each duration graph there are random peaks that don't really make sense.

# COMMAND ----------

high_duration_trips = station_trips.filter(station_trips.trip_duration > 20)
display(high_duration_trips)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## High Duration Average Trip Duration by Season
# MAGIC
# MAGIC We are going to take a look at these filtered dataset to see if anything is not normal.

# COMMAND ----------

season_duration = (
    high_duration_trips
    .groupBy("season")
    .agg(F.avg("trip_duration").alias("avg_duration"))
    .orderBy("season")
    .toPandas()
)
sns.barplot(x="season", y="avg_duration", data=season_duration)
plt.title("Average Trip Duration by Season")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## High Duration  by Weather Condition

# COMMAND ----------

condition_duration = (
    high_duration_trips
    .groupBy("main")
    .agg(F.avg("trip_duration").alias("avg_duration"))
    .orderBy("avg_duration", ascending=False)
    .toPandas()
)
sns.barplot(x="main", y="avg_duration", data=condition_duration)
plt.title("Average Trip Duration by Weather Condition")
plt.xticks(rotation=45, ha='right')
plt.show()


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## High Duration Count of Trips by User Type

# COMMAND ----------

count_by_user_type = (
    high_duration_trips
    .groupBy("member_casual")
    .agg(F.count("rideable_type").alias("trip_count"))
    .orderBy("member_casual")
    .toPandas()
)
sns.barplot(x="member_casual", y="trip_count", data=count_by_user_type)
plt.title("Count of Trips by User Type")
plt.show()


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## High Duration Average Trip Duration by User Type

# COMMAND ----------

duration_by_user_type = (
    high_duration_trips
    .groupBy("member_casual")
    .agg(F.avg("trip_duration").alias("avg_duration"))
    .orderBy("member_casual")
    .toPandas()
)
sns.barplot(x="member_casual", y="avg_duration", data=duration_by_user_type)
plt.title("Average Trip Duration by User Type")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC This is different than the overall version of these graphs. There are more casual members in this filtered dataset, while the average duration stays the same. We learn here that casual user_types like to use their bikes longer. We can further explore this by plotting the average trip duration by start time. 

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## High Duration Average Trip Duration vs. Start Time, Colored by User Type

# COMMAND ----------

avg_duration_by_user_type = (
    high_duration_trips
    .groupBy("member_casual", "started_at")
    .agg(avg("trip_duration").alias("avg_duration"))
    .orderBy("started_at")
    .toPandas()
)

sns.scatterplot(x="started_at", y="avg_duration", hue="member_casual", data=avg_duration_by_user_type)
plt.title("Average Trip Duration vs. Start Time, Colored by User Type")
plt.legend()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We can take a look at these outliers in more depth.

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## High Duration Average Trip Duration by User Type and Rideable Type

# COMMAND ----------

duration_by_user_type = (
    high_duration_trips
    .groupBy("rideable_type", "member_casual")
    .agg(F.avg("trip_duration").alias("avg_duration"), F.stddev("trip_duration").alias("stddev_duration"))
    .orderBy("rideable_type")
    .toPandas()
)

sns.barplot(x="rideable_type", y="avg_duration", hue="member_casual", data=duration_by_user_type)
plt.title("Average Trip Duration by User Type")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We can see that casual members who rent the docked bike type have absurdly long trip durations, while electric bikes and classic bikes have more durations from 40-60 minutes. Seems like this particular docked bike may be an outlier.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # What are the monthly trip trends for each season? 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Trip Count by Month

# COMMAND ----------


monthly_rentals_trip = station_trips.groupBy(month("started_at").alias("month")).agg(count("*").alias("trip_count")).orderBy("month").toPandas()

plt.plot(monthly_rentals_trip["month"], monthly_rentals_trip["trip_count"])
plt.title("Monthly Rentals")
plt.xlabel("Month")
plt.ylabel("Trip Count")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC There is a peak in march, while there is a dip in april. There is also a lack of data in September and October. 

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Average Duration by Month

# COMMAND ----------

monthly_rentals_duration = (
    station_trips
    .groupBy(month("started_at").alias("month"))
    .agg(avg("trip_duration").alias("avg_duration"))
    .orderBy("month")
    .toPandas()
)

plt.plot(monthly_rentals_duration["month"], monthly_rentals_duration["avg_duration"])
plt.title("Monthly Average Duration")
plt.xlabel("Month")
plt.ylabel("Average Duration")
plt.show()


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Average Temperature by Month

# COMMAND ----------

monthly_rentals_temp = (
    station_trips
    .groupBy(month("started_at").alias("month"))
    .agg(avg("tempF").alias("avg_tempF"))
    .orderBy("month")
    .toPandas()
)

plt.plot(monthly_rentals_temp["month"], monthly_rentals_temp["avg_tempF"])
plt.title("Monthly Average Temperature")
plt.xlabel("Month")
plt.ylabel("Average Temperature (F)")
plt.show()



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## All 3 Graphs

# COMMAND ----------

max_temp = monthly_rentals_temp["avg_tempF"].max()
max_duration = monthly_rentals_duration["avg_duration"].max()
max_count = monthly_rentals_trip["trip_count"].max()

monthly_rentals_temp["norm_temp"] = monthly_rentals_temp["avg_tempF"] / max_temp * 0.7
monthly_rentals_duration["norm_duration"] = monthly_rentals_duration["avg_duration"] / max_duration * 0.7
monthly_rentals_trip["norm_count"] = monthly_rentals_trip["trip_count"] / max_count * 1.0

sns.lineplot(data=monthly_rentals_temp, x="month", y="norm_temp", label="Temperature")
sns.lineplot(data=monthly_rentals_duration, x="month", y="norm_duration", label="Duration")
sns.lineplot(data=monthly_rentals_trip, x="month", y="norm_count", label="Trip Count")
plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
plt.title("Monthly Averages")
plt.xlabel("Month")
plt.ylabel("Value (Normalized)")
plt.show()


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Note that we had to normalize all 3 columns because the trip count was so large compared to the others. Seems like the temperature and duration are positively correlated. We can further analyze the monthly trends by look at it on an hourly basis and a weekly basis.

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Monthly Rentals by Day of Week

# COMMAND ----------

monthly_day_weekday = station_trips.groupBy(month("started_at").alias("month"), dayofweek("started_at").alias("day_of_week")) \
                        .agg(count("*").alias("rental_count")) \
                        .withColumn("month", col("month").cast(IntegerType())) \
                        .orderBy("month", "day_of_week")

monthly_day_weekday_pd = monthly_day_weekday.toPandas()
weekday_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
monthly_weekly_pivot = monthly_day_weekday_pd.pivot("month", "day_of_week", "rental_count")
monthly_weekly_pivot.columns = weekday_labels
monthly_weekly_pivot = monthly_weekly_pivot.reindex([1,2,3,4,5,6,7,8,9,10,11,12])

plt.figure(figsize=(12,8))
sns.heatmap(monthly_weekly_pivot, cmap="YlGnBu")
plt.title("Day Of Week Rentals by Month")
plt.xlabel("Day Of Week")
plt.ylabel("Month")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Monthly Rentals by Hour

# COMMAND ----------

monthly_hourly = station_trips.groupBy(month("started_at").alias("month"), hour("started_at").alias("hour")) \
                    .agg(count("*").alias("rental_count")) \
                    .withColumn("month", col("month").cast(IntegerType())) \
                    .orderBy("month", "hour")

monthly_hourly_pd = monthly_hourly.toPandas()
monthly_hourly_pivot = monthly_hourly_pd.pivot("month", "hour", "rental_count")
monthly_hourly_pivot = monthly_hourly_pivot.reindex([1,2,3,4,5,6,7,8,9,10,11,12])

plt.figure(figsize=(12,8))
sns.heatmap(monthly_hourly_pivot, cmap="YlGnBu")
plt.title("Hourly Rentals by Month")
plt.xlabel("Hour of Day")
plt.ylabel("Month")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC  When looking at it by day of week, we can see that March on the weekdays (specifically Wednesday, Thursday, and Friday) have more bike trips. Therefore, we can conclude that the weather getting better leads to more people wanting to ride their bike. When looking at it on an hourly basis, we can see that evenings are still very active. It's weird that in both December and March that 11:00 PM is really active, so we will look more into that.

# COMMAND ----------

from pyspark.sql.functions import month, hour

late_march = (month("started_at") == 3) & ((hour("started_at") == 22) | (hour("started_at") == 23))
late_december = (month("started_at") == 12) & (hour("started_at") == 23)
combined_late = late_march | late_december
late_df = station_trips.filter(combined_late)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Average Duration of Trips by Month

# COMMAND ----------



late_df_duration = (
    late_df
    .groupBy(month("started_at").alias("month"))
    .agg(avg("trip_duration").alias("avg_duration"))
    .orderBy("month")
    .toPandas()
)

sns.barplot(data=late_df_duration, x="month", y="avg_duration", color="blue")
plt.title("Average Duration of Trips by Month")
plt.xlabel("Month")
plt.ylabel("Duration (seconds)")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Average Temperature by Month

# COMMAND ----------

late_df_temp = (
    late_df
    .groupBy(month("started_at").alias("month"))
    .agg(avg("tempF").alias("avg_tempF"))
    .orderBy("month")
    .toPandas()
)

sns.barplot(data=late_df_temp, x="month", y="avg_tempF", color="red")
plt.title("Average Temperature by Month")
plt.xlabel("Month")
plt.ylabel("Temperature (F)")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Trip Count by Weather Condition

# COMMAND ----------


condition_duration = (
    late_df
    .groupBy("main")
    .agg(F.count("rideable_type").alias("trip_count"))
    .orderBy("trip_count", ascending=False)
    .toPandas()
)
sns.barplot(x="main", y="trip_count", data=condition_duration)
plt.title("Trip Count by Weather Condition")
plt.xticks(rotation=45, ha='right')
plt.show()



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Trip Count by User Type

# COMMAND ----------


condition_duration = (
    late_df
    .groupBy("member_casual")
    .agg(F.count("rideable_type").alias("trip_count"))
    .orderBy("trip_count", ascending=False)
    .toPandas()
)
sns.barplot(x="member_casual", y="trip_count", data=condition_duration)
plt.title("Trip Count by User Type")
plt.xticks(rotation=45, ha='right')
plt.show()



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC There isn't anything out of the ordinary, which leads me to believe that really late nights are popular for bike riding, due to factors out of the control of the dataset. Perhaps there's good entertainment/food options that cause people to be out late. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # What are the daily trip trends for your given station?
# MAGIC
# MAGIC We are going to look at the trip_counts, and average duration through different two different hues (member/casual and weekday/weekend) grouped by date, hour, dayofweek.

# COMMAND ----------

from pyspark.sql.functions import date_format
from pyspark.sql.functions import date_format, dayofweek, avg

daily_trips = (
    station_trips
    .groupBy(date_format("started_at", "yyyy-MM-dd").alias("date"), hour("started_at").alias("hour"), dayofweek("started_at").alias("day_of_week"), "member_casual")
    .agg(count("*").alias("trip_count"), avg("trip_duration").alias("avg_duration"))
    .withColumn("day_type", when(col("day_of_week").isin([1,7]), "weekend").otherwise("weekday"))
    .orderBy("date")
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Trip Count by Date 

# COMMAND ----------

sns.lineplot(data=daily_trips.toPandas(), x="date", y="trip_count")
plt.title("Trip Count by Date - Weekend vs. Weekday")
plt.xlabel("Date")
plt.ylabel("Trip Count")
plt.show()


# COMMAND ----------

# MAGIC %md ## Trip Count by Hour

# COMMAND ----------

sns.barplot(data=daily_trips.toPandas(), x="hour", y="trip_count")
plt.title("Trip Count by Hour")
plt.xlabel("Hour")
plt.ylabel("Trip Count")
plt.show()


# COMMAND ----------

# MAGIC %md ##Trip Count by Day of Week

# COMMAND ----------

sns.barplot(data=daily_trips.toPandas(), x="day_of_week", y="trip_count")
plt.title("Trip Count by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Trip Count")
plt.show()


# COMMAND ----------

# MAGIC %md ## Trip Count by Date - Weekend vs. Weekday

# COMMAND ----------

sns.lineplot(data=daily_trips.toPandas(), x="date", y="trip_count", hue="day_type")
plt.title("Trip Count by Date - Weekend vs. Weekday")
plt.xlabel("Date")
plt.ylabel("Trip Count")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Trip Count by Hour - Weekend vs. Weekday

# COMMAND ----------

sns.barplot(data=daily_trips.toPandas(), x="hour", y="trip_count", hue="day_type")
plt.title("Trip Count by Hour - Weekend vs. Weekday")
plt.xlabel("Hour")
plt.ylabel("Trip Count")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Trip Count by Date - Member vs. Casual

# COMMAND ----------

sns.lineplot(data=daily_trips.toPandas(), x="date", y="trip_count", hue="member_casual")
plt.title("Trip Count by Date - Member vs. Casual")
plt.xlabel("Date")
plt.ylabel("Trip Count")
plt.show()


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Trip Count by Hour - Member vs. Casual

# COMMAND ----------

sns.barplot(data=daily_trips.toPandas(), x="hour", y="trip_count", hue="member_casual")
plt.title("Trip Count by Hour - Member vs. Casual")
plt.xlabel("Hour")
plt.ylabel("Trip Count")
plt.show()


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Trip Count by Day of Week - Member vs. Casual

# COMMAND ----------

sns.barplot(data=daily_trips.toPandas(), x="day_of_week", y="trip_count", hue="member_casual")
plt.title("Trip Count by Day of Week - Member vs. Casual")
plt.xlabel("Day of Week")
plt.ylabel("Trip Count")
plt.show()


# COMMAND ----------

# MAGIC %md ## Average Duration by Date

# COMMAND ----------

sns.lineplot(data=daily_trips.toPandas(), x="date", y="avg_duration")
plt.title("Average Duration by Date")
plt.xlabel("Date")
plt.ylabel("Avg Duration (minutes)")
plt.show()


# COMMAND ----------

# MAGIC %md ## Average Duration by Hour

# COMMAND ----------

sns.barplot(data=daily_trips.toPandas(), x="hour", y="avg_duration")
plt.title("Average Duration by Hour")
plt.xlabel("Hour")
plt.ylabel("Avg Duration (minutes)")
plt.show()


# COMMAND ----------

# MAGIC %md ## Average Duration by Day of Week

# COMMAND ----------

sns.barplot(data=daily_trips.toPandas(), x="hour", y="avg_duration")
plt.title("Average Duration by Hour")
plt.xlabel("Hour")
plt.ylabel("Avg Duration (minutes)")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Average Duration by Date - Weekend vs. Weekday

# COMMAND ----------

sns.lineplot(data=daily_trips.toPandas(), x="date", y="avg_duration", hue="day_type")
plt.title("Average Duration by Date - Weekend vs. Weekday")
plt.xlabel("Date")
plt.ylabel("Avg Duration (minutes)")
plt.show()


# COMMAND ----------

# MAGIC %md ## Average Duration by Hour - Weekend vs. Weekday

# COMMAND ----------

sns.barplot(data=daily_trips.toPandas(), x="hour", y="avg_duration", hue="day_type")
plt.title("Average Duration by Hour - Weekend vs. Weekday")
plt.xlabel("Hour")
plt.ylabel("Avg Duration (minutes)")
plt.show()


# COMMAND ----------

# MAGIC %md ## Average Duration by Date and Member/Casual

# COMMAND ----------

sns.lineplot(data=daily_trips.toPandas(), x="date", y="avg_duration", hue="member_casual")
plt.title("Average Duration by Date and Member/Casual")
plt.xlabel("Date")
plt.ylabel("Duration (minutes)")
plt.show()


# COMMAND ----------

# MAGIC %md ##Average Duration by Hour and Member/Casual

# COMMAND ----------

sns.barplot(data=daily_trips.toPandas(), x="hour", y="avg_duration", hue="member_casual")
plt.title("Average Duration by Hour and Member/Casual")
plt.xlabel("Hour")
plt.ylabel("Duration (minutes)")
plt.show()



# COMMAND ----------

# MAGIC %md ## Average Duration by Day of Week and Member/Casual

# COMMAND ----------

sns.barplot(data=daily_trips.toPandas(), x="day_of_week", y="avg_duration", hue="member_casual")
plt.title("Average Duration by Day of Week and Member/Casual")
plt.xlabel("Day of Week")
plt.ylabel("Duration (minutes)")
plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Weekdays generally have more trip counts than weekends. Evenings and even early mornings are very active, even on the weekends. Members overall have more trip counts than casuals. Members follow the hourly pattern that the overall dataset follows, while casual technically follows it as well, but there's a lot less casuals overall. Weekdays have more members (and overall trips), while casuals are even across the week. 
# MAGIC
# MAGIC Looking at the average duration by both weekend and weekday, both generally follow the same patterns. However, when look at it by hour, there are two major differences on the weekend: 5AM and 10AM has more bike rides. This could be because people just have extra free time for leisure bike riding. In terms of casual and member, overall, there are signficantly longer bike rides by casuals. When look at it by hour, casuals have longer bike rides early in the morning, which could add to the theory of more leisure bike rides by casuals. Day of Week seems to disagree with this a bit as there are longer durations on weekdays, but it's still overall longer than members. Perhaps, they are both using it for work and leisure, but more for leisure, while members are using it more for getting to work.

# COMMAND ----------

# MAGIC %md
# MAGIC ## How does a holiday affect the daily (non-holiday) system use trend?
# MAGIC
# MAGIC First, we are going to do a basic comparison of holidays vs. non-holidays.  

# COMMAND ----------

# MAGIC %md ## Count of Trips by Is_Holiday

# COMMAND ----------

count_by_user_type = (
    station_trips
    .groupBy("is_holiday")
    .agg(F.count("rideable_type").alias("trip_count"))
    .orderBy("is_holiday")
    .toPandas()
)
sns.barplot(x="is_holiday", y="trip_count", data=count_by_user_type)
plt.title("Count of Trips by Is_Holiday")
plt.show()


# COMMAND ----------

is_holiday2 = (station_trips
                    .groupBy('started_at_date', hour("started_at").alias("hour"), 'is_holiday')
                    .agg({'ride_id': 'count', 'trip_duration': 'avg'})
                    .withColumnRenamed('count(ride_id)', 'trip_count')
                    .withColumnRenamed('avg(trip_duration)', 'avg_duration')
                    .orderBy(col('started_at_date'))
                    .toPandas())


sns.lineplot(data=is_holiday2, x="started_at_date", y="trip_count", hue='is_holiday')
plt.title("Total Trips by Date by Holiday")
plt.xlabel("Date")
plt.ylabel("Count of Rentals")
plt.xticks(rotation=45)
plt.show()

sns.lineplot(data=is_holiday2, x="started_at_date", y="avg_duration", hue='is_holiday')
plt.title("Average Duration by Date by Holiday")
plt.xlabel("Date")
plt.ylabel("Average Trip Duration")
plt.xticks(rotation=45)
plt.show()

sns.lineplot(data=is_holiday2, x="hour", y="trip_count", hue='is_holiday')
plt.title("Total Trips by Hour by Holiday")
plt.xlabel("Hour")
plt.ylabel("Count of Rentals")
plt.xticks(range(0, 24))
plt.show()

sns.lineplot(data=is_holiday2, x="hour", y="avg_duration", hue='is_holiday')
plt.title("Average Duration by Hour by Holiday")
plt.xlabel("Hour")
plt.ylabel("Average Trip Duration")
plt.xticks(range(0, 24))
plt.show()



# COMMAND ----------

# MAGIC %md 
# MAGIC We can see that people don't use the system on holidays. However, we are able to see that the amount of trips is generally higher by non-holidays. Additionally, when look at the trip duration by date, if you look at the actual holidays, the trip durations are longer. Lastly, the trip duration by hour shows that people are biking in the afternoon more than on non-holidays. This leads me to believe that  when they do bike, it's most likely for leisure. Now let's filter by holiday, and try to dive deeper.

# COMMAND ----------

holiday_trips = (station_trips
                    .filter(station_trips.is_holiday == True)
                    .groupBy('started_at_date', hour("started_at").alias("hour"), 'member_casual', 'tempF')
                    .agg({'ride_id': 'count', 'trip_duration': 'avg'})
                    .withColumnRenamed('count(ride_id)', 'trip_count')
                    .withColumnRenamed('avg(trip_duration)', 'avg_duration')
                    .orderBy(col('started_at_date'))
                    .toPandas())


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Trip Count by Date During Holidays

# COMMAND ----------

sns.lineplot(data=holiday_trips, x='started_at_date', y='trip_count')
plt.title('Trip Count by Date During Holidays')
plt.xlabel('Date')
plt.ylabel('Trip count')
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Trip Count by Hour During Holidays

# COMMAND ----------

sns.lineplot(data=holiday_trips, x='hour', y='trip_count')
plt.title('Trip Count by Hour During Holidays')
plt.xlabel('Hour')
plt.ylabel('Trip count')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Trip Count by Date - Member vs. Casual

# COMMAND ----------

sns.lineplot(data=holiday_trips, x='started_at_date', y='trip_count', hue='member_casual')
plt.title('Trip Count by Date - Member vs. Casual')
plt.xlabel('Hour')
plt.ylabel('Trip count')
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Trip Count by Hour - Member vs. Casual

# COMMAND ----------

sns.lineplot(data=holiday_trips, x='hour', y='trip_count', hue='member_casual')
plt.title('Trip Count by Hour - Member vs. Casual')
plt.xlabel('Hour')
plt.ylabel('Trip count')
plt.xticks(range(0, 24))
plt.show()


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Average trip duration by date during holidays

# COMMAND ----------

sns.lineplot(data=holiday_trips, x='started_at_date', y='avg_duration')
plt.title('Average trip duration by date during holidays')
plt.xlabel('Date')
plt.ylabel('Average trip duration (minutes)')
plt.xticks(rotation=45)
plt.show()



# COMMAND ----------

# MAGIC %md ## Average trip duration by Hour during holidays

# COMMAND ----------

sns.lineplot(data=holiday_trips, x='hour', y='avg_duration')
plt.title('Average trip duration by date during holidays')
plt.xlabel('Hour')
plt.ylabel('Average trip duration (minutes)')
plt.xticks(range(0, 24))
plt.show()



# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Average Duration by Date - Member vs. Casual

# COMMAND ----------

sns.lineplot(data=holiday_trips, x='started_at_date', y='avg_duration', hue='member_casual')
plt.title('Average Duration by Date - Member vs. Casual')
plt.xlabel('Date')
plt.ylabel('Average trip duration (minutes)')
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------

# MAGIC %md ## Average Duration by Hour - Member vs. Casual

# COMMAND ----------

sns.lineplot(data=holiday_trips, x='hour', y='avg_duration', hue='member_casual')
plt.title('%md ## Average Duration by Hour - Member vs. Casual')
plt.xlabel('Hour')
plt.ylabel('Average trip duration (minutes)')
plt.xticks(range(0, 24))
plt.show()


# COMMAND ----------

# MAGIC %md ## Average Temperature by date during holidays

# COMMAND ----------

sns.lineplot(data=holiday_trips, x='started_at_date', y='tempF')
plt.title('Average Temperature by date during holidays')
plt.xlabel('Date')
plt.ylabel('Temperature (F)')
plt.xticks(rotation=45)
plt.show()



# COMMAND ----------

# MAGIC %md ## Average Temperature by hour during holidays

# COMMAND ----------

sns.lineplot(data=holiday_trips, x='hour', y='tempF')
plt.title('Average Temperature by hour during holidays')
plt.xlabel('Hour')
plt.ylabel('Temperature (F)')
plt.xticks(range(0, 24))
plt.show()



# COMMAND ----------

# MAGIC %md 
# MAGIC On most holidays except for July 4th, bike trips are lower. Only about 7.5 bike trips per day. Additionally, when looking by hour, we can see that afternoon and evening have the most bike trips. Next we can see that the member user type uses the bike system on holidays rather than casual members. When look at the user type by hour, we can see the same trend that we saw previously. 
# MAGIC
# MAGIC When looking at average duration, the avearge duration is around the average, if not a bit lower, except for July 4th which has the duration go up a lot higher. When looking by hour,  around 5AM is the the peak.  Casuals tend to have longer durations on both July 4th and around 5AM. It seems like this group rents bikes for an activity like a July 4th race or just for fun.
# MAGIC
# MAGIC Lastly, we decided to take a look at the temperature and nothing is out of the ordinary.

# COMMAND ----------

display(daily_trips)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## How does weather affect the daily/hourly trend of system use?

# COMMAND ----------

from pyspark.sql.functions import date_format, hour

daily_trips = (station_trips
               .groupBy(date_format('started_at', 'yyyy-MM-dd').alias('date'),
                        hour('started_at').alias('hour'), dayofweek("started_at").alias("day_of_week"),
                        'main', 'description')
               .agg({'ride_id': 'count',
                     'tempF': 'mean',
                     'wind_speed_mph': 'mean',
                     'pop': 'mean',
                     'humidity': 'mean',
                     'snow_1h': 'mean',
                     'rain_1h': 'mean',
                     'trip_duration': 'mean'})
               .withColumnRenamed('count(ride_id)', 'num_trips')
               .withColumnRenamed('avg(tempF)', 'avg_tempF')
               .withColumnRenamed('avg(wind_speed_mph)', 'avg_wind_speed_mph')
               .withColumnRenamed('avg(pop)', 'avg_pop')
               .withColumnRenamed('avg(humidity)', 'avg_humidity')
               .withColumnRenamed('avg(snow_1h)', 'avg_snow_1h')
               .withColumnRenamed('avg(rain_1h)', 'avg_rain_1h')
               .withColumnRenamed('avg(trip_duration)', 'avg_trip_duration')
               .toPandas())


# COMMAND ----------

# MAGIC %md ## Heatmap
# MAGIC
# MAGIC Let's first look at the correlations between all of the numerical columns.

# COMMAND ----------

df_pandas = daily_trips
corr_matrix = df_pandas.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
plt.show()


# COMMAND ----------

# MAGIC %md Interesting correlations are: avg_humidity and num_trips is negative, num_trips and avg_tempF is positively correlated, avg_wind_speed and num_trips are positively correlated. Now I am going to visualize these metrics over timeseries.

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ##Average temperature and number of trips per day

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(30, 8)) 
ax1 = sns.lineplot(x='date', y='avg_tempF', data=daily_trips, label='Temperature (F)')
ax1.set_xticks(ax1.get_xticks()[::90]) 

ax2 = ax1.twinx()
sns.lineplot(x='date', y='num_trips', data=daily_trips, ax=ax2, color='r', label='Num Trips')

plt.title('Average temperature and number of trips per day')
plt.xlabel('Date')
ax1.set_ylabel('Temperature (F)')
ax2.set_ylabel('Number of trips')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Average temperature and number of trips per hour

# COMMAND ----------

plt.figure(figsize=(15, 8)) 
ax1 = sns.lineplot(x='hour', y='avg_tempF', data=daily_trips)
ax1.set_xticks(ax1.get_xticks()[::45]) 

ax2 = ax1.twinx()
sns.lineplot(x='hour', y='num_trips', data=daily_trips, ax=ax2, color='r')

plt.title('Average temperature and number of trips per hour')
plt.xlabel('Hour')
ax1.set_ylabel('Temperature (F)')
ax2.set_ylabel('Number of trips')
ax.legend(labels=['Temperature (F)'], loc='upper left')
ax2.legend(labels=['Num Trips'], loc='upper right')
plt.show()





# COMMAND ----------

# MAGIC %md ##Average wind speed and number of trips per day

# COMMAND ----------

plt.figure(figsize=(30, 8)) 
ax1 = sns.lineplot(x='date', y='avg_wind_speed_mph', data=daily_trips, label='Wind Speed (mph)')
ax1.set_xticks(ax1.get_xticks()[::45]) 

ax2 = ax1.twinx()
sns.lineplot(x='date', y='num_trips', data=daily_trips, ax=ax2, color='r', label='Num Trips')

plt.title('Average wind speed and number of trips per day')
plt.xlabel('Date')
ax1.set_ylabel('Wind Speed (mph)')
ax2.set_ylabel('Number of trips')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()


# COMMAND ----------

# MAGIC %md ##Average wind speed and number of trips per hour

# COMMAND ----------

plt.figure(figsize=(15, 8)) 
ax1 = sns.lineplot(x='hour', y='avg_wind_speed_mph', data=daily_trips)
ax1.set_xticks(ax1.get_xticks()[::45]) 

ax2 = ax1.twinx()
sns.lineplot(x='hour', y='num_trips', data=daily_trips, ax=ax2, color='r')

plt.title('Average wind speed and number of trips per hour')
plt.xlabel('Hour')
ax1.set_ylabel('Wind Speed (mph)')
ax2.set_ylabel('Number of trips')
ax1.legend(labels=['Wind Speed (mph)'], loc='upper left')
ax2.legend(labels=['Num Trips'], loc='upper right')

plt.show()





# COMMAND ----------

# MAGIC %md ##Probability of Precipitation	 and Number of Trips per hour

# COMMAND ----------

plt.figure(figsize=(30, 8)) 
ax1 = sns.lineplot(x='date', y='avg_pop', data=daily_trips, label='Probability of Precipitation')
ax1.set_xticks(ax1.get_xticks()[::45]) 

ax2 = ax1.twinx()
sns.lineplot(x='date', y='num_trips', data=daily_trips, ax=ax2, color='r', label='Num Trips')

plt.title('Average Probability of Precipitation	 and Number of Trips per hour')
plt.xlabel('Date')
ax1.set_ylabel('Probability of Precipitation')
ax2.set_ylabel('Number of trips')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()


# COMMAND ----------

# MAGIC %md ##Probability of Precipitation and Number of Trips per hour

# COMMAND ----------

plt.figure(figsize=(15, 8)) 
ax1 = sns.lineplot(x='hour', y='avg_pop', data=daily_trips)
ax1.set_xticks(ax1.get_xticks()[::45]) 

ax2 = ax1.twinx()
sns.lineplot(x='hour', y='num_trips', data=daily_trips, ax=ax2, color='r')

plt.title('Average Probability of Precipitation and Number of Trips per hour')
plt.xlabel('Hour')
ax1.set_ylabel('Probability of Precipitation')
ax2.set_ylabel('Number of trips')
ax1.legend(labels=['Population Density'], loc='upper left')
ax2.legend(labels=['Num Trips'], loc='upper right')

plt.show()


# COMMAND ----------

# MAGIC %md ##Average humidity and number of trips per day

# COMMAND ----------


plt.figure(figsize=(30, 8)) 
ax1 = sns.lineplot(x='date', y='avg_humidity', data=daily_trips, label='Humidity')
ax1.set_xticks(ax1.get_xticks()[::45]) 

ax2 = ax1.twinx()
sns.lineplot(x='date', y='num_trips', data=daily_trips, ax=ax2, color='r', label='Num Trips')

plt.title('Average humidity and number of trips per day')
plt.xlabel('Date')
ax1.set_ylabel('Humidity')
ax2.set_ylabel('Number of trips')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()



# COMMAND ----------

# MAGIC %md ##Average humidity and number of trips per hour

# COMMAND ----------

plt.figure(figsize=(15, 8)) 
ax1 = sns.lineplot(x='hour', y='avg_humidity', data=daily_trips)
ax1.set_xticks(ax1.get_xticks()[::45]) 

ax2 = ax1.twinx()
sns.lineplot(x='hour', y='num_trips', data=daily_trips, ax=ax2, color='r')

plt.title('Average humidity and number of trips per hour')
plt.xlabel('Hour')
ax1.set_ylabel('Humidity')
ax2.set_ylabel('Number of trips')
ax1.legend(labels=['Humidity'], loc='upper left')
ax2.legend(labels=['Num Trips'], loc='upper right')

plt.show()


# COMMAND ----------

# MAGIC %md ##Average snowfall and number of trips per day

# COMMAND ----------

plt.figure(figsize=(30, 8)) 
ax1 = sns.lineplot(x='date', y='avg_snow_1h', data=daily_trips, label='Snow (mm)')
ax1.set_xticks(ax1.get_xticks()[::45])

ax2 = ax1.twinx()
sns.lineplot(x='date', y='num_trips', data=daily_trips, ax=ax2, color='r', label='Num Trips')

plt.title('Average snowfall and number of trips per day')
plt.xlabel('Date')
ax1.set_ylabel('Snow (mm)')
ax2.set_ylabel('Number of trips')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()



# COMMAND ----------

# MAGIC %md ##Average snowfall and number of trips per hour

# COMMAND ----------

plt.figure(figsize=(15, 8))
ax1 = sns.lineplot(x='hour', y='avg_snow_1h', data=daily_trips)
ax1.set_xticks(ax1.get_xticks()[::45])

ax2 = ax1.twinx()
sns.lineplot(x='hour', y='num_trips', data=daily_trips, ax=ax2, color='r')

plt.title('Average snowfall and number of trips per hour')
plt.xlabel('Hour')
ax1.set_ylabel('Snow (mm)')
ax2.set_ylabel('Number of trips')
ax1.legend(labels=['Snow (mm)'], loc='upper left')
ax2.legend(labels=['Num Trips'], loc='upper right')

plt.show()






# COMMAND ----------

# MAGIC %md ##Rainfall and number of trips per day

# COMMAND ----------

plt.figure(figsize=(30, 8)) 
ax1 = sns.lineplot(x='date', y='avg_rain_1h', data=daily_trips, label='Rain (mm)')
ax1.set_xticks(ax1.get_xticks()[::45]) 

ax2 = ax1.twinx()
sns.lineplot(x='date', y='num_trips', data=daily_trips, ax=ax2, color='r', label='Num Trips')

plt.title('Average Rainfall and number of trips per day')
plt.xlabel('Date')
ax1.set_ylabel('Rain (mm)')
ax2.set_ylabel('Number of trips')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()

# COMMAND ----------

# MAGIC %md ## Rainfall and number of trips per hour

# COMMAND ----------

plt.figure(figsize=(15, 8)) 
ax1 = sns.lineplot(x='hour', y='avg_rain_1h', data=daily_trips)
ax1.set_xticks(ax1.get_xticks()[::45]) 

ax2 = ax1.twinx()
sns.lineplot(x='hour', y='num_trips', data=daily_trips, ax=ax2, color='r')

plt.title('Average Rainfall and number of trips per hour')
plt.xlabel('Hour')
ax1.set_ylabel('Rain (mm)')
ax2.set_ylabel('Number of trips')
ax1.legend(labels=['Rain (mm)'], loc='upper left')
ax2.legend(labels=['Num Trips'], loc='upper right')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Note: The date timeseries starts in November 2021 and ends in March 2023.
# MAGIC
# MAGIC When looking at weather patterns, we first decided to examine the relationship between temperature (F) and the number of trips. Generally, they follow the same pattern except for the start period. When we look at the hour, the number of trips falls significantly in the morning while the temperature stays around 52 degrees Fahrenheit. During the afternoon and evening, both start to rise. For wind speed, it generally follows the same pattern. This is the same per hour, where they both fall in the morning and rise in the evening. Honestly, this is a bit weird because you would think that more wind would lead to fewer bike rides. However, according to the correlation matrix, they are not that correlated. Probability of Precipitation and Number of Rides do not seem to match at all. Humidity and the Number of Bike Rides seem like polar opposites. Both snowfall and rainfall seem to have a significant effect on the number of bikes rented. It's difficult to see, but whenever there is rain or snow, the number of bikes rented seems to decrease.

# COMMAND ----------

from pyspark.sql.functions import count, mean

grouped_by_main = (station_trips
                   .groupBy('main', 'description')
                   .agg(count('ride_id').alias('num_trips'),
                        mean('tempF').alias('avg_tempF'),
                        mean('wind_speed_mph').alias('avg_wind_speed_mph'),
                        mean('pop').alias('avg_pop'),
                        mean('humidity').alias('avg_humidity'),
                        mean('snow_1h').alias('avg_snow_1h'),
                        mean('rain_1h').alias('avg_rain_1h'),
                        mean('trip_duration').alias('avg_trip_duration')))

display(grouped_by_main)




# COMMAND ----------

# MAGIC %md Here we can take a look at the different types of weather patterns that happen. 
