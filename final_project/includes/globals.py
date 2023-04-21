# Databricks notebook source
# MAGIC %run ./includes

# COMMAND ----------

import os

# COMMAND ----------

spark.conf.set("spark.sql.session.timeZone", "America/New_York")

# COMMAND ----------

def construct_nested_group_path(*args: str) -> str:
    """Constructs a path on top of the group data path using the input strings."""
    return os.path.join(GROUP_DATA_PATH, *args)

# COMMAND ----------

# Define paths for the two new bronze tables
BRONZE_STATION_HISTORY_PATH = construct_nested_group_path("bronze_station_history.delta")
BRONZE_WEATHER_HISTORY_PATH = construct_nested_group_path("bronze_weather_history.delta")

# Define paths for the five new silver tables
SILVER_STATION_INFO_PATH = construct_nested_group_path("silver_station_info.delta")
SILVER_STATION_STATUS_PATH = construct_nested_group_path("silver_station_status.delta")
SILVER_NYC_WEATHER_PATH = construct_nested_group_path("silver_nyc_weather.delta")
SILVER_STATION_HISTORY_PATH = construct_nested_group_path("silver_station_history.delta")
SILVER_WEATHER_HISTORY_PATH = construct_nested_group_path("silver_weather_history.delta")
SILVER_HISTORICAL_PATH = construct_nested_group_path("silver_historical.delta")

# Define the directories that will store checkpoints
BRONZE_STATION_HISTORY_CHECKPOINTS = construct_nested_group_path("_checkpoints", "bronze_station_history")
BRONZE_WEATHER_HISTORY_CHECKPOINTS = construct_nested_group_path("_checkpoints", "bronze_weather_history")
SILVER_HISTORICAL_CHECKPOINTS = construct_nested_group_path("_checkpoints", "silver_historical")

# COMMAND ----------

rows = [
    ["BRONZE_WEATHER_HISTORY_PATH", BRONZE_WEATHER_HISTORY_PATH, "Historical NYC Weather (bronze)"],
    ["BRONZE_STATION_HISTORY_PATH", BRONZE_STATION_HISTORY_PATH, "Historical Station Data (bronze)"],
    ["SILVER_STATION_INFO_PATH", SILVER_STATION_INFO_PATH, "Station Info (silver)"],
    ["SILVER_STATION_STATUS_PATH", SILVER_STATION_STATUS_PATH, "Station Status (silver)"],
    ["SILVER_NYC_WEATHER_PATH", SILVER_NYC_WEATHER_PATH, "NYC Weather (silver)"],
    ["SILVER_STATION_HISTORY_PATH", SILVER_STATION_HISTORY_PATH, "Historical NYC Weather (silver)"],
    ["SILVER_WEATHER_HISTORY_PATH", SILVER_WEATHER_HISTORY_PATH, "Historical Station Data (silver)"],
    ["SILVER_HISTORICAL_PATH", SILVER_HISTORICAL_PATH, "Historical Data (silver)"],
    ["BRONZE_STATION_HISTORY_CHECKPOINTS", BRONZE_STATION_HISTORY_CHECKPOINTS, "Historical Station Checkpoints (bronze)"],
    ["BRONZE_WEATHER_HISTORY_CHECKPOINTS", BRONZE_WEATHER_HISTORY_CHECKPOINTS, "Historical Weather Checkpoints (bronze)"],
    ["SILVER_HISTORICAL_CHECKPOINTS", SILVER_HISTORICAL_CHECKPOINTS, "Historical Checkpoints (silver)"],
]

# COMMAND ----------

rows_HTML = [f"<tr><td>{name}</td><td>{value}</td><td>{description}</td></tr>" for name, value, description in rows]

# COMMAND ----------

displayHTML(f"""
<H2>The following are additional group-specific global variables.</H2>
<table border=1>
<tr><td><b>Variable Name</b></td><td><b>Value</b></td><td><b>Description</b></td></tr>
{''.join(rows_HTML)}
</table>
""")
