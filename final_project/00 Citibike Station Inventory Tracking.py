# Databricks notebook source
# MAGIC %md
# MAGIC ##DSCC 202 - 402 Final Project Spring 2023
# MAGIC <p>
# MAGIC <img src='https://data-science-at-scale.s3.amazonaws.com/images/fp2023.png'>
# MAGIC </p>
# MAGIC see product description and rubric in repo same directory as this notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC # Group 1
# MAGIC Jake Brehm, Josh Wang, Corryn Collins, Varun Arvind, Tessa Charles

# COMMAND ----------

from datetime import datetime as dt
from datetime import timedelta
import json

dbutils.widgets.removeAll()

dbutils.widgets.text('01.promote_model', 'No')

promote_model = str(dbutils.widgets.get('01.promote_model'))

print(promote_model)

# COMMAND ----------

# DBTITLE 1,Run the ETL Notebook
# Run the ETL notebook
result = dbutils.notebook.run("01 etl", 3600, {"01.promote_model": promote_model})

# Check the results
assert json.loads(result)["exit_code"] == "OK", "ETL failed!"

# COMMAND ----------

# DBTITLE 1,Run the EDA Notebook
# Run the EDA notebook
result = dbutils.notebook.run("02 eda", 3600, {"01.promote_model": promote_model})

# Check the results
assert json.loads(result)["exit_code"] == "OK", "EDA failed!"

# COMMAND ----------

# DBTITLE 1,Run Model Development Notebook
# Run the model development notebook
result = dbutils.notebook.run("03 mdl", 3600, {"01.promote_model": promote_model})

# Check the results
assert json.loads(result)["exit_code"] == "OK", "Model development failed!"

# COMMAND ----------

# DBTITLE 1,Run Station Inventory Forecast Notebook
# Run the application notebook
result = dbutils.notebook.run("04 app", 3600, {"01.promote_model": promote_model})

# Check the results
assert json.loads(result)["exit_code"] == "OK", "Application failed!"
