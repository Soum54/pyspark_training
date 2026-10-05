from typing import Tuple, List
from pyspark.sql import SparkSession,DataFrame
from pyspark.sql.types import *
from pyspark.sql.functions import col,unix_timestamp,to_timestamp,when,avg
from pyspark.sql.window import Window

def load_inline_triage_data(spark: SparkSession) -> DataFrame:
    
    data=[("C001","ER","2026-09-01 08:00:00","2026-09-01 08:25:00","High","ACTIVE"),
          ("C002","Trauma","2026-09-01 08:10:00","2026-09-01 08:05:00","Critical","ACTIVE"),
          ("C003","Pediatrics","2026-09-01 09:00:00","2026-09-01 09:45:00","Low","ACTIVE"),
          ("C004","ER","2026-09-01 09:30:00","2026-09-01 10:00:00","High","CLOSED")]
    columns=["case_id","department","arrival_time","doctor_start_time","priority","status"]
    return spark.createDataFrame(data,columns)

def compute_wait_minutes(df: DataFrame) -> DataFrame:
    start=unix_timestamp(to_timestamp(col("doctor_start_time")))
    arrival=unix_timestamp(to_timestamp(col("arrival_time")))
    wait_minutes=(start-arrival)/60
    return df.withColumn("wait_minutes",when(wait_minutes<0,0).otherwise(wait_minutes))

def filter_priority_cases(df: DataFrame) -> DataFrame:
    return df.filter((col("priority").isin("Critical","High")) & (col("wait_minutes").isNotNull()))

def average_wait_by_department(df: DataFrame) -> DataFrame:
    return df.groupBy("department").agg(avg("wait_minutes").alias("avg_wait_minutes"))

def list_active_departments(df: DataFrame) -> List[str]:
    df=df.filter(col("status")=="ACTIVE")
    df=df.select("department").distinct().orderBy("department").collect()
    result=[]
    for i in df:
        result.append(i["department"])
    return result
