from pyspark.sql import SparkSession,DataFrame
from pyspark.sql.functions import col,avg,when,sum,unix_timestamp
from pyspark.sql.types import *
from typing import List,Tuple

def load_inline_data(spark:SparkSession)->DataFrame:
  data = [
        ("O1001", "PartnerA", "2025-12-01 10:00:00", "2025-12-01 12:00:00", "Traffic"),
        ("O1002", "PartnerA", "2025-12-01 09:00:00", "2025-12-01 10:30:00", "Traffic"),
        ("O1003", "PartnerB", "2025-12-02 14:00:00", "2025-12-02 18:00:00", "Weather"),
        ("O1004", "PartnerB", "2025-12-02 08:00:00", "2025-12-02 07:00:00", "Restaurant"),
        ("O1005", "PartnerC", "2025-12-03 11:00:00", "2025-12-03 13:00:00", "Restaurant"),
        ("O1006", "PartnerC", "2025-12-03 16:00:00", "2025-12-03 20:00:00", "Traffic"),
    ]
  columns=["order_id","partner","expected_delivery","actual_delivery","delay_reason"]
  df=spark.createDataFrame(data,columns)
  return df

def compute_delay(df:DataFrame)->DataFrame:
  hours=unix_timestamp("actual_delivery")-unix_timestamp("expected_delivery")
  df=df.withColumn("delay_hours",when(hours<0,0).otherwise(hours/3600))
  return df

def get_delayed_orders(df:DataFrame,threshold:float)->DataFrame:
  hours=unix_timestamp("actual_delivery")-unix_timestamp("expected_delivery")
  df=df.withColumn("delay_hours",when(hours<0,0).otherwise(hours/3600))
  df=df.filter(col("delay_hours")>threshold)
  return df

def most_delayed_partner(df:DataFrame)->DataFrame:
  df=df.groupBy("partner").agg(sum("delay_hours").alias("total_delay")).orderBy(col("total_delay").desc()).limit(1)
  return df

def most_common_delay_reason(df:DataFrame)->DataFrame:
  df=df.groupBy("delay_reason").agg(avg("delay_hours").alias("avg_delay")).orderBy(col("avg_delay").desc()).limit(1)
  return df
