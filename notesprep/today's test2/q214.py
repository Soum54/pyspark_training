from pyspark.sql import SparkSession,DataFrame
from pyspark.sql.functions import col,to_date,date_trunc,sum
from pyspark.sql.types import *
from typing import List,Tuple

def load_irrigation_data(spark:SparkSession,path:str)->DataFrame:
    df=spark.read.csv(path,header=True,inferSchema=True)
    df=df.withColumn("irrigation_date",to_date("irrigation_date"))
    return df

def with_month(df:DataFrame)->DataFrame:
    df=df.withColumn("month",date_trunc("month",col("irrigation_date")))
    df=df.withColumn("month",to_date("month"))
    return df

def filter_valid(df:DataFrame)->DataFrame:
    df=df.filter(col("liters_used")>=0)
    return df

def monthly_crop_water(df:DataFrame)->DataFrame:
    df=df.groupBy("crop_type","month").agg(sum("liters_used").alias("total_liters"))
    return df

def top_crop_by_month(df:DataFrame)->str:
    df=(df.groupBy("crop_type").agg(sum("total_liters").alias("total_liters"))
        .orderBy(col("total_liters").desc()).first()
    )
    return df["crop_type"]
