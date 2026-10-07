from pyspark.sql import SparkSession,DataFrame
from pyspark.sql.functions import col,count,lit,when,sum
from pyspark.sql.types import *
from typing import List,Tuple

def load_rides_csv(spark:SparkSession,path:str)->DataFrame:
    df=spark.read.csv(path,header=True,inferSchema=True)
    return df

def find_loyal_customers(df:DataFrame,n:int)->DataFrame:
    df=df.groupBy("CustomerID").agg(count("*").alias("number_of_rides"))
    df=df.filter(col("number_of_rides")>=n)
    return df.select("CustomerID")

def apply_discounts(df:DataFrame,loyal_customers:DataFrame)->DataFrame:
    df1=loyal_customers.withColumn("Is_loyal",lit(True))
    df2=df.join(df1,"CustomerID","left")
    df2=df2.withColumn("discount",when(col("Is_loyal")=="True",0.10).otherwise(0.07))
    return df2

def top_three_longest_trips(df:DataFrame)->DataFrame:
    df=df.select("RideId","source","destination","distance").orderBy(col("distance").desc()).limit(3)
    return df

def get_top_earners(df:DataFrame)->DataFrame:
    df=df.groupBy("DriverID").agg(sum("earnings").alias("total_earnings")).limit(3)
    return df
