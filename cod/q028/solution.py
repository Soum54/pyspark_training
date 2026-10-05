from typing import Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col,to_date,date_trunc,sum

def load_battery_swap_data(spark: SparkSession, path: str) -> DataFrame:
    df=spark.read.csv(path,header=True,inferSchema=True)
    return df.withColumn("swap_date",to_date("swap_date"))

def add_swap_month(df: DataFrame) -> DataFrame:
    return df.withColumn("swap_month",date_trunc("month",col("swap_date")))

def filter_valid_swaps(df: DataFrame) -> DataFrame:
    return df.filter((col("energy_kwh")>=0) &
                (col("battery_type").isNotNull()) )

def monthly_battery_energy(df: DataFrame) -> DataFrame:
    df=df.groupBy("battery_type","swap_month").agg(sum("energy_kwh").alias("total_energy_kwh"))
    return df.select("battery_type","swap_month","total_energy_kwh")

def top_battery_type(df: DataFrame) -> Tuple[str, float]:
    df=df.groupBy("battery_type").agg(sum("total_energy_kwh").alias("total_energy_kwh")).orderBy(col("total_energy_kwh").desc(),col("battery_type").asc()).first()
    if df is None:
        return ("",0.0)
    return (df["battery_type"],float(df["total_energy_kwh"]))
