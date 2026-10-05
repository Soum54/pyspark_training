from typing import Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import to_timestamp,col,unix_timestamp,when,avg,sum,count
from pyspark.sql.types import DoubleType

def load_delivery_trips(spark: SparkSession, path: str) -> DataFrame:
    df=spark.read.csv(path,header=True,inferSchema=True)
    df=(df.withColumn("dispatch_ts",to_timestamp("dispatch_ts"))
        .withColumn("promised_ts",to_timestamp("promised_ts"))
        .withColumn("actual_ts",to_timestamp("actual_ts"))
    )
    df=(df.withColumn("distance_km",col("distance_km").cast(DoubleType()))
        .withColumn("fare_amount",col("fare_amount"))
    )
    return df

def compute_trip_metrics(df: DataFrame) -> DataFrame:
    delay_hours=(unix_timestamp("actual_ts")-unix_timestamp("promised_ts"))/3600
    df=df.withColumn("delay_hours",when(delay_hours<0,0.0).otherwise(delay_hours))
    trip_hours=(unix_timestamp("actual_ts")-unix_timestamp("dispatch_ts"))/3600
    df=df.withColumn("trip_hours",when(trip_hours<0,0.0).otherwise(trip_hours))
    df=df.withColumn("avg_speed_kmph",when(trip_hours>0,col("distance_km")/col("trip_hours")).otherwise(0.0))
    return df

def filter_significant_delays(
    df: DataFrame,
    delay_threshold: float,
    min_distance: float
) -> DataFrame:
    df=df.filter(
        (col("delay_hours")>delay_threshold) &
        (col("distance_km")>=min_distance)
    )
    return df

def partner_delay_summary(df: DataFrame) -> DataFrame:
    df=(df.filter(col("status")=="Completed").groupBy("partner")
        .agg(sum("delay_hours").alias("total_delay_hours"),
             avg("fare_amount").alias("avg_fare"),
             count("trip_id").alias("trip_count"))
    )
    return df.select("partner","total_delay_hours","avg_fare","trip_count")

def highest_average_delay_reason(df: DataFrame) -> Tuple[str, float]:
    df=df.filter((col("delay_reason").isNotNull()) &
                 (col("delay_hours").isNotNull())
    )
    df=df.groupBy("delay_reason").agg(avg("delay_hours").alias("avg_delay_hours")).orderBy(col("avg_delay_hours").desc(),col("delay_reason").asc()).first()
    if df is None:
        return ("",0.0)
    return (df["delay_reason"],df["avg_delay_hours"])